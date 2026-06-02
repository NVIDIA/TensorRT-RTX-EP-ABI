# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from python_tests.compile_worker import RegisteredEp, _get_trt_rtx_devices, _register_ep


EXCLUDE_FAST_GELU = {"nv_op_types_to_exclude": "FastGelu"}


def _fast_gelu(x):
    return x * (0.5 + 0.5 * np.tanh(x * (0.035677408136300125 * x * x + 0.7978845608028654)))


def _expected_dq_matmul_fast_gelu(x_data):
    q_weights = np.array([16, -8, 5, 12, -3, 9], dtype=np.int8).astype(np.float32)
    weights = ((q_weights - 3.0) * 0.25).reshape(3, 2)
    return _fast_gelu(x_data.reshape(2, 3) @ weights)


def _expected_qdq_matmul_fast_gelu(x_data):
    weights = np.array(
        [1.25, -0.75, 0.50, 2.00, -1.50, 0.25],
        dtype=np.float32,
    ).reshape(3, 2)
    scale = 0.25
    zero_point = 5
    quantized = np.clip(np.rint(x_data / scale) + zero_point, -128, 127).astype(np.int8)
    dequantized = (quantized.astype(np.float32) - zero_point) * scale
    return _fast_gelu(dequantized.reshape(2, 3) @ weights)


def _make_session_options(
    registered_ep: RegisteredEp,
    provider_options=None,
    profiling_prefix: str | Path | None = None,
):
    ort = registered_ep.ort
    ort.set_default_logger_severity(0)
    so = ort.SessionOptions()
    so.log_verbosity_level = 0

    if profiling_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = str(profiling_prefix)

    if not hasattr(so, "add_provider_for_devices"):
        raise RuntimeError("onnxruntime.SessionOptions does not expose add_provider_for_devices")

    so.add_provider_for_devices(
        _get_trt_rtx_devices(registered_ep),
        {str(k): str(v) for k, v in (provider_options or {}).items()},
    )
    return so


def _create_session(registered_ep: RegisteredEp, model_path: Path, **session_options_kwargs):
    so = _make_session_options(registered_ep, **session_options_kwargs)
    return registered_ep.ort.InferenceSession(str(model_path), sess_options=so)


def _read_profile_text(session) -> str:
    profile_path = Path(session.end_profiling())
    if not profile_path.is_file():
        raise RuntimeError(f"Profiling output not found at {profile_path}")
    return profile_path.read_text(encoding="utf-8", errors="ignore")


def _assert_profile_contains_split(profile, ep_name):
    assert "FastGelu" in profile
    assert "CPUExecutionProvider" in profile
    assert ep_name in profile


def _count_nodes_by_op_type(model, op_type: str, domain: str | None = None) -> int:
    count = 0

    def visit_graph(graph):
        nonlocal count
        for node in graph.node:
            if node.op_type == op_type and (domain is None or node.domain == domain):
                count += 1
            for attr in node.attribute:
                if attr.type == attr.GRAPH:
                    visit_graph(attr.g)
                elif attr.type == attr.GRAPHS:
                    for nested in attr.graphs:
                        visit_graph(nested)

    visit_graph(model.graph)
    return count


def _run_runtime(registered_ep: RegisteredEp, model_path: Path, profile_prefix: Path, qdq: bool):
    session = _create_session(
        registered_ep,
        model_path,
        provider_options=EXCLUDE_FAST_GELU,
        profiling_prefix=profile_prefix,
    )

    if qdq:
        x_data = np.array([1.10, -0.35, 2.30, -1.70, 0.40, 0.95], dtype=np.float32)
        expected = _expected_qdq_matmul_fast_gelu(x_data)
    else:
        x_data = np.array([1.0, 2.0, -1.0, 0.5, -0.25, 3.0], dtype=np.float32)
        expected = _expected_dq_matmul_fast_gelu(x_data)

    outputs = session.run(["O"], {"X": x_data.reshape(2, 3)})
    np.testing.assert_allclose(outputs[0], expected, rtol=0, atol=1e-5)
    _assert_profile_contains_split(_read_profile_text(session), registered_ep.ep_name)


def _run_compile(registered_ep: RegisteredEp, model_path: Path, compiled_path: Path, profile_prefix: Path):
    import onnx

    if not hasattr(registered_ep.ort, "ModelCompiler"):
        raise RuntimeError("onnxruntime build does not expose ModelCompiler")

    so = _make_session_options(registered_ep, provider_options=EXCLUDE_FAST_GELU)
    compiler = registered_ep.ort.ModelCompiler(
        so,
        str(model_path),
        embed_compiled_data_into_model=True,
    )
    compiler.compile_to_file(str(compiled_path))
    if not compiled_path.is_file():
        raise RuntimeError(f"Compiled model not found at {compiled_path}")

    compiled_model = onnx.load(str(compiled_path))
    assert _count_nodes_by_op_type(compiled_model, "EPContext") >= 1
    assert _count_nodes_by_op_type(compiled_model, "FastGelu", "com.microsoft") == 1

    _run_runtime(registered_ep, compiled_path, profile_prefix, qdq=False)


def _run_foreign_source(registered_ep: RegisteredEp, model_path: Path):
    try:
        _create_session(registered_ep, model_path)
    except Exception as exc:
        if "EPContext" not in str(exc):
            raise
        return

    raise AssertionError("Expected foreign-source EPContext model to fail session creation")


def _run_no_source(registered_ep: RegisteredEp, model_path: Path):
    # A legacy EPContext node without a "source" attribute should still be
    # claimed by the EP. Three pass conditions:
    #   - session creation succeeds (full backward compat),
    #   - session fails with an ORT-level error that doesn't say the node went
    #     unclaimed (e.g., the dummy context blob fails to deserialize — that
    #     proves the EP DID claim the node and tried to process it).
    # Re-raise only on (a) the specific "node unclaimed" marker, or
    # (b) non-ORT exceptions that would otherwise mask real bugs.
    try:
        _create_session(registered_ep, model_path)
    except Exception as exc:
        if not type(exc).__module__.startswith("onnxruntime"):
            raise
        if "is not compatible with any execution provider" in str(exc):
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dq-runtime", "qdq-runtime", "dq-compile", "foreign-source", "no-source"],
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile-prefix", type=Path, required=True)
    parser.add_argument("--compiled-path", type=Path)
    args = parser.parse_args()

    registered_ep = _register_ep()
    if args.mode == "dq-runtime":
        _run_runtime(registered_ep, args.model_path, args.profile_prefix, qdq=False)
    elif args.mode == "qdq-runtime":
        _run_runtime(registered_ep, args.model_path, args.profile_prefix, qdq=True)
    elif args.mode == "dq-compile":
        if args.compiled_path is None:
            raise ValueError("--compiled-path is required for dq-compile")
        _run_compile(registered_ep, args.model_path, args.compiled_path, args.profile_prefix)
    elif args.mode == "foreign-source":
        _run_foreign_source(registered_ep, args.model_path)
    else:
        _run_no_source(registered_ep, args.model_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
