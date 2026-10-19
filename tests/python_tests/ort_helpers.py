# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

EP_CONTEXT_ENABLE = "ep.context_enable"
EP_CONTEXT_FILE_PATH = "ep.context_file_path"
EP_CONTEXT_EMBED_MODE = "ep.context_embed_mode"
DISABLE_CPU_EP_FALLBACK = "session.disable_cpu_ep_fallback"


def clear_path(path: str | Path):
    path = Path(path)
    if not path.exists():
        return

    allowed_roots = [
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]
    resolved = path.resolve()
    safe_roots = [
        root for root in allowed_roots
        if root.parent != root and root != Path(root.anchor).resolve()
    ]
    if not any(resolved != root and root in resolved.parents for root in safe_roots):
        raise RuntimeError(f"Refusing to remove path outside working directory: {resolved}")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def get_trt_rtx_devices(registered_ep):
    devices = [
        d for d in registered_ep.ort.get_ep_devices()
        if getattr(d, "ep_name", None) == registered_ep.ep_name
    ]
    if not devices:
        pytest.fail(f"No OrtEpDevice found for mandatory EP {registered_ep.ep_name}")
    return devices


def assert_nv_ep_session(session, registered_ep):
    providers = session.get_providers()
    assert registered_ep.ep_name in providers, (
        f"Session does not expose mandatory EP {registered_ep.ep_name}; "
        f"available providers: {providers}"
    )


def make_session_options(
    registered_ep,
    provider_options: Mapping[str, str] | None = None,
    session_config: Mapping[str, str] | None = None,
    profiling_prefix: str | Path | None = None,
    log_severity_level: int | None = None,
    allow_cpu_fallback: bool = False,
):
    ort = registered_ep.ort
    so = ort.SessionOptions()

    config = {str(key): str(value) for key, value in (session_config or {}).items()}
    if allow_cpu_fallback:
        if config.get(DISABLE_CPU_EP_FALLBACK) == "1":
            raise ValueError(
                f"allow_cpu_fallback=True conflicts with {DISABLE_CPU_EP_FALLBACK}=1"
            )
    else:
        configured_value = config.setdefault(DISABLE_CPU_EP_FALLBACK, "1")
        if configured_value != "1":
            raise ValueError(
                f"TRT RTX EP tests require {DISABLE_CPU_EP_FALLBACK}=1; "
                "pass allow_cpu_fallback=True only for an intentional split-graph test"
            )

    for key, value in config.items():
        so.add_session_config_entry(str(key), str(value))

    if profiling_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = str(profiling_prefix)

    if log_severity_level is not None:
        so.log_severity_level = log_severity_level

    if not hasattr(so, "add_provider_for_devices"):
        pytest.fail("onnxruntime.SessionOptions does not expose mandatory add_provider_for_devices")

    so.add_provider_for_devices(
        get_trt_rtx_devices(registered_ep),
        {str(k): str(v) for k, v in (provider_options or {}).items()},
    )
    return so


def create_session(registered_ep, model_path_or_bytes, **session_options_kwargs):
    so = make_session_options(registered_ep, **session_options_kwargs)
    session = registered_ep.ort.InferenceSession(model_path_or_bytes, sess_options=so)
    assert_nv_ep_session(session, registered_ep)
    return session


def concrete_shape(shape, override=None):
    if override is not None:
        return list(override)
    result = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            result.append(dim)
        else:
            result.append(1)
    return result


def numpy_dtype_for_ort_type(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(int16)": np.int16,
        "tensor(uint16)": np.uint16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    if ort_type == "tensor(bfloat16)":
        pytest.skip("Normal Python session.run feeds do not expose a portable BF16 ndarray path")
    try:
        return mapping[ort_type]
    except KeyError:
        pytest.fail(f"No deterministic-feed mapping for ONNX Runtime type {ort_type}")


def make_deterministic_feeds(session, shape_overrides: Mapping[str, list[int]] | None = None):
    feeds = {}
    shape_overrides = shape_overrides or {}
    for inp in session.get_inputs():
        shape = concrete_shape(inp.shape, shape_overrides.get(inp.name))
        dtype = numpy_dtype_for_ort_type(inp.type)
        element_count = int(np.prod(shape, dtype=np.int64))
        values = np.arange(1, element_count + 1, dtype=np.int64).reshape(shape)
        if dtype == np.bool_:
            feeds[inp.name] = (values % 2).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            feeds[inp.name] = ((values % 11) / 4.0 - 1.0).astype(dtype)
        else:
            feeds[inp.name] = (values % 11).astype(dtype)
    return feeds


def run_session_once(session, feeds=None, shape_overrides=None):
    if feeds is None:
        feeds = make_deterministic_feeds(session, shape_overrides)
    return session.run(None, feeds)


def assert_base_model_output(session, shape_overrides=None):
    feeds = make_deterministic_feeds(session, shape_overrides)
    outputs = session.run(["O"], feeds)
    assert len(outputs) == 1
    expected = feeds["X"] + feeds["Y"] + feeds["Z"] + feeds["S"][0]
    if np.issubdtype(expected.dtype, np.floating):
        tolerance = 2e-3 if expected.dtype == np.float16 else 1e-5
        np.testing.assert_allclose(outputs[0], expected, rtol=0, atol=tolerance)
    else:
        np.testing.assert_array_equal(outputs[0], expected)
    return outputs[0]


def require_model_compiler(ort):
    if not hasattr(ort, "ModelCompiler"):
        pytest.fail("onnxruntime build does not expose required ModelCompiler")
    return ort.ModelCompiler


def count_nodes_by_op_type(model, op_type: str, domain: str | None = None) -> int:
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


def read_profile_text(session) -> str:
    profile_path = Path(session.end_profiling())
    if not profile_path.is_file():
        pytest.fail(f"Profiling output not found at {profile_path}")
    return profile_path.read_text(encoding="utf-8", errors="ignore")
