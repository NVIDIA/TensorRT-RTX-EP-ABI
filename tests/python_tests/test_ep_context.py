# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")

from python_tests import model_builder
from python_tests.ort_helpers import (
    clear_path,
    count_nodes_by_op_type,
    require_model_compiler,
)


TESTS_ROOT = Path(__file__).resolve().parent.parent


def _run_ep_context_worker(registered_ep, mode, model_path, profile_prefix, compiled_path=None):
    # Force parent-side EP registration and skip checks before isolating native failures in a child process.
    _ = registered_ep

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(TESTS_ROOT)
        if not existing_pythonpath
        else str(TESTS_ROOT) + os.pathsep + existing_pythonpath
    )

    cmd = [
        sys.executable,
        "-m",
        "python_tests.ep_context_worker",
        "--mode",
        mode,
        "--model-path",
        str(model_path),
        "--profile-prefix",
        str(profile_prefix),
    ]
    if compiled_path is not None:
        cmd.extend(["--compiled-path", str(compiled_path)])

    return subprocess.run(
        cmd,
        cwd=str(Path(model_path).parent),
        env=env,
        capture_output=True,
    )


def _decode_worker_output(output):
    if isinstance(output, str):
        return output
    if b"\x00" in output[:200]:
        return output.decode("utf-16-le", errors="replace")
    return output.decode("utf-8", errors="replace")


def _trim_worker_output(output):
    if len(output) <= 16000:
        return output
    return output[:8000] + "\n... <worker output truncated> ...\n" + output[-8000:]


def _assert_worker_succeeded(result):
    stdout = _trim_worker_output(_decode_worker_output(result.stdout))
    stderr = _trim_worker_output(_decode_worker_output(result.stderr))
    assert result.returncode == 0, (
        f"EPContext worker failed with exit code {result.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def test_lowered_asymmetric_dq_matmul_preserves_split_graph(registered_ep, work_dir):
    model_path = work_dir / "ep_context_lowered_asymmetric_dq_fast_gelu.onnx"
    clear_path(model_path)
    model_builder.create_asymmetric_dq_matmul_fast_gelu_model(model_path)

    result = _run_ep_context_worker(
        registered_ep,
        "dq-runtime",
        model_path,
        work_dir / "ep_context_lowered_asymmetric_dq_fast_gelu_profile",
    )
    _assert_worker_succeeded(result)


def test_lowered_asymmetric_qdq_matmul_preserves_split_graph(registered_ep, work_dir):
    model_path = work_dir / "ep_context_lowered_asymmetric_qdq_fast_gelu.onnx"
    clear_path(model_path)
    model_builder.create_asymmetric_qdq_matmul_fast_gelu_model(model_path)

    result = _run_ep_context_worker(
        registered_ep,
        "qdq-runtime",
        model_path,
        work_dir / "ep_context_lowered_asymmetric_qdq_fast_gelu_profile",
    )
    _assert_worker_succeeded(result)


def test_compile_lowered_asymmetric_dq_preserves_split_graph_and_output(registered_ep, work_dir):
    require_model_compiler(registered_ep.ort)

    model_path = work_dir / "ep_context_compile_lowered_asymmetric_dq_fast_gelu.onnx"
    compiled_path = work_dir / "ep_context_compile_lowered_asymmetric_dq_fast_gelu_ctx.onnx"
    clear_path(model_path)
    clear_path(compiled_path)
    model_builder.create_asymmetric_dq_matmul_fast_gelu_model(model_path)

    result = _run_ep_context_worker(
        registered_ep,
        "dq-compile",
        model_path,
        work_dir / "ep_context_compile_lowered_asymmetric_dq_fast_gelu_profile",
        compiled_path=compiled_path,
    )
    _assert_worker_succeeded(result)
    assert compiled_path.is_file()

    compiled_model = onnx.load(str(compiled_path))
    assert count_nodes_by_op_type(compiled_model, "EPContext") >= 1
    assert count_nodes_by_op_type(compiled_model, "FastGelu", "com.microsoft") == 1


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("OpenVINOExecutionProvider", "foreign_source"),
        ("TensorrtExecutionProvider", "classic_trt_source"),
    ],
)
def test_ep_context_node_foreign_sources_are_skipped(registered_ep, work_dir, source, name):
    model_path = work_dir / f"ep_context_{name}_nv.onnx"
    clear_path(model_path)
    model_builder.create_synthetic_ep_context_model(model_path, source)

    result = _run_ep_context_worker(
        registered_ep,
        "foreign-source",
        model_path,
        work_dir / f"ep_context_{name}_nv_profile",
    )
    _assert_worker_succeeded(result)


def test_ep_context_node_without_source_is_backward_compatible(registered_ep, work_dir):
    model_path = work_dir / "ep_context_no_source_nv.onnx"
    clear_path(model_path)
    model_builder.create_synthetic_ep_context_model(
        model_path,
        source_attr="",
        include_source_attr=False,
    )

    result = _run_ep_context_worker(
        registered_ep,
        "no-source",
        model_path,
        work_dir / "ep_context_no_source_nv_profile",
    )
    _assert_worker_succeeded(result)
