# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto

from python_tests import model_builder
from python_tests.ort_helpers import (
    EP_CONTEXT_ENABLE,
    EP_CONTEXT_FILE_PATH,
    assert_base_model_output,
    clear_path,
    create_session,
    make_session_options,
    run_session_once,
)


TESTS_ROOT = Path(__file__).resolve().parent.parent

def _decode_native_output(output):
    if b"\x00" in output[:200]:
        return output.decode("utf-16-le", errors="replace")
    return output.decode("utf-8", errors="replace")


def _captured_text(captured):
    return _decode_native_output(captured.out) + _decode_native_output(captured.err)


def _context_embed_and_reload(registered_ep, work_dir, model_stem, dims, reload_overrides=None):
    model_path = work_dir / f"{model_stem}.onnx"
    ctx_path = work_dir / f"{model_stem}_ctx.onnx"
    clear_path(model_path)
    clear_path(ctx_path)
    model_builder.create_base_model(model_path, "test", dims)

    def compile_context_model():
        session = create_session(
            registered_ep,
            str(model_path),
            session_config={
                EP_CONTEXT_ENABLE: "1",
                EP_CONTEXT_FILE_PATH: str(ctx_path),
            },
        )
        assert_base_model_output(session)

    compile_context_model()
    gc.collect()

    assert ctx_path.is_file()

    reloaded = create_session(registered_ep, str(ctx_path))
    assert_base_model_output(reloaded, shape_overrides=reload_overrides)


def test_context_embed_and_reload(registered_ep, work_dir):
    _context_embed_and_reload(registered_ep, work_dir, "nv_execution_provider_test", (1, 3, 2))


def test_context_embed_and_reload_dynamic(registered_ep, work_dir):
    _context_embed_and_reload(
        registered_ep,
        work_dir,
        "nv_execution_provider_dyn_test",
        (1, -1, -1),
        reload_overrides={"X": [1, 5, 5], "Y": [1, 5, 1], "Z": [1, 5, 5]},
    )


def test_context_embed_and_reload_data_dynamic(registered_ep, work_dir):
    _context_embed_and_reload(
        registered_ep,
        work_dir,
        "nv_execution_provider_data_dyn_test",
        (1, -1, -1),
        reload_overrides={"X": [1, 5, 5], "Y": [1, 5, 5], "Z": [1, 5, 5]},
    )


@pytest.mark.parametrize(
    ("dtype_name", "onnx_dtype"),
    [
        ("fp32", TensorProto.FLOAT),
        ("fp16", TensorProto.FLOAT16),
        pytest.param(
            "bf16",
            getattr(TensorProto, "BFLOAT16", TensorProto.UNDEFINED),
            marks=pytest.mark.skip(reason="Python session.run has no portable BF16 ndarray feed"),
        ),
        ("int64", TensorProto.INT64),
        ("int32", TensorProto.INT32),
    ],
)
def test_io_types(registered_ep, work_dir, dtype_name, onnx_dtype):
    model_path = work_dir / f"nv_execution_provider_{dtype_name}.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path, f"test_{dtype_name}", (1, 5, 10), dtype=onnx_dtype)

    session = create_session(registered_ep, str(model_path))
    assert_base_model_output(session)


def test_session_outputs(registered_ep, work_dir):
    topk_path = work_dir / "topk_and_multiple_graph_outputs.onnx"
    clear_path(topk_path)
    model_builder.create_topk_and_multiple_graph_outputs_model(topk_path)
    so = make_session_options(registered_ep)
    if not hasattr(so, "add_free_dimension_override_by_name"):
        pytest.fail("onnxruntime.SessionOptions lacks required free-dimension override API")
    so.add_free_dimension_override_by_name("N", 300)
    session = registered_ep.ort.InferenceSession(str(topk_path), sess_options=so)
    assert len(session.get_outputs()) == 4
    topk_input = np.arange(300, dtype=np.float32)
    scores, less, quotients, labels = session.run(None, {"input": topk_input})
    np.testing.assert_array_equal(scores, topk_input[::-1])
    assert not np.any(less)
    np.testing.assert_array_equal(quotients, np.arange(299, -1, -1, dtype=np.int64) // 26)
    np.testing.assert_array_equal(labels, np.arange(299, -1, -1, dtype=np.int64) % 26)

    dropout_path = work_dir / "node_output_not_used.onnx"
    clear_path(dropout_path)
    model_builder.create_node_output_not_used_model(dropout_path)
    session = create_session(registered_ep, str(dropout_path))
    assert len(session.get_outputs()) == 1
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    w = np.arange(6, dtype=np.float32).reshape(2, 3)
    output = session.run(["Y"], {"X": x, "W": w})[0]
    np.testing.assert_allclose(output, x @ w, rtol=0, atol=1e-5)


def test_runtime_caching(registered_ep, work_dir, capfdbinary):
    model_path = work_dir / "nv_execution_provider_runtime_caching.onnx"
    ctx_path = work_dir / "nv_execution_provider_runtime_caching_ctx.onnx"
    cache_dir = work_dir / "runtime_cache"
    clear_path(model_path)
    clear_path(ctx_path)
    clear_path(cache_dir)
    model_builder.create_base_model(model_path, "test", (1, 3, 2))

    def create_context_and_cache():
        session = create_session(
            registered_ep,
            str(model_path),
            provider_options={"nv_runtime_cache_path": str(cache_dir)},
            session_config={
                EP_CONTEXT_ENABLE: "1",
                EP_CONTEXT_FILE_PATH: str(ctx_path),
            },
        )
        run_session_once(session)

    create_context_and_cache()
    gc.collect()

    assert cache_dir.is_dir()
    cache_files = [p for p in cache_dir.iterdir() if p.is_file()]
    assert len(cache_files) == 1

    invalid_cache = b"not a TensorRT RTX runtime cache"
    cache_files[0].write_bytes(invalid_cache)
    capfdbinary.readouterr()
    recovered = create_session(
        registered_ep,
        str(ctx_path),
        provider_options={"nv_runtime_cache_path": str(cache_dir)},
        log_severity_level=1,
    )
    assert_base_model_output(recovered)
    recovery_log = _captured_text(capfdbinary.readouterr())
    del recovered
    gc.collect()
    assert "TensorRT RTX failed to deserialize runtime cache" in recovery_log
    assert cache_files[0].stat().st_size > len(invalid_cache)

    new_cache_dir = work_dir / "runtime_cache_new"
    clear_path(new_cache_dir)

    def load_context_with_new_cache():
        create_session(
            registered_ep,
            str(ctx_path),
            provider_options={"nv_runtime_cache_path": str(new_cache_dir)},
        )

    load_context_with_new_cache()
    gc.collect()
    assert new_cache_dir.is_dir()
    assert len([p for p in new_cache_dir.iterdir() if p.is_file()]) == 1


def test_runtime_cache_unsafe_partition_name_disables_caching(registered_ep, work_dir, capfdbinary):
    model_path = work_dir / "runtime_cache_unsafe_name.onnx"
    ctx_path = work_dir / "runtime_cache_unsafe_name_ctx.onnx"
    unsafe_ctx_path = work_dir / "runtime_cache_unsafe_name_modified_ctx.onnx"
    cache_dir = work_dir / "runtime_cache_unsafe_name"
    for path in (model_path, ctx_path, unsafe_ctx_path, cache_dir):
        clear_path(path)

    model_builder.create_base_model(model_path, "runtime_cache_unsafe_name")
    session = create_session(
        registered_ep,
        str(model_path),
        session_config={
            EP_CONTEXT_ENABLE: "1",
            EP_CONTEXT_FILE_PATH: str(ctx_path),
        },
    )
    run_session_once(session)
    del session
    gc.collect()
    assert ctx_path.is_file()

    context_model = onnx.load(str(ctx_path), load_external_data=False)
    updated = False
    for node in context_model.graph.node:
        if node.op_type != "EPContext":
            continue
        for attr in node.attribute:
            if attr.name == "partition_name":
                attr.s = b".."
                updated = True
    assert updated
    onnx.save(context_model, str(unsafe_ctx_path))
    cache_dir.mkdir()

    capfdbinary.readouterr()
    session = create_session(
        registered_ep,
        str(unsafe_ctx_path),
        provider_options={"nv_runtime_cache_path": str(cache_dir)},
    )
    assert_base_model_output(session)
    warning_log = _captured_text(capfdbinary.readouterr())
    del session
    gc.collect()
    assert "Runtime cache disabled for this node" in warning_log
    assert "parent-directory traversal" in warning_log
    assert not any(cache_dir.iterdir())


def test_runtime_cache_write_failure_does_not_escape_teardown(registered_ep, work_dir):
    model_path = work_dir / "runtime_cache_write_failure.onnx"
    cache_dir = work_dir / "runtime_cache_write_failure"
    clear_path(model_path)
    clear_path(cache_dir)
    model_builder.create_base_model(model_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(TESTS_ROOT)
        if not env.get("PYTHONPATH")
        else str(TESTS_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    env["NV_TRT_RTX_EP_LIBRARY_PATH"] = str(registered_ep.library_path)
    env["NV_TRT_RTX_EP_NAME"] = registered_ep.ep_name
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "python_tests.runtime_cache_worker",
            "--model",
            str(model_path),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert cache_dir.is_file()
    teardown_log = result.stdout + result.stderr
    assert "cache save was skipped" not in teardown_log
    assert "[NvTensorRTRTX EP] Failed to save runtime cache" in teardown_log


def test_sync_gpu_allocator_runs(registered_ep, work_dir):
    model_path = work_dir / "nv_execution_provider_sync_gpu_allocator.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path)
    session = create_session(
        registered_ep,
        str(model_path),
        provider_options={"nv_use_sync_gpu_allocator": "1"},
    )
    assert_base_model_output(session)


@pytest.mark.parametrize(
    "option_name",
    ["nv_multi_rotary_cache_concat_offset", "multi_rotary_cache_concat_offset"],
)
def test_multi_rotary_cache_concat_offset_aliases_are_accepted(registered_ep, work_dir, option_name):
    model_path = work_dir / f"{option_name}.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path)
    session = create_session(
        registered_ep,
        str(model_path),
        provider_options={option_name: "4096"},
    )
    assert_base_model_output(session)


def test_multi_rotary_cache_concat_offset_conflicting_aliases_are_rejected(registered_ep, work_dir):
    model_path = work_dir / "multi_rotary_cache_concat_offset_conflict.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path)
    session_options = make_session_options(
        registered_ep,
        provider_options={
            "nv_multi_rotary_cache_concat_offset": "4096",
            "multi_rotary_cache_concat_offset": "8192",
        },
    )
    with pytest.raises(Exception, match="multi_rotary_cache_concat_offset"):
        registered_ep.ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            enable_fallback=0,
        )


def test_high_rank_input_is_rejected_without_cpu_fallback(registered_ep, work_dir):
    model_path = work_dir / "nv_execution_provider_high_rank.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path, "high_rank", (1,) * 9)
    with pytest.raises(Exception, match=r"(?i)(rank|dimensions|MAX_DIMS|fallback)"):
        create_session(
            registered_ep,
            str(model_path),
            session_config={"session.disable_cpu_ep_fallback": "1"},
        )


def _blackwell_or_above():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False

    for line in result.stdout.splitlines():
        parts = line.strip().split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[0]) * 100 + int(parts[1]) * 10 >= 1200:
                return True
    return False


@pytest.mark.blackwell
def test_fp8_custom_op_model(registered_ep, work_dir):
    if not _blackwell_or_above():
        pytest.skip("Test requires SM 12.0+ GPU")

    model_path = work_dir / "nv_execution_provider_fp8_quantize_dequantize_test.onnx"
    clear_path(model_path)
    try:
        model_builder.create_fp8_custom_op_model(model_path)
    except RuntimeError as exc:
        pytest.fail(f"Could not build required hardware-specific custom-op model: {exc}")

    session = create_session(registered_ep, str(model_path))
    data = (np.arange(4 * 64, dtype=np.float16) % 100) / np.float16(100.0)
    outputs = session.run(["Y"], {"X": data.reshape(4, 64)})
    assert len(outputs) == 1
    assert outputs[0].dtype == np.float16
    assert outputs[0].shape == (4, 64)


@pytest.mark.blackwell
def test_fp4_custom_op_model(registered_ep, work_dir):
    if not _blackwell_or_above():
        pytest.skip("Test requires SM 12.0+ GPU")

    model_path = work_dir / "nv_execution_provider_fp4_dynamic_quantize_test.onnx"
    clear_path(model_path)
    try:
        model_builder.create_fp4_custom_op_model(model_path)
    except RuntimeError as exc:
        pytest.fail(f"Could not build required hardware-specific custom-op model: {exc}")

    session = create_session(registered_ep, str(model_path))
    data = (np.arange(64 * 64, dtype=np.float16) % 100) / np.float16(100.0)
    outputs = session.run(["X_dequantized"], {"X": data.reshape(64, 64)})
    assert len(outputs) == 1
    assert outputs[0].dtype == np.float16
    assert outputs[0].shape == (64, 64)
