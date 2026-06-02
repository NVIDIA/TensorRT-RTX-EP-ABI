# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import subprocess

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto

from python_tests import model_builder
from python_tests.ort_helpers import (
    EP_CONTEXT_ENABLE,
    EP_CONTEXT_FILE_PATH,
    clear_path,
    create_session,
    make_session_options,
    run_session_once,
)


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
        run_session_once(session)

    compile_context_model()
    gc.collect()

    assert ctx_path.is_file()

    reloaded = create_session(registered_ep, str(ctx_path))
    run_session_once(reloaded, shape_overrides=reload_overrides)


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
    run_session_once(session)


def test_session_outputs(registered_ep, work_dir):
    topk_path = work_dir / "topk_and_multiple_graph_outputs.onnx"
    clear_path(topk_path)
    model_builder.create_topk_and_multiple_graph_outputs_model(topk_path)
    so = make_session_options(registered_ep)
    if hasattr(so, "add_free_dimension_override_by_name"):
        so.add_free_dimension_override_by_name("N", 300)
    session = registered_ep.ort.InferenceSession(str(topk_path), sess_options=so)
    assert len(session.get_outputs()) == 4

    dropout_path = work_dir / "node_output_not_used.onnx"
    clear_path(dropout_path)
    model_builder.create_node_output_not_used_model(dropout_path)
    session = create_session(registered_ep, str(dropout_path))
    assert len(session.get_outputs()) == 1


def test_runtime_caching(registered_ep, work_dir):
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
    assert len([p for p in cache_dir.iterdir() if p.is_file()]) == 1

    def load_cached_context():
        create_session(
            registered_ep,
            str(ctx_path),
            provider_options={"nv_runtime_cache_path": str(cache_dir)},
        )

    load_cached_context()
    gc.collect()
    assert len([p for p in cache_dir.iterdir() if p.is_file()]) == 1

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
        pytest.skip(str(exc))

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
        pytest.skip(str(exc))

    session = create_session(registered_ep, str(model_path))
    data = (np.arange(64 * 64, dtype=np.float16) % 100) / np.float16(100.0)
    outputs = session.run(["X_dequantized"], {"X": data.reshape(64, 64)})
    assert len(outputs) == 1
    assert outputs[0].dtype == np.float16
    assert outputs[0].shape == (64, 64)
