# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from python_tests import model_builder
from python_tests.ort_helpers import (
    EP_CONTEXT_FILE_PATH,
    assert_base_model_output,
    assert_nv_ep_session,
    clear_path,
    count_nodes_by_op_type,
    create_session,
    make_deterministic_feeds,
    make_session_options,
    require_model_compiler,
    run_session_once,
)

onnx = pytest.importorskip("onnx")

SPECIAL_CHARS_DIRNAME = "special (chars) r\u00e9sum\u00e9"
SPECIAL_CHARS_CASES = [
    ("EAcuteResume", "special (chars) r\u00e9sum\u00e9"),
    ("Lodz", "\u0141\u00f3d\u017a"),
    ("Zlutoucky", "\u017dlu\u0165ou\u010dk\u00fd"),
    ("YamadaCjk", "\u5c71\u7530"),
    ("MoskvaCyrillic", "\u041c\u043e\u0441\u043a\u0432\u0430"),
    ("PartyEmoji", "test_\U0001f389_party"),
    ("MultiScript", "\u00e9_\u6f22_\u03b1_\u03a9"),
]
TESTS_ROOT = Path(__file__).resolve().parent.parent

MODEL_COMPILER_UNICODE_PATH_SKIP_REASON = (
    "ONNX Runtime's Python ModelCompiler does not preserve Unicode input/output "
    "file paths on Windows"
)


@dataclass(frozen=True)
class CompileConfig:
    embed_mode: bool
    input_from_buffer: bool
    output_to_buffer: bool

    def id(self):
        return (
            ("EmbedOn" if self.embed_mode else "EmbedOff")
            + "_In"
            + ("Buf" if self.input_from_buffer else "File")
            + "_Out"
            + ("Buf" if self.output_to_buffer else "File")
        )


COMPILE_CONFIGS = [
    CompileConfig(False, False, False),
    CompileConfig(False, False, True),
    CompileConfig(False, True, False),
    CompileConfig(False, True, True),
    CompileConfig(True, False, False),
    CompileConfig(True, False, True),
    CompileConfig(True, True, False),
    CompileConfig(True, True, True),
]


def _skip_unsupported_non_embedded_buffer_output(cfg):
    if cfg.output_to_buffer and not cfg.embed_mode:
        pytest.skip(
            "Python ModelCompiler does not expose the C++ "
            "SetEpContextBinaryInformation hook needed for non-embedded bytes output"
        )


def _session_options_for_compile(registered_ep, cache_dir):
    return make_session_options(
        registered_ep,
        provider_options={
            "enable_cuda_graph": "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )


def _compile_model_flexible(
    registered_ep,
    input_path,
    output_path,
    cfg,
    cache_dir,
    input_model_bytes=None,
):
    ModelCompiler = require_model_compiler(registered_ep.ort)
    _skip_unsupported_non_embedded_buffer_output(cfg)
    so = _session_options_for_compile(registered_ep, cache_dir)
    if cfg.input_from_buffer:
        input_model = input_model_bytes if input_model_bytes is not None else input_path.read_bytes()
    else:
        input_model = str(input_path)

    compiler = ModelCompiler(
        so,
        input_model,
        embed_compiled_data_into_model=cfg.embed_mode,
    )
    if cfg.input_from_buffer:
        compiler._input_model_bytes_ref = input_model

    if cfg.output_to_buffer:
        compiled = compiler.compile_to_bytes()
        assert compiled
        return compiled

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiler.compile_to_file(str(output_path))
    assert output_path.is_file()
    return output_path


def _session_from_compile_result(registered_ep, result, intended_ctx_path):
    session_config = {}
    if isinstance(result, bytes):
        session_config[EP_CONTEXT_FILE_PATH] = str(intended_ctx_path)
    so = make_session_options(registered_ep, session_config=session_config)
    session = registered_ep.ort.InferenceSession(
        result if isinstance(result, bytes) else str(result),
        sess_options=so,
    )
    assert_nv_ep_session(session, registered_ep)
    return session


def _assert_session_matches_cpu(registered_ep, source_model, session):
    feeds = make_deterministic_feeds(session)
    cpu_session = registered_ep.ort.InferenceSession(
        str(source_model),
        providers=["CPUExecutionProvider"],
    )
    expected = cpu_session.run(None, feeds)
    actual = session.run(None, feeds)
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=3e-3, atol=3e-3)


def _assert_compile_result_materialized(result):
    if isinstance(result, bytes):
        assert result
    else:
        assert Path(result).is_file()


def _assert_compiled_has_ep_context(result):
    if isinstance(result, bytes):
        compiled_model = onnx.load_model_from_string(result)
    else:
        compiled_model = onnx.load(str(result))
    assert count_nodes_by_op_type(compiled_model, "EPContext") >= 1, (
        "Compile produced no EPContext node — the EP did not claim any subgraph"
    )


def _run_compile_worker(registered_ep, input_path, output_path, cfg, cache_dir):
    _ = registered_ep
    env = os.environ.copy()
    pythonpath_entries = [str(TESTS_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    command = [
        sys.executable,
        "-m",
        "python_tests.compile_worker",
        "--input-model",
        str(input_path),
        "--output-model",
        str(output_path),
        "--cache-dir",
        str(cache_dir),
        "--embed-mode",
        str(int(cfg.embed_mode)),
        "--input-from-buffer",
        str(int(cfg.input_from_buffer)),
        "--output-to-buffer",
        str(int(cfg.output_to_buffer)),
    ]
    return subprocess.run(
        command,
        cwd=str(output_path.parent),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("cfg", COMPILE_CONFIGS, ids=[c.id() for c in COMPILE_CONFIGS])
def test_compiles_model(registered_ep, work_dir, resnet18_model_path, cfg):
    output_path = work_dir / "context.onnx"
    clear_path(output_path)

    result = _compile_model_flexible(
        registered_ep,
        resnet18_model_path,
        output_path,
        cfg,
        work_dir / "rt_cache",
    )
    _assert_compile_result_materialized(result)
    _assert_compiled_has_ep_context(result)

    session = _session_from_compile_result(registered_ep, result, output_path)
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1
    _assert_session_matches_cpu(registered_ep, resnet18_model_path, session)


@pytest.mark.parametrize("cfg", COMPILE_CONFIGS, ids=[c.id() for c in COMPILE_CONFIGS])
def test_concurrent_compile(registered_ep, work_dir, resnet18_model_path, cfg):
    _skip_unsupported_non_embedded_buffer_output(cfg)
    cache_dir = work_dir / "rt_cache"
    clear_path(cache_dir)

    num_threads = 5
    output_paths = [work_dir / f"context_thread_{i}.onnx" for i in range(num_threads)]
    for output_path in output_paths:
        clear_path(output_path)

    input_model_bytes = resnet18_model_path.read_bytes() if cfg.input_from_buffer else None

    def worker(index):
        try:
            return index, _compile_model_flexible(
                registered_ep,
                resnet18_model_path,
                output_paths[index],
                cfg,
                cache_dir,
                input_model_bytes=input_model_bytes,
            ), None
        except Exception as exc:
            return index, None, exc

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(worker, range(num_threads)))

    for index, result, exc in results:
        assert exc is None, f"Thread {index} compilation failed: {exc!r}"
        _assert_compile_result_materialized(result)
        _assert_compiled_has_ep_context(result)


@pytest.mark.parametrize("cfg", COMPILE_CONFIGS, ids=[c.id() for c in COMPILE_CONFIGS])
def test_concurrent_session_instantiation_and_sequential_inference(
    registered_ep,
    work_dir,
    resnet18_model_path,
    cfg,
):
    _skip_unsupported_non_embedded_buffer_output(cfg)
    ctx_path = work_dir / "context_concurrent.onnx"
    result = _compile_model_flexible(
        registered_ep,
        resnet18_model_path,
        ctx_path,
        cfg,
        work_dir / "rt_cache_concurrent_sessions",
    )

    def create_one(index):
        try:
            return index, _session_from_compile_result(registered_ep, result, ctx_path), None
        except Exception as exc:
            return index, None, exc

    with ThreadPoolExecutor(max_workers=5) as pool:
        sessions = list(pool.map(create_one, range(5)))

    for index, session, exc in sessions:
        assert exc is None, f"Thread {index} session construction failed: {exc!r}"
        _assert_session_matches_cpu(registered_ep, resnet18_model_path, session)


@pytest.mark.skipif(sys.platform.startswith("win"), reason=MODEL_COMPILER_UNICODE_PATH_SKIP_REASON)
def test_context_input_path_special_chars(registered_ep, work_dir, resnet18_model_path):
    src_dir = work_dir / SPECIAL_CHARS_DIRNAME / "input"
    ctx_path = work_dir / SPECIAL_CHARS_DIRNAME / "ctx_from_special" / "context.onnx"
    src_dir.mkdir(parents=True)
    input_path = src_dir / "resnet18_Opset18_timm.onnx"
    clear_path(input_path)
    clear_path(ctx_path)
    shutil.copyfile(resnet18_model_path, input_path)

    result = _compile_model_flexible(
        registered_ep,
        input_path,
        ctx_path,
        CompileConfig(True, False, False),
        work_dir / "rt_cache_special_input",
    )
    _assert_compile_result_materialized(result)
    _assert_compiled_has_ep_context(result)
    session = _session_from_compile_result(registered_ep, result, ctx_path)
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1
    _assert_session_matches_cpu(registered_ep, input_path, session)


@pytest.mark.skipif(sys.platform.startswith("win"), reason=MODEL_COMPILER_UNICODE_PATH_SKIP_REASON)
def test_context_output_path_special_chars(registered_ep, work_dir, resnet18_model_path):
    ctx_path = work_dir / SPECIAL_CHARS_DIRNAME / "ctx_to_special" / "context.onnx"
    clear_path(ctx_path)

    result = _compile_model_flexible(
        registered_ep,
        resnet18_model_path,
        ctx_path,
        CompileConfig(True, False, False),
        work_dir / "rt_cache_special_output",
    )
    _assert_compile_result_materialized(result)
    _assert_compiled_has_ep_context(result)
    session = _session_from_compile_result(registered_ep, result, ctx_path)
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1
    _assert_session_matches_cpu(registered_ep, resnet18_model_path, session)


def test_runtime_cache_path_from_compiled_model(registered_ep, work_dir, resnet18_model_path):
    ctx_path = work_dir / "rt_cache_normal_ctx" / "context.onnx"
    cache_dir = work_dir / "rt_cache_normal_test"
    clear_path(ctx_path)
    clear_path(cache_dir)

    result = _compile_model_flexible(
        registered_ep,
        resnet18_model_path,
        ctx_path,
        CompileConfig(True, False, False),
        cache_dir,
    )
    _assert_compile_result_materialized(result)

    session = create_session(
        registered_ep,
        str(result),
        provider_options={
            "enable_cuda_graph": "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1
    run_session_once(session)
    del session
    gc.collect()

    assert cache_dir.is_dir()
    assert any(path.is_file() for path in cache_dir.iterdir())

    session = create_session(
        registered_ep,
        str(result),
        provider_options={
            "enable_cuda_graph": "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1
    run_session_once(session)
    del session
    gc.collect()


@pytest.mark.parametrize(
    "dirname",
    [case[1] for case in SPECIAL_CHARS_CASES],
    ids=[case[0] for case in SPECIAL_CHARS_CASES],
)
def test_runtime_cache_path_special_chars(registered_ep, work_dir, resnet18_model_path, dirname):
    ctx_path = work_dir / "rt_cache_special_ctx" / "context.onnx"
    cache_dir = work_dir / dirname / "rt_cache"
    clear_path(ctx_path)
    clear_path(cache_dir)

    result = _compile_model_flexible(
        registered_ep,
        resnet18_model_path,
        ctx_path,
        CompileConfig(True, False, False),
        cache_dir,
    )
    _assert_compile_result_materialized(result)

    session = create_session(
        registered_ep,
        str(result),
        provider_options={
            "enable_cuda_graph": "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1
    run_session_once(session)
    del session
    gc.collect()

    assert cache_dir.is_dir()
    assert any(path.is_file() for path in cache_dir.iterdir())

    session = create_session(
        registered_ep,
        str(result),
        provider_options={
            "enable_cuda_graph": "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    run_session_once(session)
    del session
    gc.collect()


@pytest.mark.parametrize(
    "dirname",
    [case[1] for case in SPECIAL_CHARS_CASES],
    ids=[case[0] for case in SPECIAL_CHARS_CASES],
)
def test_runtime_cache_buffer_load_special_chars(registered_ep, work_dir, resnet18_model_path, dirname):
    ctx_path = work_dir / "rt_cache_buffer_ctx" / "context.onnx"
    cache_dir = work_dir / dirname / "rt_cache_buffer"
    clear_path(ctx_path)
    clear_path(cache_dir)

    result = _compile_model_flexible(
        registered_ep,
        resnet18_model_path,
        ctx_path,
        CompileConfig(True, False, False),
        work_dir / "rt_cache_buffer_compile",
    )
    compiled_bytes = Path(result).read_bytes()

    for _ in range(2):
        session = create_session(
            registered_ep,
            compiled_bytes,
            provider_options={
                "enable_cuda_graph": "1",
                "nv_runtime_cache_path": str(cache_dir),
            },
            session_config={EP_CONTEXT_FILE_PATH: str(ctx_path)},
        )
        run_session_once(session)
        del session
        gc.collect()

    assert cache_dir.is_dir()
    assert any(path.is_file() for path in cache_dir.iterdir())


def test_compile_buffer_with_external_initializers_from_memory(registered_ep, work_dir):
    compiler_type = require_model_compiler(registered_ep.ort)
    model_path = work_dir / "external_initializer_matmul.onnx"
    data_path = work_dir / "external_initializer_matmul.data"
    ctx_path = work_dir / "external_initializer_matmul_ctx.onnx"
    for path in (model_path, data_path, ctx_path):
        clear_path(path)
    model_builder.create_external_initializer_matmul_model(model_path, data_path)

    model_bytes = model_path.read_bytes()
    data_bytes = data_path.read_bytes()
    so = make_session_options(registered_ep)
    if not hasattr(so, "add_external_initializers_from_files_in_memory"):
        pytest.fail("onnxruntime build does not expose required in-memory external initializers")
    so.add_external_initializers_from_files_in_memory(
        [data_path.name],
        [data_bytes],
        [len(data_bytes)],
    )
    compiler = compiler_type(so, model_bytes, embed_compiled_data_into_model=True)
    compiler._input_model_bytes_ref = model_bytes
    compiler._external_data_bytes_ref = data_bytes
    compiled = compiler.compile_to_bytes()
    assert compiled

    session = _session_from_compile_result(registered_ep, compiled, ctx_path)
    output = session.run(
        ["Y"],
        {"X": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)},
    )[0]
    np.testing.assert_allclose(output, np.array([[70.0, 80.0, 90.0]], dtype=np.float32), rtol=0, atol=1e-2)


def test_compile_to_stream_produces_runnable_model(registered_ep, work_dir):
    compiler_type = require_model_compiler(registered_ep.ort)
    model_path = work_dir / "compile_to_stream.onnx"
    ctx_path = work_dir / "compile_to_stream_ctx.onnx"
    clear_path(model_path)
    clear_path(ctx_path)
    model_builder.create_base_model(model_path)

    compiler = compiler_type(
        make_session_options(registered_ep),
        str(model_path),
        embed_compiled_data_into_model=True,
    )
    chunks = []
    compiler.compile_to_stream(chunks.append)
    compiled = b"".join(chunks)
    assert compiled
    _assert_compiled_has_ep_context(compiled)
    session = _session_from_compile_result(registered_ep, compiled, ctx_path)
    assert_base_model_output(session)
