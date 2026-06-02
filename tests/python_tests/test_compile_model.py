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

import pytest

from python_tests.ort_helpers import (
    EP_CONTEXT_FILE_PATH,
    clear_path,
    count_nodes_by_op_type,
    create_session,
    make_session_options,
    require_model_compiler,
    run_session_once,
)

onnx = pytest.importorskip("onnx")

SPECIAL_CHARS_DIRNAME = "special (chars) r\u00e9sum\u00e9"
TESTS_ROOT = Path(__file__).resolve().parent.parent

# The EP receives provider-option paths as std::string and feeds them into
# std::filesystem::path, which on Windows decodes via the system ANSI codepage
# rather than UTF-8. Python ORT bindings send UTF-8 bytes, so non-ASCII path
# characters (e.g. "\u00e9") get mis-decoded and the EP fails with "invalid
# argument". The C++ tests only pass on machines configured with the UTF-8
# ACP. Skip until the EP accepts UTF-8 explicitly.
SPECIAL_CHARS_SKIP_REASON = (
    "EP decodes provider-option paths via Windows ACP, not UTF-8; "
    "non-ASCII path characters fail from Python until the EP is fixed"
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
    return registered_ep.ort.InferenceSession(result if isinstance(result, bytes) else str(result), sess_options=so)


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


@pytest.mark.skipif(sys.platform.startswith("win"), reason=SPECIAL_CHARS_SKIP_REASON)
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
    run_session_once(session)


@pytest.mark.skipif(sys.platform.startswith("win"), reason=SPECIAL_CHARS_SKIP_REASON)
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
    run_session_once(session)


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


@pytest.mark.skipif(sys.platform.startswith("win"), reason=SPECIAL_CHARS_SKIP_REASON)
def test_runtime_cache_path_special_chars(registered_ep, work_dir, resnet18_model_path):
    base_dir = work_dir / SPECIAL_CHARS_DIRNAME
    ctx_path = base_dir / "rt_cache_ctx" / "context.onnx"
    cache_dir = base_dir / "rt_cache"
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
