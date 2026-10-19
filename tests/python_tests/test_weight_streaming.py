# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import re

import numpy as np
import pytest

from python_tests import model_builder
from python_tests.ort_helpers import (
    EP_CONTEXT_EMBED_MODE,
    EP_CONTEXT_ENABLE,
    EP_CONTEXT_FILE_PATH,
    assert_nv_ep_session,
    clear_path,
    create_session,
    make_session_options,
    run_session_once,
)


WEIGHT_STREAMING_OPTION = "nv_weight_streaming_budget"
BUDGET_LOG_PATTERN = re.compile(
    r"Weight streaming budget applied.*?"
    r"requested=([^,]+), resolved=(-?\d+), actual=(-?\d+), "
    r"streamable_weights_size=(-?\d+), scratch_memory_size=(-?\d+)"
)


def _run_outputs(registered_ep, model, provider_options=None):
    session = create_session(
        registered_ep,
        str(model),
        provider_options=provider_options,
    )
    return run_session_once(session)

def _run_cpu_outputs(registered_ep, model):
    session = registered_ep.ort.InferenceSession(
        str(model),
        providers=["CPUExecutionProvider"],
    )
    return run_session_once(session)


def _decode_native_output(output):
    if b"\x00" in output[:200]:
        return output.decode("utf-16-le", errors="replace")
    return output.decode("utf-8", errors="replace")


def _captured_text(captured):
    return _decode_native_output(captured.out) + _decode_native_output(captured.err)


def _parse_budget_logs(text):
    return [
        {
            "requested": match.group(1),
            "resolved": int(match.group(2)),
            "actual": int(match.group(3)),
            "streamable_size": int(match.group(4)),
            "scratch_size": int(match.group(5)),
        }
        for match in BUDGET_LOG_PATTERN.finditer(text)
    ]


def _run_outputs_with_budget_logs(registered_ep, model, capfdbinary, provider_options=None):
    capfdbinary.readouterr()
    session_options = make_session_options(
        registered_ep,
        provider_options=provider_options,
        log_severity_level=0,
    )
    session = registered_ep.ort.InferenceSession(str(model), sess_options=session_options)
    assert_nv_ep_session(session, registered_ep)
    outputs = run_session_once(session)
    captured_text = _captured_text(capfdbinary.readouterr())
    logs = _parse_budget_logs(captured_text)
    if provider_options and WEIGHT_STREAMING_OPTION in provider_options and not logs:
        relevant_lines = [line for line in captured_text.splitlines() if "stream" in line.lower()]
        pytest.fail(f"Applied-budget log was not parseable. Streaming log lines: {relevant_lines!r}")
    return outputs, logs


def _assert_budget_applied(logs, budget):
    assert logs, f"Expected '{WEIGHT_STREAMING_OPTION}={budget}' to emit an applied-budget log"
    explicit_bytes = {
        str(1024 * 1024): 1024 * 1024,
        "1B": 1,
        "1M": 1024 * 1024,
    }
    for log in logs:
        assert log["requested"] == budget
        assert log["actual"] >= 0
        assert log["streamable_size"] >= 0
        assert log["actual"] <= log["streamable_size"]
        assert log["scratch_size"] >= 0
        if budget == "-1":
            assert log["actual"] == log["resolved"]
        elif budget == "1":
            assert log["resolved"] == 0
            assert log["actual"] == 0
        elif budget == "50%":
            expected = int(0.5 * log["streamable_size"])
            assert log["resolved"] == expected
            assert log["actual"] == min(expected, log["streamable_size"])
        elif budget == str(1024**4):
            assert log["resolved"] == 1024**4
            assert log["actual"] == log["streamable_size"]
        else:
            expected = explicit_bytes[budget]
            assert log["resolved"] == expected
            assert log["actual"] == min(expected, log["streamable_size"])


def _assert_outputs_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ("-2", "Invalid nv_weight_streaming_budget"),
        ("1KB", "base-2 unit suffixes"),
        ("101%", "percentage must be between 0 and 100"),
    ],
)
def test_invalid_weight_streaming_budgets_are_rejected(registered_ep, work_dir, budget, message):
    model_path = work_dir / f"invalid_weight_streaming_{budget.replace('%', 'pct')}.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path)
    session_options = make_session_options(
        registered_ep,
        provider_options={WEIGHT_STREAMING_OPTION: budget},
    )
    with pytest.raises(Exception, match=message):
        registered_ep.ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            enable_fallback=0,
        )


@pytest.mark.parametrize(
    "budget",
    ["-1", str(1024 * 1024), "1B", "1M", "50%", str(1024**4)],
    ids=["automatic", "explicit-bytes", "byte-suffix", "mib-suffix", "percent", "oversized"],
)
def test_weight_streaming_budget_runs_and_matches_default(
    registered_ep,
    resnet18_model_path,
    budget,
    capfdbinary,
):
    expected = _run_cpu_outputs(registered_ep, resnet18_model_path)
    actual, logs = _run_outputs_with_budget_logs(
        registered_ep,
        resnet18_model_path,
        capfdbinary,
        {WEIGHT_STREAMING_OPTION: budget},
    )
    _assert_budget_applied(logs, budget)
    _assert_outputs_close(actual, expected)


def _build_streaming_context(
    registered_ep,
    source_model,
    ctx_path,
    budget,
    *,
    embed_mode=False,
    extra_options=None,
):
    clear_path(ctx_path)
    options = dict(extra_options or {})
    options[WEIGHT_STREAMING_OPTION] = budget
    config = {
        EP_CONTEXT_ENABLE: "1",
        EP_CONTEXT_FILE_PATH: str(ctx_path),
    }
    if embed_mode:
        config[EP_CONTEXT_EMBED_MODE] = "1"
    session = create_session(
        registered_ep,
        str(source_model),
        provider_options=options,
        session_config=config,
    )
    run_session_once(session)
    del session
    gc.collect()
    assert ctx_path.is_file()


def test_non_streaming_ep_context_rejects_budget(registered_ep, work_dir):
    model_path = work_dir / "weight_streaming_non_streaming.onnx"
    ctx_path = work_dir / "weight_streaming_non_streaming_ctx.onnx"
    clear_path(model_path)
    clear_path(ctx_path)
    model_builder.create_base_model(model_path)

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

    session_options = make_session_options(
        registered_ep,
        provider_options={WEIGHT_STREAMING_OPTION: "-1"},
    )
    with pytest.raises(Exception, match="nv_weight_streaming_budget"):
        registered_ep.ort.InferenceSession(
            str(ctx_path),
            sess_options=session_options,
            enable_fallback=0,
        )


@pytest.mark.parametrize("embed_mode", [False, True], ids=["external", "embedded"])
def test_streaming_ep_context_round_trip(
    registered_ep,
    work_dir,
    resnet18_model_path,
    embed_mode,
    capfdbinary,
):
    ctx_path = work_dir / f"weight_streaming_{'embedded' if embed_mode else 'external'}_ctx.onnx"
    _build_streaming_context(
        registered_ep,
        resnet18_model_path,
        ctx_path,
        "-1",
        embed_mode=embed_mode,
    )
    expected = _run_cpu_outputs(registered_ep, resnet18_model_path)
    actual, logs = _run_outputs_with_budget_logs(
        registered_ep,
        ctx_path,
        capfdbinary,
        {WEIGHT_STREAMING_OPTION: "-1"},
    )
    _assert_budget_applied(logs, "-1")
    _assert_outputs_close(actual, expected)


def test_streaming_capable_ep_context_runs_without_budget(
    registered_ep,
    work_dir,
    resnet18_model_path,
    capfdbinary,
):
    ctx_path = work_dir / "weight_streaming_capable_budget_unset_ctx.onnx"
    _build_streaming_context(registered_ep, resnet18_model_path, ctx_path, "-1")
    _, logs = _run_outputs_with_budget_logs(registered_ep, ctx_path, capfdbinary)
    assert not logs, "Budget was applied even though nv_weight_streaming_budget was unset"


def test_runtime_cache_composes_with_weight_streaming_and_budget_changes(
    registered_ep,
    work_dir,
    resnet18_model_path,
    capfdbinary,
):
    cache_dir = work_dir / "weight_streaming_runtime_cache"
    clear_path(cache_dir)
    expected = _run_cpu_outputs(registered_ep, resnet18_model_path)

    automatic, automatic_logs = _run_outputs_with_budget_logs(
        registered_ep,
        resnet18_model_path,
        capfdbinary,
        {
            WEIGHT_STREAMING_OPTION: "-1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    _assert_budget_applied(automatic_logs, "-1")
    _assert_outputs_close(automatic, expected)
    assert cache_dir.is_dir()
    assert any(path.is_file() for path in cache_dir.iterdir())

    minimum_vram, minimum_vram_logs = _run_outputs_with_budget_logs(
        registered_ep,
        resnet18_model_path,
        capfdbinary,
        {
            WEIGHT_STREAMING_OPTION: "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    _assert_budget_applied(minimum_vram_logs, "1")
    _assert_outputs_close(minimum_vram, expected)
    _assert_outputs_close(minimum_vram, automatic)
    assert any(path.is_file() for path in cache_dir.iterdir())
