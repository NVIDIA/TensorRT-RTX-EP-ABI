# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import re

import pytest

from python_tests import model_builder
from python_tests.ort_helpers import assert_base_model_output, clear_path, create_session


AUTO_PROFILE_PATTERN = re.compile(r"trt_rtx_profile_\d{8}_\d{6}_\d+\.json")


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _model(work_dir: Path, name: str) -> Path:
    path = work_dir / f"{name}.onnx"
    clear_path(path)
    model_builder.create_base_model(path, name)
    return path


def _profile_events(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("traceEvents", []) if isinstance(payload, dict) else payload


def _run_profiled(registered_ep, model_path: Path, output_path: Path, runs: int = 1, extra_options=None):
    clear_path(output_path)
    options = {
        "nv_enable_profiling": "1",
        "nv_profiling_output_file": str(output_path),
    }
    options.update(extra_options or {})
    session = create_session(registered_ep, str(model_path), provider_options=options)
    for _ in range(runs):
        assert_base_model_output(session)
    del session
    gc.collect()
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    return _profile_events(output_path)


def test_profiler_disabled_by_default(registered_ep, work_dir):
    model_path = _model(work_dir, "profiler_disabled")
    with _working_directory(work_dir):
        before = {path.name for path in work_dir.glob("trt_rtx_profile_*.json")}
        session = create_session(registered_ep, str(model_path))
        assert_base_model_output(session)
        del session
        gc.collect()
        after = {path.name for path in work_dir.glob("trt_rtx_profile_*.json")}
    assert after == before


@pytest.mark.parametrize("filename", ["profiler_created.json", "my_custom_profile_output.json"])
def test_profiler_creates_requested_output_file(registered_ep, work_dir, filename):
    model_path = _model(work_dir, Path(filename).stem)
    events = _run_profiled(registered_ep, model_path, work_dir / filename)
    assert events


def test_profiler_auto_timestamp_filename(registered_ep, work_dir):
    model_path = _model(work_dir, "profiler_auto")
    with _working_directory(work_dir):
        before = {path.name for path in work_dir.glob("trt_rtx_profile_*.json")}
        session = create_session(
            registered_ep,
            str(model_path),
            provider_options={"nv_enable_profiling": "1"},
        )
        assert_base_model_output(session)
        del session
        gc.collect()
        created = [
            path
            for path in work_dir.glob("trt_rtx_profile_*.json")
            if path.name not in before and AUTO_PROFILE_PATTERN.fullmatch(path.name)
        ]
    assert len(created) == 1


def test_profiler_json_contains_layers_sections_and_ids(registered_ep, work_dir):
    model_path = _model(work_dir, "profiler_structure")
    events = _run_profiled(registered_ep, model_path, work_dir / "profiler_structure.json")
    assert any(event.get("cat") == "nv::trt::layer" for event in events)
    assert any(event.get("name") == "process_name" for event in events)
    assert any(isinstance(event.get("pid"), int) and event["pid"] >= 0 for event in events)


@pytest.mark.parametrize("runs", [2, 3])
def test_profiler_runs_have_distinct_tids(registered_ep, work_dir, runs):
    model_path = _model(work_dir, f"profiler_{runs}_runs")
    events = _run_profiled(
        registered_ep,
        model_path,
        work_dir / f"profiler_{runs}_runs.json",
        runs=runs,
    )
    tids = {event.get("tid") for event in events if isinstance(event.get("tid"), int)}
    assert set(range(runs)).issubset(tids)


def test_profiler_composes_with_cuda_graph_request(registered_ep, work_dir):
    model_path = _model(work_dir, "profiler_cuda_graph")
    events = _run_profiled(
        registered_ep,
        model_path,
        work_dir / "profiler_cuda_graph.json",
        extra_options={"enable_cuda_graph": "1"},
    )
    assert events


def test_profiler_bad_output_path_does_not_fail_inference(registered_ep, work_dir, capfdbinary):
    model_path = _model(work_dir, "profiler_bad_path")
    bad_path = work_dir / "missing" / "nested" / "profile.json"
    clear_path(bad_path.parent)
    capfdbinary.readouterr()
    session = create_session(
        registered_ep,
        str(model_path),
        provider_options={
            "nv_enable_profiling": "1",
            "nv_profiling_output_file": str(bad_path),
        },
    )
    assert_base_model_output(session)
    del session
    gc.collect()
    captured = capfdbinary.readouterr()
    diagnostic = (captured.out + captured.err).decode("utf-8", errors="replace")
    assert "[NvTensorRTRTX EP] TrtRtxProfiler: failed to open profile output file" in diagnostic
    assert not bad_path.exists()
