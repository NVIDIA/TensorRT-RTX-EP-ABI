# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
import gc
import os
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from python_tests import model_builder
from python_tests.ort_helpers import clear_path, create_session, run_session_once

MEM_ADDR_LOCATION = "_MEM_ADDR_"


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _external_data_value(tensor, key: str) -> str | None:
    for entry in tensor.external_data:
        if entry.key == key:
            return entry.value
    return None


def _assert_no_mem_addr_tensors(graph):
    for tensor in graph.initializer:
        assert _external_data_value(tensor, "location") != MEM_ADDR_LOCATION

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                assert _external_data_value(attr.t, "location") != MEM_ADDR_LOCATION
            for tensor in attr.tensors:
                assert _external_data_value(tensor, "location") != MEM_ADDR_LOCATION
            if attr.HasField("g"):
                _assert_no_mem_addr_tensors(attr.g)
            for nested_graph in attr.graphs:
                _assert_no_mem_addr_tensors(nested_graph)


def _dumped_subgraphs_from_run(registered_ep, model_path: Path, work_dir: Path, provider_options=None):
    before = {path.resolve() for path in work_dir.glob("*.onnx")}
    dump_options = {"nv_dump_subgraphs": "1"}
    dump_options.update(provider_options or {})
    with _working_directory(work_dir):
        session = create_session(
            registered_ep,
            str(model_path),
            provider_options=dump_options,
        )
        run_session_once(session)
        del session
        gc.collect()

    after = {path.resolve() for path in work_dir.glob("*.onnx")}
    return sorted(Path(path) for path in after - before)


def _assert_dump_runs_standalone(registered_ep, dump_path: Path):
    dumped = onnx.load(str(dump_path), load_external_data=False)
    _assert_no_mem_addr_tensors(dumped.graph)

    session = registered_ep.ort.InferenceSession(str(dump_path), providers=["CPUExecutionProvider"])
    run_session_once(session)


def test_whole_model_dump_runs_standalone_and_matches_cpu(registered_ep, work_dir):
    model_path = work_dir / "initializer_matmul.onnx"
    clear_path(model_path)
    model_builder.create_initializer_matmul_model(model_path)

    dumps = _dumped_subgraphs_from_run(registered_ep, model_path, work_dir)

    assert len(dumps) == 1
    assert not dumps[0].with_suffix(".data").exists()
    dumped = onnx.load(str(dumps[0]), load_external_data=False)
    assert dumped.graph.initializer
    _assert_dump_runs_standalone(registered_ep, dumps[0])

    feeds = {"X": np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)}
    original_session = registered_ep.ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dumped_session = registered_ep.ort.InferenceSession(str(dumps[0]), providers=["CPUExecutionProvider"])

    expected_outputs = original_session.run(None, feeds)
    actual_outputs = dumped_session.run(None, feeds)
    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_multiple_trt_subgraphs_create_unique_standalone_dumps(registered_ep, work_dir):
    model_path = work_dir / "two_initializer_matmul_subgraphs.onnx"
    clear_path(model_path)
    model_builder.create_two_initializer_matmul_subgraphs_model(model_path)

    dumps = _dumped_subgraphs_from_run(
        registered_ep,
        model_path,
        work_dir,
        provider_options={"nv_op_types_to_exclude": "FastGelu"},
    )

    assert len(dumps) >= 2
    assert len({dump.name for dump in dumps}) == len(dumps)
    for dump in dumps:
        _assert_dump_runs_standalone(registered_ep, dump)
