# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from python_tests import model_builder
from python_tests.ort_helpers import (
    clear_path,
    count_nodes_by_op_type,
    create_session,
    make_deterministic_feeds,
    make_session_options,
    require_model_compiler,
)


def _compile_to_ep_context(registered_ep, input_path, output_path):
    ModelCompiler = require_model_compiler(registered_ep.ort)
    so = make_session_options(registered_ep)
    compiler = ModelCompiler(
        so,
        str(input_path),
        embed_compiled_data_into_model=True,
    )
    compiler.compile_to_file(str(output_path))
    assert output_path.is_file()

    compiled = onnx.load(str(output_path))
    assert count_nodes_by_op_type(compiled, "EPContext") >= 1

    ep_session = create_session(registered_ep, str(output_path))
    feeds = make_deterministic_feeds(ep_session)
    cpu_session = registered_ep.ort.InferenceSession(
        str(input_path),
        providers=["CPUExecutionProvider"],
    )
    expected = cpu_session.run(None, feeds)
    actual = ep_session.run(None, feeds)
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("min_value", "max_value", "name"),
    [
        (-1.0, math.inf, "min_only"),
        (-math.inf, 1.0, "max_only"),
    ],
)
def test_trt_compile_webnn_clamp_min_only_and_max_only_succeeds(
    registered_ep,
    work_dir,
    min_value,
    max_value,
    name,
):
    input_path = work_dir / f"trt_compile_webnn_clamp_{name}.onnx"
    output_path = work_dir / f"trt_compile_webnn_clamp_{name}_ctx.onnx"
    clear_path(input_path)
    clear_path(output_path)
    model_builder.create_clip_model(input_path, min_value, max_value)
    _compile_to_ep_context(registered_ep, input_path, output_path)


@pytest.mark.parametrize("op_type", ["AveragePool", "MaxPool"])
def test_trt_compile_webnn_dilated_pool_succeeds(registered_ep, work_dir, op_type):
    input_path = work_dir / f"trt_compile_webnn_{op_type.lower()}_dilation.onnx"
    output_path = work_dir / f"trt_compile_webnn_{op_type.lower()}_dilation_ctx.onnx"
    clear_path(input_path)
    clear_path(output_path)
    model_builder.create_pool_model(input_path, op_type)
    _compile_to_ep_context(registered_ep, input_path, output_path)
