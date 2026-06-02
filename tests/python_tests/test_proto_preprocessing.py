# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest

onnx = pytest.importorskip("onnx")

from python_tests import model_builder
from python_tests.ort_helpers import clear_path, count_nodes_by_op_type, make_session_options, require_model_compiler


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
