# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper


def _dims(name: str, dims):
    result = []
    dyn_idx = 0
    for dim in dims:
        if dim == -1:
            result.append(f"{name}_dim_{dyn_idx}")
            dyn_idx += 1
        else:
            result.append(dim)
    return result


def value_info(name: str, elem_type: int, dims):
    if dims is None:
        value = onnx.ValueInfoProto()
        value.name = name
        value.type.tensor_type.elem_type = elem_type
        return value
    return helper.make_tensor_value_info(name, elem_type, _dims(name, dims))


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def make_model(graph, opsets=None, ir_version=7):
    model = helper.make_model(
        graph,
        opset_imports=opsets or [helper.make_opsetid("", 13)],
    )
    model.ir_version = ir_version
    return model


def create_base_model(
    path,
    graph_name="test",
    dims=(1, 3, 2),
    add_fast_gelu=False,
    dtype=TensorProto.FLOAT,
):
    inputs = [
        value_info("X", dtype, dims),
        value_info("Y", dtype, dims),
        value_info("Z", dtype, dims),
        value_info("S", dtype, (1,)),
    ]
    outputs = [value_info("O", dtype, None)]
    nodes = [
        helper.make_node("Add", ["X", "Y"], ["node_1_out_1"], name="node_1"),
        helper.make_node("Add", ["node_1_out_1", "Z"], ["node_2_out_1"], name="node_2"),
    ]
    last_output = "node_2_out_1"
    opsets = [helper.make_opsetid("", 13)]
    if add_fast_gelu:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
        nodes.append(
            helper.make_node(
                "FastGelu",
                [last_output],
                ["node_3_out_1"],
                name="node_3",
                domain="com.microsoft",
            )
        )
        last_output = "node_3_out_1"

    nodes.append(helper.make_node("Add", [last_output, "S"], ["O"], name="node_5"))
    graph = helper.make_graph(nodes, graph_name, inputs, outputs)
    save_model(make_model(graph, opsets), path)


def create_topk_and_multiple_graph_outputs_model(path):
    inputs = [value_info("input", TensorProto.FLOAT, ("N",))]
    outputs = [
        value_info("scores", TensorProto.FLOAT, ("K",)),
        value_info("Less_output_0", TensorProto.BOOL, ("K",)),
        value_info("Div_17_output_0", TensorProto.INT64, ("K",)),
        value_info("labels", TensorProto.INT64, ("K",)),
    ]
    initializers = [
        helper.make_tensor("K", TensorProto.INT64, [1], [300]),
        helper.make_tensor("zero", TensorProto.INT64, [], [0]),
        helper.make_tensor("twenty_six", TensorProto.INT64, [], [26]),
    ]
    nodes = [
        helper.make_node("TopK", ["input", "K"], ["scores", "topk_indices"], name="TopK"),
        helper.make_node("Less", ["topk_indices", "zero"], ["Less_output_0"], name="Less"),
        helper.make_node("Div", ["topk_indices", "twenty_six"], ["Div_17_output_0"], name="Div"),
        helper.make_node("Mod", ["topk_indices", "twenty_six"], ["labels"], name="Mod"),
    ]
    graph = helper.make_graph(nodes, "TopKGraph", inputs, outputs, initializer=initializers)
    save_model(make_model(graph), path)


def create_node_output_not_used_model(path):
    inputs = [
        value_info("X", TensorProto.FLOAT, (3, 2)),
        value_info("W", TensorProto.FLOAT, (2, 3)),
    ]
    outputs = [value_info("Y", TensorProto.FLOAT, (3, 3))]
    nodes = [
        helper.make_node("Dropout", ["X"], ["dropout_out", "dropout_mask"], name="DropoutNode"),
        helper.make_node("MatMul", ["dropout_out", "W"], ["Y"], name="MatMulNode"),
    ]
    graph = helper.make_graph(nodes, "DropoutMatMulGraph", inputs, outputs)
    save_model(make_model(graph), path)


def create_initializer_matmul_model(path):
    weights = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        dtype=np.float32,
    )
    inputs = [value_info("X", TensorProto.FLOAT, (1, 4))]
    outputs = [value_info("Y", TensorProto.FLOAT, (1, 3))]
    initializers = [
        helper.make_tensor("W", TensorProto.FLOAT, [4, 3], weights.tobytes(), raw=True),
    ]
    nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul")]
    graph = helper.make_graph(nodes, "InitializerMatMulGraph", inputs, outputs, initializer=initializers)
    save_model(make_model(graph), path)


def create_two_initializer_matmul_subgraphs_model(path):
    weights_1 = np.eye(4, dtype=np.float32)
    weights_2 = np.array(
        [
            1.0,
            0.5,
            -1.0,
            2.0,
            0.25,
            -0.75,
            1.5,
            -1.5,
        ],
        dtype=np.float32,
    )
    inputs = [value_info("X", TensorProto.FLOAT, (1, 4))]
    outputs = [value_info("Y", TensorProto.FLOAT, (1, 2))]
    value_infos = [
        value_info("matmul_1_output", TensorProto.FLOAT, (1, 4)),
        value_info("fast_gelu_output", TensorProto.FLOAT, (1, 4)),
    ]
    initializers = [
        helper.make_tensor("W1", TensorProto.FLOAT, [4, 4], weights_1.tobytes(), raw=True),
        helper.make_tensor("W2", TensorProto.FLOAT, [4, 2], weights_2.tobytes(), raw=True),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["matmul_1_output"], name="matmul_1"),
        helper.make_node(
            "FastGelu",
            ["matmul_1_output"],
            ["fast_gelu_output"],
            name="fast_gelu",
            domain="com.microsoft",
        ),
        helper.make_node("MatMul", ["fast_gelu_output", "W2"], ["Y"], name="matmul_2"),
    ]
    graph = helper.make_graph(
        nodes,
        "TwoInitializerMatMulSubgraphs",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos,
    )
    save_model(make_model(graph, [helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)]), path)


def create_asymmetric_dq_matmul_fast_gelu_model(path):
    q_weights = np.array([16, -8, 5, 12, -3, 9], dtype=np.int8)
    zero_point = np.array([3], dtype=np.int8)
    inputs = [value_info("X", TensorProto.FLOAT, (2, 3))]
    outputs = [value_info("O", TensorProto.FLOAT, (2, 2))]
    value_infos = [
        value_info("dequantizedWeights", TensorProto.FLOAT, (3, 2)),
        value_info("matmulOutput", TensorProto.FLOAT, (2, 2)),
    ]
    initializers = [
        helper.make_tensor("Wq", TensorProto.INT8, [3, 2], q_weights.tobytes(), raw=True),
        helper.make_tensor("weightScale", TensorProto.FLOAT, [1], [0.25]),
        helper.make_tensor("weightZeroPoint", TensorProto.INT8, [1], zero_point.tobytes(), raw=True),
    ]
    nodes = [
        helper.make_node(
            "DequantizeLinear",
            ["Wq", "weightScale", "weightZeroPoint"],
            ["dequantizedWeights"],
            name="dq_weights",
        ),
        helper.make_node("MatMul", ["X", "dequantizedWeights"], ["matmulOutput"], name="matmul"),
        helper.make_node("FastGelu", ["matmulOutput"], ["O"], name="fast_gelu", domain="com.microsoft"),
    ]
    graph = helper.make_graph(
        nodes,
        "LoweredAsymmetricDqFastGeluGraph",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos,
    )
    save_model(make_model(graph, [helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)]), path)


def create_asymmetric_qdq_matmul_fast_gelu_model(path):
    weights = np.array(
        [1.25, -0.75, 0.50, 2.00, -1.50, 0.25],
        dtype=np.float32,
    )
    zero_point = np.array([5], dtype=np.int8)
    inputs = [value_info("X", TensorProto.FLOAT, (2, 3))]
    outputs = [value_info("O", TensorProto.FLOAT, (2, 2))]
    value_infos = [
        value_info("quantizedX", TensorProto.INT8, (2, 3)),
        value_info("dequantizedX", TensorProto.FLOAT, (2, 3)),
        value_info("matmulOutput", TensorProto.FLOAT, (2, 2)),
    ]
    initializers = [
        helper.make_tensor("activationScale", TensorProto.FLOAT, [1], [0.25]),
        helper.make_tensor("activationZeroPoint", TensorProto.INT8, [1], zero_point.tobytes(), raw=True),
        helper.make_tensor("W", TensorProto.FLOAT, [3, 2], weights.tolist()),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["X", "activationScale", "activationZeroPoint"],
            ["quantizedX"],
            name="quantize_x",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["quantizedX", "activationScale", "activationZeroPoint"],
            ["dequantizedX"],
            name="dequantize_x",
        ),
        helper.make_node("MatMul", ["dequantizedX", "W"], ["matmulOutput"], name="matmul"),
        helper.make_node("FastGelu", ["matmulOutput"], ["O"], name="fast_gelu", domain="com.microsoft"),
    ]
    graph = helper.make_graph(
        nodes,
        "LoweredAsymmetricQdqFastGeluGraph",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos,
    )
    save_model(make_model(graph, [helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)]), path)


def create_synthetic_ep_context_model(path, source_attr="", include_source_attr=True):
    attributes = {
        "embed_mode": 1,
        "ep_cache_context": b"dummy_context_data",
    }
    if include_source_attr:
        attributes["source"] = source_attr

    node = helper.make_node(
        "EPContext",
        ["input"],
        ["output"],
        name="ep_context_node",
        domain="com.microsoft",
        **attributes,
    )
    graph = helper.make_graph(
        [node],
        "EPContextSourceTest",
        [value_info("input", TensorProto.FLOAT, (1, 3))],
        [value_info("output", TensorProto.FLOAT, (1, 3))],
    )
    save_model(make_model(graph, [helper.make_opsetid("", 11), helper.make_opsetid("com.microsoft", 1)]), path)


def create_fp8_custom_op_model(path):
    fp8_type = getattr(TensorProto, "FLOAT8E4M3FN", None)
    if fp8_type is None:
        raise RuntimeError("Installed ONNX package does not define FLOAT8E4M3FN")

    scale = np.array([np.float16(0.0078125)], dtype=np.float16)
    nodes = [
        helper.make_node(
            "TRT_FP8QuantizeLinear",
            ["X", "scale"],
            ["X_quantized"],
            name="trt_fp8_quantize_node",
            domain="trt",
        ),
        helper.make_node(
            "TRT_FP8DequantizeLinear",
            ["X_quantized", "scale"],
            ["Y"],
            name="trt_fp8_dequantize_node",
            domain="trt",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "nv_execution_provider_fp8_quantize_dequantize_graph",
        [value_info("X", TensorProto.FLOAT16, (4, 64))],
        [value_info("Y", TensorProto.FLOAT16, (4, 64))],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT16, [1], scale.tobytes(), raw=True)],
        value_info=[value_info("X_quantized", fp8_type, (4, 64))],
    )
    save_model(make_model(graph, [helper.make_opsetid("", 19), helper.make_opsetid("trt", 1)]), path)


def create_fp4_custom_op_model(path):
    fp8_type = getattr(TensorProto, "FLOAT8E4M3FN", None)
    fp4_type = getattr(TensorProto, "FLOAT4E2M1", None)
    if fp8_type is None or fp4_type is None:
        raise RuntimeError("Installed ONNX package does not define FP8/FP4 tensor types")

    scale = np.array([np.float16(0.1234)], dtype=np.float16)
    dequant_scale = np.array([np.float16(0.0625)], dtype=np.float16)
    nodes = [
        helper.make_node(
            "TRT_FP4DynamicQuantize",
            ["X", "scale"],
            ["X_quantized", "X_scale"],
            name="trt_fp4_dyn_quant",
            domain="trt",
            axis=-1,
            block_size=16,
            scale_type=17,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["X_scale", "dequant_scale"],
            ["X_scale_dequantized"],
            name="dequantize_scale_node",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["X_quantized", "X_scale_dequantized"],
            ["X_dequantized"],
            name="dequantize_data_node",
            axis=-1,
            block_size=16,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "nv_execution_provider_fp4_dynamic_quantize_graph",
        [value_info("X", TensorProto.FLOAT16, (64, 64))],
        [value_info("X_dequantized", TensorProto.FLOAT16, (64, 64))],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT16, [1], scale.tobytes(), raw=True),
            helper.make_tensor("dequant_scale", TensorProto.FLOAT16, [1], dequant_scale.tobytes(), raw=True),
        ],
        value_info=[
            value_info("X_quantized", fp4_type, (64, 64)),
            value_info("X_scale", fp8_type, (64, 4)),
            value_info("X_scale_dequantized", TensorProto.FLOAT16, (64, 4)),
        ],
    )
    save_model(make_model(graph, [helper.make_opsetid("", 23), helper.make_opsetid("trt", 1)], ir_version=10), path)


def create_clip_model(path, min_value, max_value):
    inputs = [value_info("X", TensorProto.FLOAT, (1, 4))]
    outputs = [value_info("Y", TensorProto.FLOAT, (1, 4))]
    initializers = [
        helper.make_tensor("min", TensorProto.FLOAT, [], [min_value]),
        helper.make_tensor("max", TensorProto.FLOAT, [], [max_value]),
    ]
    nodes = [helper.make_node("Clip", ["X", "min", "max"], ["Y"], name="clip")]
    graph = helper.make_graph(nodes, "ClipGraph", inputs, outputs, initializer=initializers)
    save_model(make_model(graph), path)


def create_pool_model(path, op_type):
    inputs = [value_info("X", TensorProto.FLOAT, (1, 2, 5, 5))]
    outputs = [value_info("Y", TensorProto.FLOAT, (1, 2, 1, 1))]
    nodes = [
        helper.make_node(
            op_type,
            ["X"],
            ["Y"],
            name="pool",
            kernel_shape=[3, 3],
            dilations=[2, 2],
        )
    ]
    graph = helper.make_graph(nodes, f"{op_type}DilationGraph", inputs, outputs)
    save_model(make_model(graph, [helper.make_opsetid("", 22)], ir_version=10), path)
