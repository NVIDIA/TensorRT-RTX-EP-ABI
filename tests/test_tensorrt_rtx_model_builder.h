// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Header-only ONNX model builders using raw protobuf API.
// No internal ORT headers — only onnx/onnx_pb.h.
#pragma once

#include <onnx/onnx_pb.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace model_builder {

// ---------------------------------------------------------------------------
// Float32 -> IEEE 754 binary16 (FLOAT16) conversion.
// Truncates mantissa; handles zero/inf/NaN. Subnormals flushed to zero.
// ---------------------------------------------------------------------------
inline uint16_t Float32ToFloat16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign     = (x >> 16) & 0x8000u;
    const int32_t  exp32    = static_cast<int32_t>((x >> 23) & 0xFFu);
    const uint32_t mantissa = (x >> 13) & 0x3FFu;
    if (exp32 == 0xFF) {
        // Inf or NaN — preserve NaN mantissa at least partially.
        const uint16_t m = ((x & 0x7FFFFFu) != 0u) ? 0x1u : 0u;
        return static_cast<uint16_t>(sign | 0x7C00u | m);
    }
    const int32_t exp16 = exp32 - 127 + 15;
    if (exp16 <= 0) return static_cast<uint16_t>(sign);
    if (exp16 >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp16) << 10) | mantissa);
}

// ---------------------------------------------------------------------------
// Protobuf helpers
// ---------------------------------------------------------------------------

// Add a ValueInfoProto (graph input or output) with optional shape.
// dims={} means no shape (unknown). dim value -1 becomes a symbolic param.
inline void AddValueInfo(
    google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>* list,
    const std::string& name,
    int32_t elem_type,
    const std::vector<int>& dims) {
    auto* vi = list->Add();
    vi->set_name(name);
    auto* tensor_type = vi->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(elem_type);
    if (!dims.empty()) {
        auto* shape = tensor_type->mutable_shape();
        int dyn_idx = 0;
        for (int d : dims) {
            auto* dim = shape->add_dim();
            if (d == -1) {
                dim->set_dim_param(name + "_dim_" + std::to_string(dyn_idx++));
            } else {
                dim->set_dim_value(d);
            }
        }
    }
}

// Add a node to the graph.
inline onnx::NodeProto* AddNode(
    onnx::GraphProto* graph,
    const std::string& name,
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::string& domain = "") {
    auto* node = graph->add_node();
    node->set_name(name);
    node->set_op_type(op_type);
    if (!domain.empty()) node->set_domain(domain);
    for (auto& in : inputs) node->add_input(in);
    for (auto& out : outputs) node->add_output(out);
    return node;
}

// Serialize model to file.
inline void SaveModel(const onnx::ModelProto& model,
                      const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    if (!model.SerializeToOstream(&out)) {
        throw std::runtime_error("Failed to serialize model to: " + path);
    }
}

// ---------------------------------------------------------------------------
// CreateBaseModel
// ---------------------------------------------------------------------------
//
// Builds a simple Add-chain model:
//
//   X  Y          inputs: X, Y, Z (shape=dims), S (scalar [1])
//    \ /           output: O
//  Z  Add
//   \ /
//    Add
//     |
//  [FastGelu]      (optional, com.microsoft domain)
//     |
//    Add  S
//     \ /
//      O
//
// Dynamic dims: -1 becomes a symbolic dim parameter.
// dtype: one of onnx::TensorProto_DataType values (FLOAT, FLOAT16, etc.)
//
inline void CreateBaseModel(
    const std::string& path,
    const std::string& graph_name,
    const std::vector<int>& dims,
    bool add_fast_gelu = false,
    int32_t dtype = onnx::TensorProto_DataType_FLOAT) {
    onnx::ModelProto model;
    model.set_ir_version(7);

    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    if (add_fast_gelu) {
        auto* ms_opset = model.add_opset_import();
        ms_opset->set_domain("com.microsoft");
        ms_opset->set_version(1);
    }

    auto* graph = model.mutable_graph();
    graph->set_name(graph_name);

    // Graph inputs
    AddValueInfo(graph->mutable_input(), "X", dtype, dims);
    AddValueInfo(graph->mutable_input(), "Y", dtype, dims);
    AddValueInfo(graph->mutable_input(), "Z", dtype, dims);
    AddValueInfo(graph->mutable_input(), "S", dtype, {1});

    // Graph output (no shape = dynamic)
    AddValueInfo(graph->mutable_output(), "O", dtype, {});

    // node_1: Add(X, Y) -> node_1_out_1
    AddNode(graph, "node_1", "Add", {"X", "Y"}, {"node_1_out_1"});

    // node_2: Add(node_1_out_1, Z) -> node_2_out_1
    AddNode(graph, "node_2", "Add", {"node_1_out_1", "Z"}, {"node_2_out_1"});

    std::string last_output = "node_2_out_1";

    if (add_fast_gelu) {
        // node_3: FastGelu(node_2_out_1) -> node_3_out_1  (com.microsoft domain)
        AddNode(graph, "node_3", "FastGelu", {"node_2_out_1"}, {"node_3_out_1"},
                "com.microsoft");
        last_output = "node_3_out_1";
    }

    // node_5: Add(last_output, S) -> O
    AddNode(graph, "node_5", "Add", {last_output, "S"}, {"O"});

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateAsymmetricDqMatMulFastGeluModel
// ---------------------------------------------------------------------------
//
// Builds a split-partition graph designed to validate two properties:
//   1. A constant-weight asymmetric DequantizeLinear(zero_point != 0) is
//      lowered/folded so the MatMul partition remains TRT-RTX compilable.
//   2. FastGelu (com.microsoft domain) stays outside the TRT-RTX partition.
//
// Graph:
//
//   X (float[2,3]) --------------------------.
//                                            v
//   Wq (int8[3,2]) -- DQ(zp=3, scale=0.25) -> MatMul -> FastGelu -> O
//
// Expected compiled model shape:
//   [EPContext] -> FastGelu
//
inline void CreateAsymmetricDqMatMulFastGeluModel(
    const std::string& path,
    const std::string& graph_name) {
    onnx::ModelProto model;
    model.set_ir_version(7);

    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* ms_opset = model.add_opset_import();
    ms_opset->set_domain("com.microsoft");
    ms_opset->set_version(1);

    auto* graph = model.mutable_graph();
    graph->set_name(graph_name);

    constexpr int kRows = 2;
    constexpr int kInner = 3;
    constexpr int kCols = 2;

    AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT,
                 {kRows, kInner});
    AddValueInfo(graph->mutable_output(), "O", onnx::TensorProto_DataType_FLOAT,
                 {kRows, kCols});
    AddValueInfo(graph->mutable_value_info(), "dequantizedWeights",
                 onnx::TensorProto_DataType_FLOAT, {kInner, kCols});
    AddValueInfo(graph->mutable_value_info(), "matmulOutput",
                 onnx::TensorProto_DataType_FLOAT, {kRows, kCols});

    {
        auto* t = graph->add_initializer();
        t->set_name("Wq");
        t->set_data_type(onnx::TensorProto_DataType_INT8);
        t->add_dims(kInner);
        t->add_dims(kCols);
        const std::vector<int8_t> weights = {16, -8, 5, 12, -3, 9};
        t->set_raw_data(weights.data(),
                        static_cast<int>(weights.size() * sizeof(int8_t)));
    }
    {
        auto* t = graph->add_initializer();
        t->set_name("weightScale");
        t->set_data_type(onnx::TensorProto_DataType_FLOAT);
        t->add_dims(1);
        t->add_float_data(0.25f);
    }
    {
        auto* t = graph->add_initializer();
        t->set_name("weightZeroPoint");
        t->set_data_type(onnx::TensorProto_DataType_INT8);
        t->add_dims(1);
        const int8_t zero_point = 3;
        t->set_raw_data(&zero_point, sizeof(zero_point));
    }

    AddNode(graph, "dq_weights", "DequantizeLinear",
            {"Wq", "weightScale", "weightZeroPoint"},
            {"dequantizedWeights"});
    AddNode(graph, "matmul", "MatMul", {"X", "dequantizedWeights"},
            {"matmulOutput"});
    AddNode(graph, "fast_gelu", "FastGelu", {"matmulOutput"}, {"O"},
            "com.microsoft");

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateAsymmetricQdqMatMulFastGeluModel
// ---------------------------------------------------------------------------
//
// Builds a split-partition graph that exercises asymmetric Q + DQ lowering on
// an activation path before a TRT-friendly MatMul, followed by FastGelu on CPU.
//
// Graph:
//
//   X (float[2,3]) -- Q(zp=5, scale=0.25) -- DQ(zp=5, scale=0.25) --.
//                                                                    v
//   W (float[3,2]) -----------------------------------------------> MatMul -> FastGelu -> O
//
// Expected compiled / runtime shape:
//   [Q/DQ lowering + MatMul on TRT-RTX] -> FastGelu on CPU
//
inline void CreateAsymmetricQdqMatMulFastGeluModel(
    const std::string& path,
    const std::string& graph_name) {
    onnx::ModelProto model;
    model.set_ir_version(7);

    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* ms_opset = model.add_opset_import();
    ms_opset->set_domain("com.microsoft");
    ms_opset->set_version(1);

    auto* graph = model.mutable_graph();
    graph->set_name(graph_name);

    constexpr int kRows = 2;
    constexpr int kInner = 3;
    constexpr int kCols = 2;

    AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT,
                 {kRows, kInner});
    AddValueInfo(graph->mutable_output(), "O", onnx::TensorProto_DataType_FLOAT,
                 {kRows, kCols});
    AddValueInfo(graph->mutable_value_info(), "quantizedX",
                 onnx::TensorProto_DataType_INT8, {kRows, kInner});
    AddValueInfo(graph->mutable_value_info(), "dequantizedX",
                 onnx::TensorProto_DataType_FLOAT, {kRows, kInner});
    AddValueInfo(graph->mutable_value_info(), "matmulOutput",
                 onnx::TensorProto_DataType_FLOAT, {kRows, kCols});

    {
        auto* t = graph->add_initializer();
        t->set_name("activationScale");
        t->set_data_type(onnx::TensorProto_DataType_FLOAT);
        t->add_dims(1);
        t->add_float_data(0.25f);
    }
    {
        auto* t = graph->add_initializer();
        t->set_name("activationZeroPoint");
        t->set_data_type(onnx::TensorProto_DataType_INT8);
        t->add_dims(1);
        const int8_t zero_point = 5;
        t->set_raw_data(&zero_point, sizeof(zero_point));
    }
    {
        auto* t = graph->add_initializer();
        t->set_name("W");
        t->set_data_type(onnx::TensorProto_DataType_FLOAT);
        t->add_dims(kInner);
        t->add_dims(kCols);
        const std::vector<float> weights = {
            1.25f, -0.75f,
            0.50f,  2.00f,
           -1.50f,  0.25f};
        for (float w : weights) {
            t->add_float_data(w);
        }
    }

    AddNode(graph, "quantize_x", "QuantizeLinear",
            {"X", "activationScale", "activationZeroPoint"},
            {"quantizedX"});
    AddNode(graph, "dequantize_x", "DequantizeLinear",
            {"quantizedX", "activationScale", "activationZeroPoint"},
            {"dequantizedX"});
    AddNode(graph, "matmul", "MatMul", {"dequantizedX", "W"},
            {"matmulOutput"});
    AddNode(graph, "fast_gelu", "FastGelu", {"matmulOutput"}, {"O"},
            "com.microsoft");

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateTopkAndMultipleGraphOutputsModel
// ---------------------------------------------------------------------------
//
// C++ translation of testdata/topk_and_multiple_graph_outputs.py
//
// input: "input" [N] (float, dynamic dim)
// outputs: "scores" (float), "Less_output_0" (bool),
//          "Div_17_output_0" (int64), "labels" (int64)
//
inline void CreateTopkAndMultipleGraphOutputsModel(const std::string& path) {
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("TopKGraph");

    // Input: "input" with dynamic dim N
    {
        auto* vi = graph->mutable_input()->Add();
        vi->set_name("input");
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(onnx::TensorProto_DataType_FLOAT);
        tt->mutable_shape()->add_dim()->set_dim_param("N");
    }

    // Initializers
    {
        // K = [300]
        auto* k = graph->add_initializer();
        k->set_name("K");
        k->set_data_type(onnx::TensorProto_DataType_INT64);
        k->add_dims(1);
        k->add_int64_data(300);
    }
    {
        // zero = 0 (scalar)
        auto* z = graph->add_initializer();
        z->set_name("zero");
        z->set_data_type(onnx::TensorProto_DataType_INT64);
        z->add_int64_data(0);
    }
    {
        // twenty_six = 26 (scalar)
        auto* ts = graph->add_initializer();
        ts->set_name("twenty_six");
        ts->set_data_type(onnx::TensorProto_DataType_INT64);
        ts->add_int64_data(26);
    }

    // Nodes
    AddNode(graph, "TopK", "TopK", {"input", "K"}, {"scores", "topk_indices"});
    AddNode(graph, "Less", "Less", {"topk_indices", "zero"}, {"Less_output_0"});
    AddNode(graph, "Div", "Div", {"topk_indices", "twenty_six"}, {"Div_17_output_0"});
    AddNode(graph, "Mod", "Mod", {"topk_indices", "twenty_six"}, {"labels"});

    // Outputs
    {
        auto* vi = graph->mutable_output()->Add();
        vi->set_name("scores");
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(onnx::TensorProto_DataType_FLOAT);
        tt->mutable_shape()->add_dim()->set_dim_param("K");
    }
    {
        auto* vi = graph->mutable_output()->Add();
        vi->set_name("Less_output_0");
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(onnx::TensorProto_DataType_BOOL);
        tt->mutable_shape()->add_dim()->set_dim_param("K");
    }
    {
        auto* vi = graph->mutable_output()->Add();
        vi->set_name("Div_17_output_0");
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(onnx::TensorProto_DataType_INT64);
        tt->mutable_shape()->add_dim()->set_dim_param("K");
    }
    {
        auto* vi = graph->mutable_output()->Add();
        vi->set_name("labels");
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(onnx::TensorProto_DataType_INT64);
        tt->mutable_shape()->add_dim()->set_dim_param("K");
    }

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateNodeOutputNotUsedModel
// ---------------------------------------------------------------------------
//
// C++ translation of testdata/node_output_not_used.py
//
// inputs: "X" [3,2] (float), "W" [2,3] (float)
// output: "Y" (float)
// Dropout produces two outputs; the mask is unused.
//
inline void CreateNodeOutputNotUsedModel(const std::string& path) {
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("DropoutMatMulGraph");

    // Inputs
    AddValueInfo(graph->mutable_input(), "X",
                 onnx::TensorProto_DataType_FLOAT, {3, 2});
    AddValueInfo(graph->mutable_input(), "W",
                 onnx::TensorProto_DataType_FLOAT, {2, 3});

    // Output
    AddValueInfo(graph->mutable_output(), "Y",
                 onnx::TensorProto_DataType_FLOAT, {3, 3});

    // Dropout(X) -> dropout_out, dropout_mask  (mask unused)
    AddNode(graph, "DropoutNode", "Dropout", {"X"}, {"dropout_out", "dropout_mask"});

    // MatMul(dropout_out, W) -> Y
    AddNode(graph, "MatMulNode", "MatMul", {"dropout_out", "W"}, {"Y"});

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateFP8CustomOpModel
// ---------------------------------------------------------------------------
//
// Builds:
//   X (FP16 [4,64]) -> TRT_FP8QuantizeLinear(trt domain) ->
//   X_quantized (FP8E4M3FN) -> TRT_FP8DequantizeLinear(trt domain) -> Y (FP16)
//
// The scale is a FP16 initializer shared by both nodes.
// The TRT RTX EP registers the "trt" domain custom ops at EP creation time;
// the raw protobuf references them by name + domain without needing a schema.
//
inline void CreateFP8CustomOpModel(const std::string& path,
                                   const std::string& graph_name) {
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* onnx_opset = model.add_opset_import();
    onnx_opset->set_domain("");
    onnx_opset->set_version(19);
    auto* trt_opset = model.add_opset_import();
    trt_opset->set_domain("trt");
    trt_opset->set_version(1);

    auto* graph = model.mutable_graph();
    graph->set_name(graph_name);

    const std::vector<int> dims = {4, 64};
    constexpr int32_t kFP16 = onnx::TensorProto_DataType_FLOAT16;
    constexpr int32_t kFP8  = onnx::TensorProto_DataType_FLOAT8E4M3FN;

    AddValueInfo(graph->mutable_input(),  "X", kFP16, dims);
    AddValueInfo(graph->mutable_output(), "Y", kFP16, dims);

    // scale (FP16 scalar, 0.0078125f)
    {
        auto* t = graph->add_initializer();
        t->set_name("scale");
        t->set_data_type(kFP16);
        t->add_dims(1);
        const uint16_t v = Float32ToFloat16(0.0078125f);
        t->set_raw_data(&v, sizeof(v));
    }

    AddNode(graph, "trt_fp8_quantize_node",   "TRT_FP8QuantizeLinear",
            {"X", "scale"}, {"X_quantized"}, "trt");
    AddNode(graph, "trt_fp8_dequantize_node", "TRT_FP8DequantizeLinear",
            {"X_quantized", "scale"}, {"Y"}, "trt");

    // Intermediate value info for the FP8 tensor (helps some parsers).
    {
        auto* vi = graph->mutable_value_info()->Add();
        vi->set_name("X_quantized");
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(kFP8);
        for (int d : dims) tt->mutable_shape()->add_dim()->set_dim_value(d);
    }

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateFP4CustomOpModel
// ---------------------------------------------------------------------------
//
// Builds:
//   X (FP16 [64,64]) -> TRT_FP4DynamicQuantize(trt domain) ->
//        X_quantized (FP4E2M1 [64,64]), X_scale (FP8E4M3FN [64,4])
//   X_scale -> DequantizeLinear -> X_scale_dequantized (FP16 [64,4])
//   X_quantized, X_scale_dequantized -> DequantizeLinear (axis=-1, block_size=16)
//        -> X_dequantized (FP16 [64,64])
//
inline void CreateFP4CustomOpModel(const std::string& path,
                                   const std::string& graph_name) {
    onnx::ModelProto model;
    model.set_ir_version(10);
    auto* onnx_opset = model.add_opset_import();
    onnx_opset->set_domain("");
    onnx_opset->set_version(23);
    auto* trt_opset = model.add_opset_import();
    trt_opset->set_domain("trt");
    trt_opset->set_version(1);

    auto* graph = model.mutable_graph();
    graph->set_name(graph_name);

    constexpr int32_t kFP16 = onnx::TensorProto_DataType_FLOAT16;
    constexpr int32_t kFP8  = onnx::TensorProto_DataType_FLOAT8E4M3FN;
    constexpr int32_t kFP4  = onnx::TensorProto_DataType_FLOAT4E2M1;
    const std::vector<int> input_dims  = {64, 64};
    const std::vector<int> scale_dims  = {64, 4};

    AddValueInfo(graph->mutable_input(),  "X",             kFP16, input_dims);
    AddValueInfo(graph->mutable_output(), "X_dequantized", kFP16, input_dims);

    // scale (FP16 scalar, 0.1234f)
    {
        auto* t = graph->add_initializer();
        t->set_name("scale");
        t->set_data_type(kFP16);
        t->add_dims(1);
        const uint16_t v = Float32ToFloat16(0.1234f);
        t->set_raw_data(&v, sizeof(v));
    }
    // dequant_scale (FP16 scalar, 0.0625f)
    {
        auto* t = graph->add_initializer();
        t->set_name("dequant_scale");
        t->set_data_type(kFP16);
        t->add_dims(1);
        const uint16_t v = Float32ToFloat16(0.0625f);
        t->set_raw_data(&v, sizeof(v));
    }

    // TRT_FP4DynamicQuantize(X, scale) -> X_quantized, X_scale
    {
        auto* node = AddNode(graph, "trt_fp4_dyn_quant",
                             "TRT_FP4DynamicQuantize",
                             {"X", "scale"},
                             {"X_quantized", "X_scale"},
                             "trt");

        auto* a_axis = node->add_attribute();
        a_axis->set_name("axis");
        a_axis->set_type(onnx::AttributeProto_AttributeType_INT);
        a_axis->set_i(-1);

        auto* a_block = node->add_attribute();
        a_block->set_name("block_size");
        a_block->set_type(onnx::AttributeProto_AttributeType_INT);
        a_block->set_i(16);

        auto* a_stype = node->add_attribute();
        a_stype->set_name("scale_type");
        a_stype->set_type(onnx::AttributeProto_AttributeType_INT);
        a_stype->set_i(17);
    }

    // DequantizeLinear(X_scale, dequant_scale) -> X_scale_dequantized
    AddNode(graph, "dequantize_scale_node", "DequantizeLinear",
            {"X_scale", "dequant_scale"}, {"X_scale_dequantized"});

    // DequantizeLinear(X_quantized, X_scale_dequantized) -> X_dequantized
    // (axis=-1, block_size=16)
    {
        auto* node = AddNode(graph, "dequantize_data_node", "DequantizeLinear",
                             {"X_quantized", "X_scale_dequantized"},
                             {"X_dequantized"});

        auto* a_axis = node->add_attribute();
        a_axis->set_name("axis");
        a_axis->set_type(onnx::AttributeProto_AttributeType_INT);
        a_axis->set_i(-1);

        auto* a_block = node->add_attribute();
        a_block->set_name("block_size");
        a_block->set_type(onnx::AttributeProto_AttributeType_INT);
        a_block->set_i(16);
    }

    // Intermediate value info (optional, but helpful for downstream tools).
    auto add_vi = [&](const std::string& name, int32_t dtype,
                      const std::vector<int>& shape) {
        auto* vi = graph->mutable_value_info()->Add();
        vi->set_name(name);
        auto* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(dtype);
        for (int d : shape) tt->mutable_shape()->add_dim()->set_dim_value(d);
    };
    add_vi("X_quantized",         kFP4,  input_dims);
    add_vi("X_scale",             kFP8,  scale_dims);
    add_vi("X_scale_dequantized", kFP16, scale_dims);

    SaveModel(model, path);
}

// ---------------------------------------------------------------------------
// CreateLargeModel
// ---------------------------------------------------------------------------
//
// Builds a multi-layer MatMul chain with large FP16 weights stored as
// external data. Designed to exercise the `SetInputModelFromBuffer` +
// `AddExternalInitializersFromFilesInMemory` and
// `SetOutputModelExternalInitializersFile` code paths.
//
// input: "X" [1, hidden]  (FP16)
// output: "O" [1, hidden] (FP16)
//
// Size control: num_layers * hidden * hidden * 2 bytes.
// Default: 8 layers of [1024,1024] FP16 weights ~= 16 MB of weights.
//
inline void CreateLargeModel(const std::string& model_path,
                             const std::string& external_data_path,
                             int num_layers = 8,
                             int hidden     = 1024) {
    // ---- 1. Build the weight data file ------------------------------------
    // All per-layer weights are concatenated in a single binary file.
    // Each weight is hidden*hidden FP16 values.
    const size_t per_layer_elems = static_cast<size_t>(hidden) * hidden;
    const size_t per_layer_bytes = per_layer_elems * sizeof(uint16_t);

    std::ofstream data_out(external_data_path, std::ios::binary);
    if (!data_out.is_open()) {
        throw std::runtime_error(
            "Cannot open external data file for writing: " + external_data_path);
    }

    // Deterministic pseudo-random fp16 values (small magnitude, bounded).
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 0x2800);  // small fp16 magnitudes

    std::vector<uint16_t> buffer(per_layer_elems);
    for (int l = 0; l < num_layers; ++l) {
        for (auto& v : buffer) v = static_cast<uint16_t>(dist(rng));
        data_out.write(reinterpret_cast<const char*>(buffer.data()),
                       static_cast<std::streamsize>(per_layer_bytes));
    }
    if (!data_out.good()) {
        throw std::runtime_error(
            "Failed to write external data to: " + external_data_path);
    }
    data_out.close();

    // ---- 2. Build the ONNX model protobuf ---------------------------------
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("large_model");

    constexpr int32_t kFP16 = onnx::TensorProto_DataType_FLOAT16;

    AddValueInfo(graph->mutable_input(),  "X", kFP16, {1, hidden});
    AddValueInfo(graph->mutable_output(), "O", kFP16, {1, hidden});

    // Extract the filename (location inside ONNX must be a relative path).
    std::string location = external_data_path;
    {
        const auto pos = location.find_last_of("/\\");
        if (pos != std::string::npos) location = location.substr(pos + 1);
    }

    std::string prev = "X";
    for (int l = 0; l < num_layers; ++l) {
        const std::string w_name   = "W" + std::to_string(l);
        const std::string mm_out   = "mm" + std::to_string(l);
        const std::string relu_out = (l == num_layers - 1)
                                         ? std::string("O")
                                         : ("r" + std::to_string(l));

        // Weight initializer — stored as external data.
        auto* t = graph->add_initializer();
        t->set_name(w_name);
        t->set_data_type(kFP16);
        t->add_dims(hidden);
        t->add_dims(hidden);
        t->set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
        {
            auto* e = t->add_external_data();
            e->set_key("location");
            e->set_value(location);
        }
        {
            auto* e = t->add_external_data();
            e->set_key("offset");
            e->set_value(std::to_string(
                static_cast<long long>(static_cast<size_t>(l) * per_layer_bytes)));
        }
        {
            auto* e = t->add_external_data();
            e->set_key("length");
            e->set_value(std::to_string(static_cast<long long>(per_layer_bytes)));
        }

        // X @ W -> mm
        AddNode(graph, "mm_" + std::to_string(l), "MatMul",
                {prev, w_name}, {mm_out});
        // Relu(mm) -> next
        AddNode(graph, "relu_" + std::to_string(l), "Relu",
                {mm_out}, {relu_out});
        prev = relu_out;
    }

    SaveModel(model, model_path);
}

// ---------------------------------------------------------------------------
// CreateSyntheticEPContextModel
// ---------------------------------------------------------------------------
//
// Creates an ONNX model with a single EPContext node (com.microsoft domain)
// with an optional "source" attribute. Used to test that the TRT RTX EP
// correctly skips EPContext nodes whose source belongs to a different EP.
//
inline void CreateSyntheticEPContextModel(const std::string& model_path,
                                          const std::string& source_attr,
                                          bool include_source_attr = true) {
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* onnx_opset = model.add_opset_import();
    onnx_opset->set_domain("");
    onnx_opset->set_version(11);
    auto* ms_opset = model.add_opset_import();
    ms_opset->set_domain("com.microsoft");
    ms_opset->set_version(1);

    auto* graph = model.mutable_graph();
    graph->set_name("EPContextSourceTest");

    AddValueInfo(graph->mutable_input(),  "input",
                 onnx::TensorProto_DataType_FLOAT, {1, 3});
    AddValueInfo(graph->mutable_output(), "output",
                 onnx::TensorProto_DataType_FLOAT, {1, 3});

    auto* node = AddNode(graph, "ep_context_node", "EPContext",
                         {"input"}, {"output"}, "com.microsoft");

    // embed_mode attribute
    {
        auto* a = node->add_attribute();
        a->set_name("embed_mode");
        a->set_type(onnx::AttributeProto_AttributeType_INT);
        a->set_i(1);
    }
    // ep_cache_context attribute (dummy data)
    {
        auto* a = node->add_attribute();
        a->set_name("ep_cache_context");
        a->set_type(onnx::AttributeProto_AttributeType_STRING);
        a->set_s("dummy_context_data");
    }
    // source attribute (optional)
    if (include_source_attr) {
        auto* a = node->add_attribute();
        a->set_name("source");
        a->set_type(onnx::AttributeProto_AttributeType_STRING);
        a->set_s(source_attr);
    }

    SaveModel(model, model_path);
}

// ---------------------------------------------------------------------------
// CreateLargeScratchModel
// ---------------------------------------------------------------------------
//
// Builds an fp32 model whose TRT runtime workspace (getDeviceMemorySizeV2 /
// updateDeviceMemorySizeForShapes) is dominated by a single large
// intermediate tensor T1 of shape [1, M, M].
//
//     X  [1, M, K]                      (input,  ~ M*K*4 bytes)
//       \
//        MatMul ---- W1 [K, M]          (weight, ~ M*K*4 bytes, zero-filled)
//       /
//     T1 [1, M, M]                      <-- dominates TRT scratch (M*M*4 bytes)
//       |
//      Relu
//       |
//     T2 [1, M, M]                      (same shape as T1, also in scratch)
//       \
//        MatMul ---- W2 [M, K]          (weight, ~ M*K*4 bytes, zero-filled)
//       /
//      Y  [1, M, K]                     (output, ~ M*K*4 bytes)
//
// The Relu between the two MatMuls is crucial: it breaks the algebraic
// identity `MatMul(MatMul(X, W1), W2) == MatMul(X, W1 @ W2)`, which would
// otherwise let an aggressive optimizer constant-fold `W1 @ W2` into a tiny
// [K, K] weight and skip materialising T1 entirely. With Relu in place, the
// [1, M, M] intermediate must be computed -> TRT's scratch allocation scales
// as O(M^2) as intended.
//
// Because K is small (16) and M is large, the weights and graph I/O stay
// small (~ M*K*4 bytes each, so the on-disk model is only a few MB) while
// T1 / T2 of size M*M*4 bytes dominate runtime scratch:
//
//     M = 16384 -> on-disk ~3 MB,   scratch ~ 1024 MB   (1 GB)
//     M = 23170 -> on-disk ~3 MB,   scratch ~ 2048 MB   (2 GB)
//     M = 32768 -> on-disk ~5 MB,   scratch ~ 4096 MB   (4 GB)
//
// Empirically on RTX 5090 the measured scratch is M*M*4 bytes plus a fixed
// ~7 MB TRT overhead.
//
// The absolute scratch numbers depend on TRT algo selection and device
// memory alignment; the unit tests print the observed cudaMemGetInfo delta
// so the caller can tune M for their device.
//
inline void CreateLargeScratchModel(
    const std::string& path,
    int M,
    int K = 16) {
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("LargeScratchGraph");

    AddValueInfo(graph->mutable_input(), "X",
                 onnx::TensorProto_DataType_FLOAT, {1, M, K});
    AddValueInfo(graph->mutable_output(), "Y",
                 onnx::TensorProto_DataType_FLOAT, {1, M, K});

    // Zero-filled weights: TRT only cares about shape/dtype at load time. We
    // keep K small so each weight is only a few MB regardless of M.
    auto add_weight = [&](const std::string& name, int d0, int d1) {
        auto* init = graph->add_initializer();
        init->set_name(name);
        init->set_data_type(onnx::TensorProto_DataType_FLOAT);
        init->add_dims(d0);
        init->add_dims(d1);
        const size_t num_bytes = static_cast<size_t>(d0) *
                                 static_cast<size_t>(d1) * sizeof(float);
        init->mutable_raw_data()->assign(num_bytes, '\0');
    };
    add_weight("W1", K, M);
    add_weight("W2", M, K);

    AddNode(graph, "matmul_1", "MatMul", {"X",  "W1"}, {"T1"});
    AddNode(graph, "relu_1",   "Relu",   {"T1"},       {"T2"});
    AddNode(graph, "matmul_2", "MatMul", {"T2", "W2"}, {"Y"});

    SaveModel(model, path);
}

}  // namespace model_builder
