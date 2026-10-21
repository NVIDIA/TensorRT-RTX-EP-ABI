// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "test_tensorrt_rtx_model_builder.h"

namespace control_flow_models
{

inline onnx::TensorProto* AddInt64Constant(onnx::GraphProto* graph, const std::string& node_name,
                                           const std::string& output_name, const std::vector<int64_t>& values,
                                           const std::vector<int64_t>& dims)
{
    auto* node = model_builder::AddNode(graph, node_name, "Constant", {}, {output_name});
    auto* attribute = node->add_attribute();
    attribute->set_name("value");
    attribute->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    auto* tensor = attribute->mutable_t();
    tensor->set_data_type(onnx::TensorProto_DataType_INT64);
    for (const int64_t dim : dims)
        tensor->add_dims(dim);
    for (const int64_t value : values)
        tensor->add_int64_data(value);
    return tensor;
}

inline onnx::TensorProto* AddFloatConstant(onnx::GraphProto* graph, const std::string& node_name,
                                           const std::string& output_name, const std::vector<float>& values,
                                           const std::vector<int64_t>& dims)
{
    auto* node = model_builder::AddNode(graph, node_name, "Constant", {}, {output_name});
    auto* attribute = node->add_attribute();
    attribute->set_name("value");
    attribute->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    auto* tensor = attribute->mutable_t();
    tensor->set_data_type(onnx::TensorProto_DataType_FLOAT);
    for (const int64_t dim : dims)
        tensor->add_dims(dim);
    for (const float value : values)
        tensor->add_float_data(value);
    return tensor;
}

inline onnx::TensorProto* AddBoolConstant(onnx::GraphProto* graph, const std::string& node_name,
                                          const std::string& output_name, bool value)
{
    auto* node = model_builder::AddNode(graph, node_name, "Constant", {}, {output_name});
    auto* attribute = node->add_attribute();
    attribute->set_name("value");
    attribute->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    auto* tensor = attribute->mutable_t();
    tensor->set_data_type(onnx::TensorProto_DataType_BOOL);
    tensor->add_int32_data(value ? 1 : 0);
    return tensor;
}

inline void AddIntAttribute(onnx::NodeProto* node, const std::string& name, int64_t value)
{
    auto* attribute = node->add_attribute();
    attribute->set_name(name);
    attribute->set_type(onnx::AttributeProto_AttributeType_INT);
    attribute->set_i(value);
}

inline onnx::GraphProto* AddGraphAttribute(onnx::NodeProto* node, const std::string& attribute_name,
                                           const std::string& graph_name)
{
    auto* attribute = node->add_attribute();
    attribute->set_name(attribute_name);
    attribute->set_type(onnx::AttributeProto_AttributeType_GRAPH);
    attribute->mutable_g()->set_name(graph_name);
    return attribute->mutable_g();
}

inline void AddScalarValueInfo(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>* list, const std::string& name,
                               int32_t elem_type)
{
    model_builder::AddValueInfo(list, name, elem_type, {});
    // AddValueInfo intentionally omits the shape field for an empty dimension
    // list. Loop's CPU kernel distinguishes that unknown-rank form from an
    // explicitly declared rank-0 scalar, so materialize the empty shape here.
    list->Mutable(list->size() - 1)->mutable_type()->mutable_tensor_type()->mutable_shape();
}

inline void CreateSimpleIfModel(const std::string& path)
{
    auto model = model_builder::CreateModelWithDefaultOpset("SimpleIfGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_input(), "cond", onnx::TensorProto_DataType_BOOL, {});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {2, 3});

    auto* if_node = model_builder::AddNode(graph, "simple_if", "If", {"cond"}, {"Y"});
    auto* then_graph = AddGraphAttribute(if_node, "then_branch", "simple_then");
    model_builder::AddValueInfo(then_graph->mutable_output(), "then_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(then_graph, "then_abs", "Abs", {"X"}, {"then_out"});
    auto* else_graph = AddGraphAttribute(if_node, "else_branch", "simple_else");
    model_builder::AddValueInfo(else_graph->mutable_output(), "else_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(else_graph, "else_neg", "Neg", {"X"}, {"else_out"});
    model_builder::SaveModel(model, path);
}

// The branch consumes shape values produced by nodes in the parent graph. An
// isolated branch parse lacks that producer context, while parsing the intact
// parent graph provides it.
inline void CreateParentProducedShapeIfModel(const std::string& path, bool take_then_branch)
{
    auto model = model_builder::CreateModelWithDefaultOpset("ParentProducedShapeIfGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {});

    AddFloatConstant(graph, "one", "one_value", {1.0f}, {});
    model_builder::AddNode(graph, "div", "Div", {"X", "one_value"}, {"div_out"});
    auto* flatten = model_builder::AddNode(graph, "flatten", "Flatten", {"div_out"}, {"flat"});
    AddIntAttribute(flatten, "axis", 1);
    model_builder::AddNode(graph, "shape", "Shape", {"flat"}, {"flat_shape"});
    model_builder::AddNode(graph, "size", "Size", {"flat_shape"}, {"rank"});
    AddInt64Constant(graph, "rank_minus_one", "one_i64", {1}, {});
    model_builder::AddNode(graph, "sub", "Sub", {"rank", "one_i64"}, {"missing_dims"});
    AddInt64Constant(graph, "branch_threshold", "threshold", {take_then_branch ? 2 : 0}, {});
    model_builder::AddNode(graph, "less", "Less", {"missing_dims", "threshold"}, {"cond"});

    auto* if_node = model_builder::AddNode(graph, "parent_context_if", "If", {"cond"}, {"Y"});
    auto* then_graph = AddGraphAttribute(if_node, "then_branch", "parent_context_then");
    model_builder::AddValueInfo(then_graph->mutable_output(), "then_out", onnx::TensorProto_DataType_FLOAT, {});
    AddInt64Constant(then_graph, "constant_shape", "shape_one", {1}, {1});
    model_builder::AddNode(then_graph, "abs_missing_dims", "Abs", {"missing_dims"}, {"abs_missing"});
    model_builder::AddNode(then_graph, "reshape_missing_dims", "Reshape", {"abs_missing", "shape_one"},
                           {"missing_shape"});
    model_builder::AddNode(then_graph, "expand_ones", "Expand", {"shape_one", "missing_shape"}, {"leading_ones"});
    auto* concat = model_builder::AddNode(then_graph, "prepend_ones", "Concat", {"leading_ones", "flat_shape"},
                                          {"reshape_target"});
    AddIntAttribute(concat, "axis", 0);
    model_builder::AddNode(then_graph, "reshape_with_parent_shape", "Reshape", {"flat", "reshape_target"},
                           {"then_out"});

    auto* else_graph = AddGraphAttribute(if_node, "else_branch", "parent_context_else");
    model_builder::AddValueInfo(else_graph->mutable_output(), "else_out", onnx::TensorProto_DataType_FLOAT, {});
    model_builder::AddNode(else_graph, "identity_flat", "Identity", {"flat"}, {"else_out"});
    model_builder::SaveModel(model, path);
}

inline void CreateExcludedBranchIfModel(const std::string& path, bool exclude_then_branch)
{
    auto model = model_builder::CreateModelWithDefaultOpset("ExcludedBranchIfGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_input(), "cond", onnx::TensorProto_DataType_BOOL, {});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {2, 3});
    auto* if_node = model_builder::AddNode(graph, "excluded_branch_if", "If", {"cond"}, {"Y"});
    auto* then_graph = AddGraphAttribute(if_node, "then_branch", "then_branch");
    model_builder::AddValueInfo(then_graph->mutable_output(), "then_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(then_graph, exclude_then_branch ? "excluded_then_sin" : "eligible_then_abs",
                           exclude_then_branch ? "Sin" : "Abs", {"X"}, {"then_out"});
    auto* else_graph = AddGraphAttribute(if_node, "else_branch", "else_branch");
    model_builder::AddValueInfo(else_graph->mutable_output(), "else_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(else_graph, exclude_then_branch ? "eligible_else_neg" : "excluded_else_sin",
                           exclude_then_branch ? "Neg" : "Sin", {"X"}, {"else_out"});
    model_builder::SaveModel(model, path);
}

inline void CreateUnsupportedDataTypeBranchIfModel(const std::string& path)
{
    auto model = model_builder::CreateModelWithDefaultOpset("UnsupportedDataTypeBranchIfGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_input(), "cond", onnx::TensorProto_DataType_BOOL, {});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {2, 3});
    auto* if_node = model_builder::AddNode(graph, "unsupported_dtype_if", "If", {"cond"}, {"Y"});

    auto* then_graph = AddGraphAttribute(if_node, "then_branch", "unsupported_dtype_then");
    model_builder::AddValueInfo(then_graph->mutable_value_info(), "uint16_value", onnx::TensorProto_DataType_UINT16,
                                {2, 3});
    model_builder::AddValueInfo(then_graph->mutable_output(), "then_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(then_graph, "eligible_abs_before_cast", "Abs", {"X"}, {"absolute_value"});
    auto* cast_to_uint16 =
        model_builder::AddNode(then_graph, "cast_to_unsupported_uint16", "Cast", {"absolute_value"}, {"uint16_value"});
    AddIntAttribute(cast_to_uint16, "to", onnx::TensorProto_DataType_UINT16);
    auto* cast_to_float =
        model_builder::AddNode(then_graph, "cast_back_to_float", "Cast", {"uint16_value"}, {"then_out"});
    AddIntAttribute(cast_to_float, "to", onnx::TensorProto_DataType_FLOAT);

    auto* else_graph = AddGraphAttribute(if_node, "else_branch", "supported_else");
    model_builder::AddValueInfo(else_graph->mutable_output(), "else_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(else_graph, "eligible_else_neg", "Neg", {"X"}, {"else_out"});
    model_builder::SaveModel(model, path);
}

inline void CreateParserRejectedIfModel(const std::string& path)
{
    auto model = model_builder::CreateModelWithDefaultOpset("ParserRejectedIfGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 2});
    model_builder::AddValueInfo(graph->mutable_input(), "cond", onnx::TensorProto_DataType_BOOL, {});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {});
    auto* if_node = model_builder::AddNode(graph, "parser_rejected_if", "If", {"cond"}, {"Y"});
    auto* then_graph = AddGraphAttribute(if_node, "then_branch", "det_then");
    model_builder::AddValueInfo(then_graph->mutable_output(), "det_out", onnx::TensorProto_DataType_FLOAT, {});
    model_builder::AddNode(then_graph, "unsupported_det", "Det", {"X"}, {"det_out"});
    auto* else_graph = AddGraphAttribute(if_node, "else_branch", "reduce_else");
    model_builder::AddValueInfo(else_graph->mutable_output(), "sum_out", onnx::TensorProto_DataType_FLOAT, {});
    AddInt64Constant(else_graph, "axes", "reduce_axes", {0, 1}, {2});
    auto* reduce = model_builder::AddNode(else_graph, "reduce_sum", "ReduceSum", {"X", "reduce_axes"}, {"sum_out"});
    AddIntAttribute(reduce, "keepdims", 0);
    model_builder::SaveModel(model, path);
}

inline void CreateNestedExcludedIfModel(const std::string& path)
{
    auto model = model_builder::CreateModelWithDefaultOpset("NestedExcludedIfGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_input(), "outer_cond", onnx::TensorProto_DataType_BOOL, {});
    model_builder::AddValueInfo(graph->mutable_input(), "inner_cond", onnx::TensorProto_DataType_BOOL, {});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {2, 3});

    // Build a reshape target in the parent graph. The supported nested branch cannot be parsed in isolation because
    // it captures these producer values, but it becomes parseable as part of the intact parent graph.
    AddFloatConstant(graph, "one", "one_value", {1.0f}, {});
    model_builder::AddNode(graph, "div", "Div", {"X", "one_value"}, {"div_out"});
    auto* flatten = model_builder::AddNode(graph, "flatten", "Flatten", {"div_out"}, {"flat"});
    AddIntAttribute(flatten, "axis", 1);
    model_builder::AddNode(graph, "shape", "Shape", {"flat"}, {"flat_shape"});
    model_builder::AddNode(graph, "size", "Size", {"flat_shape"}, {"rank"});
    AddInt64Constant(graph, "rank_minus_one", "one_i64", {1}, {});
    model_builder::AddNode(graph, "sub", "Sub", {"rank", "one_i64"}, {"missing_dims"});

    auto* outer_if = model_builder::AddNode(graph, "outer_if", "If", {"outer_cond"}, {"Y"});
    auto* outer_then = AddGraphAttribute(outer_if, "then_branch", "outer_then");
    model_builder::AddValueInfo(outer_then->mutable_output(), "outer_then_out", onnx::TensorProto_DataType_FLOAT,
                                {2, 3});
    auto* inner_if = model_builder::AddNode(outer_then, "inner_if", "If", {"inner_cond"}, {"outer_then_out"});
    auto* inner_then = AddGraphAttribute(inner_if, "then_branch", "inner_then");
    model_builder::AddValueInfo(inner_then->mutable_output(), "inner_then_out", onnx::TensorProto_DataType_FLOAT,
                                {2, 3});
    AddInt64Constant(inner_then, "constant_shape", "shape_one", {1}, {1});
    model_builder::AddNode(inner_then, "abs_missing_dims", "Abs", {"missing_dims"}, {"abs_missing"});
    model_builder::AddNode(inner_then, "reshape_missing_dims", "Reshape", {"abs_missing", "shape_one"},
                           {"missing_shape"});
    model_builder::AddNode(inner_then, "expand_ones", "Expand", {"shape_one", "missing_shape"}, {"leading_ones"});
    auto* concat = model_builder::AddNode(inner_then, "prepend_ones", "Concat", {"leading_ones", "flat_shape"},
                                          {"reshape_target"});
    AddIntAttribute(concat, "axis", 0);
    model_builder::AddNode(inner_then, "reshape_with_parent_shape", "Reshape", {"flat", "reshape_target"},
                           {"expanded"});
    AddInt64Constant(inner_then, "squeeze_axes", "axis_zero", {0}, {1});
    model_builder::AddNode(inner_then, "remove_leading_dimension", "Squeeze", {"expanded", "axis_zero"},
                           {"restored_shape"});
    model_builder::AddNode(inner_then, "inner_abs", "Abs", {"restored_shape"}, {"inner_then_out"});
    auto* inner_else = AddGraphAttribute(inner_if, "else_branch", "inner_else");
    model_builder::AddValueInfo(inner_else->mutable_output(), "inner_else_out", onnx::TensorProto_DataType_FLOAT,
                                {2, 3});
    model_builder::AddNode(inner_else, "nested_excluded_sin", "Sin", {"X"}, {"inner_else_out"});
    auto* outer_else = AddGraphAttribute(outer_if, "else_branch", "outer_else");
    model_builder::AddValueInfo(outer_else->mutable_output(), "outer_else_out", onnx::TensorProto_DataType_FLOAT,
                                {2, 3});
    model_builder::AddNode(outer_else, "outer_neg", "Neg", {"X"}, {"outer_else_out"});
    model_builder::SaveModel(model, path);
}

inline void CreateLoopModel(const std::string& path, bool include_excluded_sin)
{
    auto model = model_builder::CreateModelWithDefaultOpset("LoopGraph", 13);
    auto* graph = model.mutable_graph();
    AddScalarValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT);
    AddScalarValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT);
    AddInt64Constant(graph, "trip_count", "trip", {3}, {});
    AddBoolConstant(graph, "initial_condition", "initial_cond", true);
    auto* loop = model_builder::AddNode(graph, "loop", "Loop", {"trip", "initial_cond", "X"}, {"Y"});
    auto* body = AddGraphAttribute(loop, "body", "loop_body");
    AddScalarValueInfo(body->mutable_input(), "iteration", onnx::TensorProto_DataType_INT64);
    AddScalarValueInfo(body->mutable_input(), "cond_in", onnx::TensorProto_DataType_BOOL);
    AddScalarValueInfo(body->mutable_input(), "value_in", onnx::TensorProto_DataType_FLOAT);
    AddScalarValueInfo(body->mutable_output(), "cond_out", onnx::TensorProto_DataType_BOOL);
    AddScalarValueInfo(body->mutable_output(), "value_out", onnx::TensorProto_DataType_FLOAT);
    AddFloatConstant(body, "increment", "one", {1.0f}, {});
    model_builder::AddNode(body, "add", "Add", {"value_in", "one"}, {"incremented"});
    if (include_excluded_sin)
        model_builder::AddNode(body, "excluded_loop_sin", "Sin", {"incremented"}, {"value_out"});
    else
        model_builder::AddNode(body, "loop_identity", "Identity", {"incremented"}, {"value_out"});
    model_builder::AddNode(body, "condition_identity", "Identity", {"cond_in"}, {"cond_out"});
    model_builder::SaveModel(model, path);
}

// The Loop body captures a reshape target produced in the parent graph. The
// body cannot resolve that value's producer when parsed in isolation, but the
// intact Loop can resolve it during the parent-context retry.
inline void CreateParentProducedShapeLoopModel(const std::string& path)
{
    auto model = model_builder::CreateModelWithDefaultOpset("ParentProducedShapeLoopGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, {2, 3});
    AddFloatConstant(graph, "parent_one", "parent_one_value", {1.0f}, {});
    model_builder::AddNode(graph, "parent_div", "Div", {"X", "parent_one_value"}, {"div_out"});
    auto* flatten = model_builder::AddNode(graph, "parent_flatten", "Flatten", {"div_out"}, {"flat"});
    AddIntAttribute(flatten, "axis", 1);
    model_builder::AddNode(graph, "parent_shape", "Shape", {"flat"}, {"flat_shape"});
    model_builder::AddNode(graph, "parent_size", "Size", {"flat_shape"}, {"rank"});
    AddInt64Constant(graph, "rank_minus_one", "one_i64", {1}, {});
    model_builder::AddNode(graph, "parent_sub", "Sub", {"rank", "one_i64"}, {"missing_dims"});
    AddInt64Constant(graph, "trip_count", "trip", {3}, {});
    AddBoolConstant(graph, "initial_condition", "initial_cond", true);

    auto* loop = model_builder::AddNode(graph, "parent_context_loop", "Loop", {"trip", "initial_cond", "flat"}, {"Y"});
    auto* body = AddGraphAttribute(loop, "body", "parent_context_loop_body");
    AddScalarValueInfo(body->mutable_input(), "iteration", onnx::TensorProto_DataType_INT64);
    AddScalarValueInfo(body->mutable_input(), "cond_in", onnx::TensorProto_DataType_BOOL);
    model_builder::AddValueInfo(body->mutable_input(), "value_in", onnx::TensorProto_DataType_FLOAT, {2, 3});
    AddScalarValueInfo(body->mutable_output(), "cond_out", onnx::TensorProto_DataType_BOOL);
    model_builder::AddValueInfo(body->mutable_output(), "value_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    AddFloatConstant(body, "increment", "one", {1.0f}, {});
    model_builder::AddNode(body, "add", "Add", {"value_in", "one"}, {"incremented"});
    AddInt64Constant(body, "constant_shape", "shape_one", {1}, {1});
    model_builder::AddNode(body, "abs_missing_dims", "Abs", {"missing_dims"}, {"abs_missing"});
    model_builder::AddNode(body, "reshape_missing_dims", "Reshape", {"abs_missing", "shape_one"}, {"missing_shape"});
    model_builder::AddNode(body, "expand_ones", "Expand", {"shape_one", "missing_shape"}, {"leading_ones"});
    auto* concat =
        model_builder::AddNode(body, "prepend_ones", "Concat", {"leading_ones", "flat_shape"}, {"padded_shape"});
    AddIntAttribute(concat, "axis", 0);
    model_builder::AddNode(body, "reshape_with_parent_shape", "Reshape", {"incremented", "padded_shape"},
                           {"padded_value"});
    AddInt64Constant(body, "squeeze_axes", "axis_zero", {0}, {1});
    model_builder::AddNode(body, "remove_leading_dimension", "Squeeze", {"padded_value", "axis_zero"}, {"value_out"});
    model_builder::AddNode(body, "condition_identity", "Identity", {"cond_in"}, {"cond_out"});
    model_builder::SaveModel(model, path);
}

inline void CreateScanModel(const std::string& path, bool include_excluded_sin = false)
{
    auto model = model_builder::CreateModelWithDefaultOpset("ScanGraph", 13);
    auto* graph = model.mutable_graph();
    AddScalarValueInfo(graph->mutable_input(), "initial_state", onnx::TensorProto_DataType_FLOAT);
    model_builder::AddValueInfo(graph->mutable_input(), "scan_input", onnx::TensorProto_DataType_FLOAT, {3});
    AddScalarValueInfo(graph->mutable_output(), "final_state", onnx::TensorProto_DataType_FLOAT);
    model_builder::AddValueInfo(graph->mutable_output(), "scan_output", onnx::TensorProto_DataType_FLOAT, {3});
    auto* scan =
        model_builder::AddNode(graph, "scan", "Scan", {"initial_state", "scan_input"}, {"final_state", "scan_output"});
    AddIntAttribute(scan, "num_scan_inputs", 1);
    auto* body = AddGraphAttribute(scan, "body", "scan_body");
    AddScalarValueInfo(body->mutable_input(), "state_in", onnx::TensorProto_DataType_FLOAT);
    AddScalarValueInfo(body->mutable_input(), "element_in", onnx::TensorProto_DataType_FLOAT);
    AddScalarValueInfo(body->mutable_output(), "state_out", onnx::TensorProto_DataType_FLOAT);
    AddScalarValueInfo(body->mutable_output(), "element_out", onnx::TensorProto_DataType_FLOAT);
    model_builder::AddNode(body, "accumulate", "Add", {"state_in", "element_in"}, {"accumulated"});
    if (include_excluded_sin)
        model_builder::AddNode(body, "excluded_scan_sin", "Sin", {"accumulated"}, {"state_out"});
    else
        model_builder::AddNode(body, "state_identity", "Identity", {"accumulated"}, {"state_out"});
    model_builder::AddNode(body, "emit", "Identity", {"state_out"}, {"element_out"});
    model_builder::SaveModel(model, path);
}

// The Scan body captures a reshape target produced in the parent graph. This
// exercises the same parent-context retry as the Loop case without relying on
// an If branch or its sibling-specific handling.
inline void CreateParentProducedShapeScanModel(const std::string& path)
{
    auto model = model_builder::CreateModelWithDefaultOpset("ParentProducedShapeScanGraph", 13);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "initial_state", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_input(), "scan_input", onnx::TensorProto_DataType_FLOAT, {3, 2, 3});
    model_builder::AddValueInfo(graph->mutable_output(), "final_state", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(graph->mutable_output(), "scan_output", onnx::TensorProto_DataType_FLOAT, {3, 2, 3});
    AddFloatConstant(graph, "parent_one", "parent_one_value", {1.0f}, {});
    model_builder::AddNode(graph, "parent_div", "Div", {"initial_state", "parent_one_value"}, {"normalized_state"});
    auto* flatten = model_builder::AddNode(graph, "parent_flatten", "Flatten", {"normalized_state"}, {"flat_state"});
    AddIntAttribute(flatten, "axis", 1);
    model_builder::AddNode(graph, "parent_shape", "Shape", {"flat_state"}, {"flat_shape"});
    model_builder::AddNode(graph, "parent_size", "Size", {"flat_shape"}, {"rank"});
    AddInt64Constant(graph, "rank_minus_one", "one_i64", {1}, {});
    model_builder::AddNode(graph, "parent_sub", "Sub", {"rank", "one_i64"}, {"missing_dims"});

    auto* scan = model_builder::AddNode(graph, "parent_context_scan", "Scan", {"flat_state", "scan_input"},
                                        {"final_state", "scan_output"});
    AddIntAttribute(scan, "num_scan_inputs", 1);
    auto* body = AddGraphAttribute(scan, "body", "parent_context_scan_body");
    model_builder::AddValueInfo(body->mutable_input(), "state_in", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(body->mutable_input(), "element_in", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(body->mutable_output(), "state_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddValueInfo(body->mutable_output(), "element_out", onnx::TensorProto_DataType_FLOAT, {2, 3});
    model_builder::AddNode(body, "accumulate", "Add", {"state_in", "element_in"}, {"accumulated"});
    AddInt64Constant(body, "constant_shape", "shape_one", {1}, {1});
    model_builder::AddNode(body, "abs_missing_dims", "Abs", {"missing_dims"}, {"abs_missing"});
    model_builder::AddNode(body, "reshape_missing_dims", "Reshape", {"abs_missing", "shape_one"}, {"missing_shape"});
    model_builder::AddNode(body, "expand_ones", "Expand", {"shape_one", "missing_shape"}, {"leading_ones"});
    auto* concat =
        model_builder::AddNode(body, "prepend_ones", "Concat", {"leading_ones", "flat_shape"}, {"padded_shape"});
    AddIntAttribute(concat, "axis", 0);
    model_builder::AddNode(body, "reshape_with_parent_shape", "Reshape", {"accumulated", "padded_shape"},
                           {"padded_state"});
    AddInt64Constant(body, "squeeze_axes", "axis_zero", {0}, {1});
    model_builder::AddNode(body, "remove_leading_dimension", "Squeeze", {"padded_state", "axis_zero"}, {"state_out"});
    model_builder::AddNode(body, "emit", "Identity", {"state_out"}, {"element_out"});
    model_builder::SaveModel(model, path);
}

}  // namespace control_flow_models
