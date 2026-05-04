// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "clip_bound_compatibility.h"
#include "pooling_dilation_compatibility.h"
#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include "trt_proto_preprocessing.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{

constexpr int32_t kFp32 = onnx::TensorProto_DataType_FLOAT;
constexpr int32_t kFp16 = onnx::TensorProto_DataType_FLOAT16;
constexpr int32_t kInt32 = onnx::TensorProto_DataType_INT32;
constexpr const char* kOriginalDocString = "42";

onnx::ModelProto MakeModel(int64_t opset_version = 13)
{
    onnx::ModelProto model;
    model.set_ir_version(7);

    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(opset_version);

    model.mutable_graph()->set_name("preprocessing_test_graph");
    return model;
}

void AddFloatScalarInitializer(onnx::GraphProto& graph, const std::string& name, int32_t data_type, float value)
{
    auto* tensor = graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(data_type);

    if (data_type == kFp16)
    {
        const uint16_t bits = model_builder::Float32ToFloat16(value);
        tensor->set_raw_data(&bits, sizeof(bits));
    }
    else
    {
        tensor->set_raw_data(&value, sizeof(value));
    }
}

void AddFloatVectorInitializer(onnx::GraphProto& graph, const std::string& name, int32_t data_type,
                               const std::vector<float>& values)
{
    auto* tensor = graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(data_type);
    tensor->add_dims(static_cast<int64_t>(values.size()));

    if (data_type == kFp16)
    {
        std::vector<uint16_t> bits;
        bits.reserve(values.size());
        for (const float value : values)
        {
            bits.push_back(model_builder::Float32ToFloat16(value));
        }
        tensor->set_raw_data(bits.data(), bits.size() * sizeof(uint16_t));
    }
    else
    {
        tensor->set_raw_data(values.data(), values.size() * sizeof(float));
    }
}

void AddConstantNode(onnx::GraphProto& graph, const std::string& name, const std::string& output_name,
                     int32_t data_type, float value)
{
    auto* tensor = new onnx::TensorProto();
    tensor->set_name(output_name);
    tensor->set_data_type(data_type);
    if (data_type == kFp16)
    {
        const uint16_t bits = model_builder::Float32ToFloat16(value);
        tensor->set_raw_data(&bits, sizeof(bits));
    }
    else
    {
        tensor->set_raw_data(&value, sizeof(value));
    }

    auto* node = model_builder::AddNode(&graph, name, "Constant", {}, {output_name});
    auto* attr = node->add_attribute();
    attr->set_name("value");
    attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    attr->set_allocated_t(tensor);
}

void AddIntsAttribute(onnx::NodeProto& node, const std::string& name, const std::vector<int64_t>& values)
{
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto_AttributeType_INTS);
    for (const auto value : values)
    {
        attr->add_ints(value);
    }
}

void AddFloatAttribute(onnx::NodeProto& node, const std::string& name, float value)
{
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
    attr->set_f(value);
}

void AddStringAttribute(onnx::NodeProto& node, const std::string& name, const std::string& value)
{
    auto* attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    attr->set_s(value);
}

onnx::ModelProto BuildClipModel(const std::vector<std::string>& clip_inputs, int32_t data_type = kFp32,
                                const std::vector<int>& shape = {1, 3})
{
    auto model = MakeModel();
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", data_type, shape);
    model_builder::AddValueInfo(graph->mutable_output(), "Y", data_type, shape);

    auto* node = model_builder::AddNode(graph, "clip", "Clip", clip_inputs, {"Y"});
    node->set_doc_string(kOriginalDocString);
    return model;
}

onnx::ModelProto BuildLegacyClipModel()
{
    auto model = MakeModel(/*opset_version=*/6);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", kFp32, {1, 3});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", kFp32, {1, 3});

    auto* node = model_builder::AddNode(graph, "legacy_clip", "Clip", {"X"}, {"Y"});
    AddFloatAttribute(*node, "min", -1.0f);
    AddFloatAttribute(*node, "max", 1.0f);
    node->set_doc_string(kOriginalDocString);
    return model;
}

onnx::ModelProto BuildPoolModel(const std::string& op_type, int32_t data_type, const std::vector<int>& input_shape,
                                const std::vector<int>& output_shape, const std::vector<int64_t>& kernel_shape,
                                const std::vector<int64_t>& dilations, const std::vector<int64_t>& strides = {},
                                const std::vector<int64_t>& pads = {}, const std::string& auto_pad = "",
                                bool add_second_output = false)
{
    auto model = MakeModel(/*opset_version=*/22);
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", data_type, input_shape);
    model_builder::AddValueInfo(graph->mutable_output(), "Y", data_type, output_shape);
    if (add_second_output)
    {
        model_builder::AddValueInfo(graph->mutable_output(), "Indices", onnx::TensorProto_DataType_INT64,
                                    output_shape);
    }

    std::vector<std::string> outputs = {"Y"};
    if (add_second_output)
    {
        outputs.push_back("Indices");
    }

    auto* node = model_builder::AddNode(graph, "pool", op_type, {"X"}, outputs);
    node->set_doc_string(kOriginalDocString);
    AddIntsAttribute(*node, "kernel_shape", kernel_shape);
    if (!dilations.empty())
    {
        AddIntsAttribute(*node, "dilations", dilations);
    }
    if (!strides.empty())
    {
        AddIntsAttribute(*node, "strides", strides);
    }
    if (!pads.empty())
    {
        AddIntsAttribute(*node, "pads", pads);
    }
    if (!auto_pad.empty())
    {
        AddStringAttribute(*node, "auto_pad", auto_pad);
    }
    return model;
}

size_t CountNodes(const onnx::GraphProto& graph, const std::string& op_type)
{
    size_t count = 0;
    for (const auto& node : graph.node())
    {
        if (node.op_type() == op_type)
        {
            ++count;
        }
    }
    return count;
}

const onnx::NodeProto* FindNodeByOutput(const onnx::GraphProto& graph, const std::string& output_name)
{
    for (const auto& node : graph.node())
    {
        if (std::find(node.output().begin(), node.output().end(), output_name) != node.output().end())
        {
            return &node;
        }
    }
    return nullptr;
}

const onnx::TensorProto* FindInitializer(const onnx::GraphProto& graph, const std::string& name)
{
    for (const auto& initializer : graph.initializer())
    {
        if (initializer.name() == name)
        {
            return &initializer;
        }
    }
    return nullptr;
}

std::vector<int64_t> ReadInt64Initializer(const onnx::GraphProto& graph, const std::string& name)
{
    const auto* tensor = FindInitializer(graph, name);
    if (tensor == nullptr)
    {
        ADD_FAILURE() << "Missing initializer: " << name;
        return {};
    }

    std::vector<int64_t> values;
    if (tensor->has_raw_data())
    {
        const auto bytes = tensor->raw_data().size();
        values.resize(bytes / sizeof(int64_t));
        std::memcpy(values.data(), tensor->raw_data().data(), bytes);
        return values;
    }

    values.assign(tensor->int64_data().begin(), tensor->int64_data().end());
    return values;
}

void ExpectOutputNode(const onnx::ModelProto& model, const std::string& output_name, const std::string& op_type,
                      const std::vector<std::string>& inputs)
{
    const auto* node = FindNodeByOutput(model.graph(), output_name);
    ASSERT_NE(node, nullptr) << "Could not find producer for output " << output_name;
    EXPECT_EQ(node->op_type(), op_type);
    ASSERT_EQ(node->input_size(), static_cast<int>(inputs.size()));
    for (int i = 0; i < node->input_size(); ++i)
    {
        EXPECT_EQ(node->input(i), inputs[static_cast<size_t>(i)]);
    }
    EXPECT_EQ(node->doc_string(), kOriginalDocString);
}

void ExpectNativeSingleNode(const onnx::ModelProto& model, const std::string& op_type)
{
    ASSERT_EQ(model.graph().node_size(), 1);
    EXPECT_EQ(model.graph().node(0).op_type(), op_type);
    EXPECT_EQ(model.graph().node(0).doc_string(), kOriginalDocString);
}

struct SliceSpatialParams
{
    int64_t start_h = 0;
    int64_t start_w = 0;
    int64_t end_h = 0;
    int64_t end_w = 0;
    int64_t step_h = 0;
    int64_t step_w = 0;

    bool operator<(const SliceSpatialParams& other) const
    {
        return std::tie(start_h, start_w, end_h, end_w, step_h, step_w) <
               std::tie(other.start_h, other.start_w, other.end_h, other.end_w, other.step_h, other.step_w);
    }
};

std::set<SliceSpatialParams> CollectSliceSpatialParams(const onnx::GraphProto& graph)
{
    std::set<SliceSpatialParams> params;
    for (const auto& node : graph.node())
    {
        if (node.op_type() != "Slice")
        {
            continue;
        }

        if (node.input_size() < 5)
        {
            ADD_FAILURE() << "Slice node has too few inputs: " << node.name();
            continue;
        }
        const auto starts = ReadInt64Initializer(graph, node.input(1));
        const auto ends = ReadInt64Initializer(graph, node.input(2));
        const auto steps = ReadInt64Initializer(graph, node.input(4));
        if (starts.size() < 4 || ends.size() < 4 || steps.size() < 4)
        {
            ADD_FAILURE() << "Slice node has rank < 4 parameters: " << node.name();
            continue;
        }

        params.insert({starts[2], starts[3], ends[2], ends[3], steps[2], steps[3]});
    }
    return params;
}

void ExpectPoolLowered(const onnx::ModelProto& model, bool is_average_pool, size_t tap_count)
{
    EXPECT_EQ(CountNodes(model.graph(), "Slice"), tap_count);
    if (is_average_pool)
    {
        EXPECT_EQ(CountNodes(model.graph(), "Add"), tap_count - 1);
        EXPECT_EQ(CountNodes(model.graph(), "Div"), 1u);
    }
    else
    {
        EXPECT_EQ(CountNodes(model.graph(), "Max"), tap_count - 1);
    }

    const auto* output_node = FindNodeByOutput(model.graph(), "Y");
    ASSERT_NE(output_node, nullptr);
    EXPECT_EQ(output_node->op_type(), is_average_pool ? "Div" : "Max");
    EXPECT_GE(output_node->input_size(), 2);
    EXPECT_EQ(output_node->doc_string(), kOriginalDocString);
}

void SaveModel(const onnx::ModelProto& model, const std::filesystem::path& path)
{
    model_builder::SaveModel(model, path.string());
}

onnx::ModelProto ReadModel(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open ONNX model: " + path.string());
    }

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&file))
    {
        throw std::runtime_error("Failed to parse ONNX model: " + path.string());
    }
    return model;
}

void AppendTrtRtxEp(Ort::SessionOptions& session_options)
{
    const auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, devices, ep_options);
}

void CompileModelWithTrtRtx(const onnx::ModelProto& model, const std::string& model_name)
{
    const auto input_path = std::filesystem::path(model_name + ".onnx");
    const auto output_path = std::filesystem::path(model_name + "_ctx.onnx");
    clearFileIfExists(input_path);
    clearFileIfExists(output_path);
    SaveModel(model, input_path);

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetEpContextEmbedMode(true);
    compile_options.SetInputModelPath(toOrtString(input_path).c_str());
    compile_options.SetOutputModelPath(toOrtString(output_path).c_str());

    const auto status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    const auto compiled_model = ReadModel(output_path);
    EXPECT_GE(CountNodes(compiled_model.graph(), "EPContext"), 1u)
        << "Expected the tiny compatibility model to be assigned to TRT-RTX.";
}

}  // namespace

// Intent:
// WebNN clamp with default options means neither side is bounded. Chromium can
// encode that as ONNX Clip(X, -inf, +inf), but TRT-RTX rejects non-finite Clip
// activation bounds.
//
//   Before: X -- Clip(min=-inf, max=+inf) --> Y
//   After:  X -- Identity -----------------> Y
//
// This test mirrors the required WebNN clamp default-option failures across the
// scalar/vector ranks and fp32/fp16 dtypes we saw in conformance.
TEST(TensorRTRTXProtoPreprocessingTest, ClipExplicitMinusInfPlusInf_RewritesToIdentity)
{
    const std::vector<std::vector<int>> shapes = {{}, {3}, {1, 3}, {1, 1, 3}, {1, 1, 1, 3}, {1, 1, 1, 1, 3}};
    for (const auto data_type : {kFp32, kFp16})
    {
        for (const auto& shape : shapes)
        {
            auto model = BuildClipModel({"X", "min", "max"}, data_type, shape);
            AddFloatScalarInitializer(*model.mutable_graph(), "min", data_type, -std::numeric_limits<float>::infinity());
            AddFloatScalarInitializer(*model.mutable_graph(), "max", data_type, std::numeric_limits<float>::infinity());

            trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

            ExpectOutputNode(model, "Y", "Identity", {"X"});
        }
    }
}

// Intent:
// WebNN clamp with only minValue is a lower-bound-only clamp. The +inf upper
// bound is just an encoding detail that TRT-RTX cannot parse as Clip.
//
//   Before: X -- Clip(min=finite, max=+inf) --> Y
//   After:  X -- Max(X, finite_min) ---------> Y
//
// The negative/zero/positive min cases line up with the conformance tests that
// failed before the compatibility pass.
TEST(TensorRTRTXProtoPreprocessingTest, ClipFiniteMinPlusInf_RewritesToMax)
{
    for (const auto data_type : {kFp32, kFp16})
    {
        for (const float min_value : {-2.0f, 0.0f, 1.0f})
        {
            auto model = BuildClipModel({"X", "min", "max"}, data_type);
            AddFloatScalarInitializer(*model.mutable_graph(), "min", data_type, min_value);
            AddFloatScalarInitializer(*model.mutable_graph(), "max", data_type, std::numeric_limits<float>::infinity());

            trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

            ExpectOutputNode(model, "Y", "Max", {"X", "min"});
        }
    }
}

// Intent:
// WebNN clamp with only maxValue is an upper-bound-only clamp. The -inf lower
// bound is not real math we need TRT to execute; it only tells WebNN there is
// no lower bound.
//
//   Before: X -- Clip(min=-inf, max=finite) --> Y
//   After:  X -- Min(X, finite_max) ---------> Y
//
// The negative/zero/positive max cases mirror the required conformance
// failures fixed by this pass.
TEST(TensorRTRTXProtoPreprocessingTest, ClipMinusInfFiniteMax_RewritesToMin)
{
    for (const auto data_type : {kFp32, kFp16})
    {
        for (const float max_value : {-2.0f, 0.0f, 3.0f})
        {
            auto model = BuildClipModel({"X", "min", "max"}, data_type);
            AddFloatScalarInitializer(*model.mutable_graph(), "min", data_type, -std::numeric_limits<float>::infinity());
            AddFloatScalarInitializer(*model.mutable_graph(), "max", data_type, max_value);

            trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

            ExpectOutputNode(model, "Y", "Min", {"X", "max"});
        }
    }
}

// Intent:
// Finite Clip bounds are not the TRT-RTX limitation we are working around.
// Keeping native Clip preserves the original graph shape and lets TRT parse the
// compact operator when both activation parameters are finite.
//
//   X -- Clip(min=-1, max=1) --> Y
//
// This guards against the compatibility pass becoming an unnecessary generic
// Clip decomposition pass.
TEST(TensorRTRTXProtoPreprocessingTest, ClipFiniteMinFiniteMax_RemainsClip)
{
    auto model = BuildClipModel({"X", "min", "max"});
    AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, -1.0f);
    AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, 1.0f);

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "Clip");
}

// Intent:
// ONNX explicitly defines Clip behavior when min > max: all inputs become max.
// WebNN validation normally rejects an invalid range before graph execution, so
// this is an ONNX-only semantic that our WebNN compatibility pass must not
// accidentally change.
//
//   X -- Clip(min=3, max=1) --> Y
//
// Leaving this as Clip keeps the ONNX edge case intact.
TEST(TensorRTRTXProtoPreprocessingTest, ClipFiniteMinGreaterThanFiniteMax_RemainsClip)
{
    auto model = BuildClipModel({"X", "min", "max"});
    AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, 3.0f);
    AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, 1.0f);

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "Clip");
}

// Intent:
// Modern ONNX Clip allows optional min/max inputs to be omitted, but omitted
// ONNX bounds default to finite numeric_limits values, not true infinities.
//
//   X -- Clip(missing min, missing max) --> Y
//
// If this became Identity, then X=[+inf] would incorrectly remain +inf instead
// of being clamped to the finite dtype max. This test protects the P2 review
// fix.
TEST(TensorRTRTXProtoPreprocessingTest, ClipMissingBothBounds_RemainsClip)
{
    auto model = BuildClipModel({"X"});

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "Clip");
}

// Intent:
// ONNX missing min is not equivalent to an explicit WebNN -inf lower bound.
//
//   X -- Clip(min omitted, max=finite) --> Y
//
// The pass must not rewrite this to Min unless it has proven that the lower
// bound is an explicit -inf scalar.
TEST(TensorRTRTXProtoPreprocessingTest, ClipMissingMinFiniteMax_RemainsClip)
{
    auto model = BuildClipModel({"X", "", "max"});
    AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, 1.0f);

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "Clip");
}

// Intent:
// ONNX missing max is not equivalent to an explicit WebNN +inf upper bound.
//
//   X -- Clip(min=finite, max omitted) --> Y
//
// The pass must not rewrite this to Max unless the upper bound is an explicit
// +inf scalar.
TEST(TensorRTRTXProtoPreprocessingTest, ClipFiniteMinMissingMax_RemainsClip)
{
    auto model = BuildClipModel({"X", "min"});
    AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, -1.0f);

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "Clip");
}

// Intent:
// Older ONNX Clip opsets encoded min/max as attributes and had only the data
// input. Treating that one-input Clip as "both bounds missing" would silently
// turn a finite clamp into Identity.
//
//   X -- Clip[min=-1,max=1] --> Y
//
// This protects the P1 legacy-Clip review finding.
TEST(TensorRTRTXProtoPreprocessingTest, LegacyAttributeClip_RemainsClip)
{
    auto model = BuildLegacyClipModel();

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "Clip");
}

// Intent:
// WebNN-generated bounds may appear as Constant nodes instead of initializers.
// The pass intentionally recognizes scalar Constant producers, but does not
// chase arbitrary arithmetic.
//
//   min_const -->.
//                Clip --> Y       becomes       Max(X, min_const) --> Y
//   max_const -->'
//
// This keeps the compatibility pass useful across common ONNX export styles.
TEST(TensorRTRTXProtoPreprocessingTest, ClipConstantNodeInfBounds_Rewrites)
{
    auto model = BuildClipModel({"X", "min", "max"});
    AddConstantNode(*model.mutable_graph(), "min_const", "min", kFp32, -1.0f);
    AddConstantNode(*model.mutable_graph(), "max_const", "max", kFp32, std::numeric_limits<float>::infinity());

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    ExpectOutputNode(model, "Y", "Max", {"X", "min"});
}

// Intent:
// Dynamic or vector bounds may carry runtime/user semantics. Rewriting them
// would turn this parser-compatibility pass into partial constant propagation.
//
//   X, runtime_min, max=+inf -- Clip --> Y
//   X, vector_min,  max=+inf -- Clip --> Y
//
// Both cases stay native so only provably scalar compile-time bounds are
// rewritten.
TEST(TensorRTRTXProtoPreprocessingTest, ClipDynamicOrVectorBounds_RemainsClip)
{
    {
        auto model = BuildClipModel({"X", "runtime_min", "max"});
        model_builder::AddValueInfo(model.mutable_graph()->mutable_input(), "runtime_min", kFp32, {});
        AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, std::numeric_limits<float>::infinity());

        trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

        ExpectNativeSingleNode(model, "Clip");
    }
    {
        auto model = BuildClipModel({"X", "vector_min", "max"});
        AddFloatVectorInitializer(*model.mutable_graph(), "vector_min", kFp32, {-1.0f, -2.0f});
        AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, std::numeric_limits<float>::infinity());

        trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

        ExpectNativeSingleNode(model, "Clip");
    }
}

// Intent:
// The pass removes only WebNN's explicit unbounded sides. It must not rewrite
// cases that intentionally produce infinities or NaNs.
//
//   min=NaN, max=+inf
//   min=+inf, max=+inf
//   min=-inf, max=-inf
//
// These stay as Clip because replacing them with Identity/Min/Max would change
// observable non-finite behavior.
TEST(TensorRTRTXProtoPreprocessingTest, ClipNaNOrWrongInfinityDirection_RemainsClip)
{
    const std::vector<std::pair<float, float>> cases = {
        {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
    };

    for (const auto& [min_value, max_value] : cases)
    {
        auto model = BuildClipModel({"X", "min", "max"});
        AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, min_value);
        AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, max_value);

        trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

        ExpectNativeSingleNode(model, "Clip");
    }
}

// Intent:
// The provider's proto-node bookkeeping uses stable logical outputs and ORT
// doc_string node ids. A rewrite may change the operator, but it must not
// change the original output edge or lose the doc string.
//
//   X -- Clip(min=-inf,max=+inf) -- "Y"
//   X -- Identity -------------- -- "Y"
//
// This mirrors the mapping policy used by the Q/DQ lowering work.
TEST(TensorRTRTXProtoPreprocessingTest, ClipRewritePreservesOutputNameAndDocString)
{
    auto model = BuildClipModel({"X", "min", "max"});
    AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, -std::numeric_limits<float>::infinity());
    AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, std::numeric_limits<float>::infinity());

    trt_rtx_ep::RunClipBoundCompatibilityForTensorRt(model);

    const auto* node = FindNodeByOutput(model.graph(), "Y");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type(), "Identity");
    EXPECT_EQ(node->output_size(), 1);
    EXPECT_EQ(node->output(0), "Y");
    EXPECT_EQ(node->doc_string(), kOriginalDocString);
}

// Intent:
// WebNN averagePool2d with windowDimensions=[3,3] and dilations=[2,2] samples
// nine real input positions spread across an effective 5x5 region:
//
//   x . x . x
//   . . . . .
//   x . x . x
//   . . . . .
//   x . x . x
//
// TRT-RTX rejects native pooling dilation, so the pass lowers each tap to a
// Slice, adds all tap tensors, and divides by the tap count.
TEST(TensorRTRTXProtoPreprocessingTest, AveragePoolDilated3x3Stride1_Fp32Fp16_Rewrites)
{
    for (const auto data_type : {kFp32, kFp16})
    {
        auto model = BuildPoolModel("AveragePool", data_type, {1, 2, 5, 5}, {1, 2, 1, 1}, {3, 3}, {2, 2});

        trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

        ExpectPoolLowered(model, /*is_average_pool=*/true, /*tap_count=*/9);
        const auto params = CollectSliceSpatialParams(model.graph());
        EXPECT_TRUE(params.count({0, 0, 1, 1, 1, 1}));
        EXPECT_TRUE(params.count({4, 4, 5, 5, 1, 1}));
    }
}

// Intent:
// MaxPool has the same dilated tap selection as AveragePool, but combines taps
// with elementwise Max instead of Add+Div.
//
//   tap0, tap1, ... tap8 -- Max chain --> Y
//
// This mirrors the required maxPool2d fp32/fp16 conformance failures.
TEST(TensorRTRTXProtoPreprocessingTest, MaxPoolDilated3x3Stride1_Fp32Fp16_Rewrites)
{
    for (const auto data_type : {kFp32, kFp16})
    {
        auto model = BuildPoolModel("MaxPool", data_type, {1, 2, 5, 5}, {1, 2, 1, 1}, {3, 3}, {2, 2});

        trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

        ExpectPoolLowered(model, /*is_average_pool=*/false, /*tap_count=*/9);
        const auto params = CollectSliceSpatialParams(model.graph());
        EXPECT_TRUE(params.count({0, 0, 1, 1, 1, 1}));
        EXPECT_TRUE(params.count({4, 4, 5, 5, 1, 1}));
    }
}

// Intent:
// Stride and dilation interact in the lowered Slice parameters. For output
// coordinate o and tap coordinate k:
//
//   input_index = o * stride + k * dilation
//
// Therefore Slice uses start=k*dilation and step=stride. This catches the most
// likely math bug if stride support regresses.
TEST(TensorRTRTXProtoPreprocessingTest, PoolDilated3x3Stride2_RewritesSliceStepsCorrectly)
{
    auto model = BuildPoolModel("AveragePool", kFp32, {1, 1, 7, 7}, {1, 1, 2, 2}, {3, 3}, {2, 2}, {2, 2});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectPoolLowered(model, /*is_average_pool=*/true, /*tap_count=*/9);
    const auto params = CollectSliceSpatialParams(model.graph());
    EXPECT_TRUE(params.count({0, 0, 3, 3, 2, 2}));
    EXPECT_TRUE(params.count({4, 4, 7, 7, 2, 2}));
}

// Intent:
// Dilation is per spatial axis. A kernel=[2,3], dilations=[1,2] case makes
// height and width offsets intentionally different:
//
//   x . x . x
//   x . x . x
//
// This prevents an implementation that accidentally applies one dilation value
// to every axis.
TEST(TensorRTRTXProtoPreprocessingTest, PoolAsymmetricDilation_RewritesCorrectOffsets)
{
    auto model = BuildPoolModel("MaxPool", kFp32, {1, 1, 2, 5}, {1, 1, 1, 1}, {2, 3}, {1, 2});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectPoolLowered(model, /*is_average_pool=*/false, /*tap_count=*/6);
    const auto params = CollectSliceSpatialParams(model.graph());
    EXPECT_TRUE(params.count({0, 0, 1, 1, 1, 1}));
    EXPECT_TRUE(params.count({1, 4, 2, 5, 1, 1}));
}

// Intent:
// Unit dilation is already the native pooling case. Rewriting it would only
// grow the graph and could interfere with TRT's normal pooling parser path.
//
//   X -- AveragePool(dilations=[1,1]) --> Y
//
// The compatibility pass should leave this untouched.
TEST(TensorRTRTXProtoPreprocessingTest, PoolUnitDilation_RemainsNative)
{
    auto model = BuildPoolModel("AveragePool", kFp32, {1, 1, 5, 5}, {1, 1, 3, 3}, {3, 3}, {1, 1});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "AveragePool");
}

// Intent:
// Padding changes which elements participate in edge windows. AveragePool also
// needs count_include_pad-aware divisors. The current lowering deliberately
// supports only the zero-padding/full-window subset.
//
//   X -- AveragePool(pads=[1,0,0,1], dilations=[2,2]) --> Y
//
// This must stay native until explicit mask/divisor support exists.
TEST(TensorRTRTXProtoPreprocessingTest, PoolNonZeroPadding_RemainsNative)
{
    auto model = BuildPoolModel("AveragePool", kFp32, {1, 1, 5, 5}, {1, 1, 2, 2}, {3, 3}, {2, 2}, {}, {1, 0, 0, 1});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "AveragePool");
}

// Intent:
// auto_pad modes synthesize padding and output sizing rules outside the simple
// zero-pad equivalence we can prove with static Slice nodes.
//
//   X -- MaxPool(auto_pad=SAME_UPPER, dilations=[2,2]) --> Y
//
// The pass should not approximate this with the full-window lowering.
TEST(TensorRTRTXProtoPreprocessingTest, PoolAutoPadSameOrValid_RemainsNative)
{
    for (const std::string auto_pad : {"SAME_UPPER", "VALID"})
    {
        auto model = BuildPoolModel("MaxPool", kFp32, {1, 1, 5, 5}, {1, 1, 1, 1}, {3, 3}, {2, 2}, {}, {}, auto_pad);

        trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

        ExpectNativeSingleNode(model, "MaxPool");
    }
}

// Intent:
// The lowering emits Slice nodes that read real input elements for every output
// coordinate. If output metadata implies a partial final window, native pooling
// needs padding/masking semantics that are outside this compatibility pass.
//
//   input 5x5, kernel 3x3, dilation 2, output claims 2x2
//
// The second output coordinate would read past the real input, so no rewrite.
TEST(TensorRTRTXProtoPreprocessingTest, PoolCeilOrPartialWindow_RemainsNative)
{
    auto model = BuildPoolModel("AveragePool", kFp32, {1, 1, 5, 5}, {1, 1, 2, 2}, {3, 3}, {2, 2});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "AveragePool");
}

// Intent:
// Static Slice starts/ends/steps cannot represent an unknown spatial dimension
// without adding runtime Shape/Gather/arithmetic nodes. Dynamic-shape pooling
// should therefore remain native for now.
//
//   X[N,C,H?,W] -- AveragePool(dilations=[2,2]) --> Y
//
// This protects against unsafe static assumptions.
TEST(TensorRTRTXProtoPreprocessingTest, PoolDynamicShape_RemainsNative)
{
    auto model = BuildPoolModel("AveragePool", kFp32, {1, 1, -1, 5}, {1, 1, 1, 1}, {3, 3}, {2, 2});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "AveragePool");
}

// Intent:
// The lowering currently supports fp32/fp16 because AveragePool emits a scalar
// floating divisor and WebNN's failing required cases were float models.
//
//   int32 X -- AveragePool(dilations=[2,2]) --> Y
//
// Unsupported element types stay native rather than risking integer division
// or type-promotion surprises.
TEST(TensorRTRTXProtoPreprocessingTest, PoolUnsupportedType_RemainsNative)
{
    auto model = BuildPoolModel("AveragePool", kInt32, {1, 1, 5, 5}, {1, 1, 1, 1}, {3, 3}, {2, 2});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "AveragePool");
}

// Intent:
// ONNX MaxPool can optionally produce indices as a second output. Our primitive
// Max-chain lowering reproduces the pooled values only, not argmax indices.
//
//   X -- MaxPool(values, indices) --> Y, Indices
//
// This must stay native until an indices-compatible lowering is implemented.
TEST(TensorRTRTXProtoPreprocessingTest, MaxPoolWithIndicesOutput_RemainsNative)
{
    auto model = BuildPoolModel("MaxPool", kFp32, {1, 1, 5, 5}, {1, 1, 1, 1}, {3, 3}, {2, 2}, {}, {},
                                /*auto_pad=*/"", /*add_second_output=*/true);

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    ExpectNativeSingleNode(model, "MaxPool");
}

// Intent:
// Like the Clip rewrite, pooling dilation lowering must preserve the externally
// visible output name and ORT doc_string node id. Helper nodes may be inserted,
// but downstream edges should still consume the original "Y" value.
//
//   X -- AveragePool(dilated) --> Y
//   X -- Slice/Add/Div -------> Y
//
// This keeps capability/build graph mapping stable.
TEST(TensorRTRTXProtoPreprocessingTest, PoolRewritePreservesOutputNameAndDocString)
{
    auto model = BuildPoolModel("AveragePool", kFp32, {1, 2, 5, 5}, {1, 2, 1, 1}, {3, 3}, {2, 2});

    trt_rtx_ep::RunPoolingDilationCompatibilityForTensorRt(model);

    const auto* node = FindNodeByOutput(model.graph(), "Y");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_type(), "Div");
    EXPECT_EQ(node->output_size(), 1);
    EXPECT_EQ(node->output(0), "Y");
    EXPECT_EQ(node->doc_string(), kOriginalDocString);
}

// Intent:
// Direct pass tests can succeed even if the production wrapper stops calling a
// pass. The EP uses RunTensorRtProtoPreprocessing before capability discovery
// and engine build, so this wrapper-level test checks the real integration
// point.
//
//   Clip(-inf,+inf) + AveragePool(dilated)
//          |
//          v
//   Identity + Slice/Add/Div after wrapper preprocessing
TEST(TensorRTRTXProtoPreprocessingTest, RunTensorRtProtoPreprocessing_AppliesClipAndPoolingPasses)
{
    auto model = MakeModel();
    auto* graph = model.mutable_graph();
    model_builder::AddValueInfo(graph->mutable_input(), "X", kFp32, {1, 1, 5, 5});
    model_builder::AddValueInfo(graph->mutable_output(), "Y", kFp32, {1, 1, 1, 1});
    model_builder::AddValueInfo(graph->mutable_value_info(), "clip_out", kFp32, {1, 1, 5, 5});
    AddFloatScalarInitializer(*graph, "min", kFp32, -std::numeric_limits<float>::infinity());
    AddFloatScalarInitializer(*graph, "max", kFp32, std::numeric_limits<float>::infinity());

    auto* clip = model_builder::AddNode(graph, "clip", "Clip", {"X", "min", "max"}, {"clip_out"});
    clip->set_doc_string("clip_doc");
    auto* pool = model_builder::AddNode(graph, "pool", "AveragePool", {"clip_out"}, {"Y"});
    pool->set_doc_string("pool_doc");
    AddIntsAttribute(*pool, "kernel_shape", {3, 3});
    AddIntsAttribute(*pool, "dilations", {2, 2});

    trt_rtx_ep::RunTensorRtProtoPreprocessing(model);

    const auto* clip_out = FindNodeByOutput(model.graph(), "clip_out");
    ASSERT_NE(clip_out, nullptr);
    EXPECT_EQ(clip_out->op_type(), "Identity");
    EXPECT_EQ(CountNodes(model.graph(), "Slice"), 9u);
    EXPECT_EQ(CountNodes(model.graph(), "Div"), 1u);
}

// Intent:
// This is a tiny runtime smoke test for the original WebNN clamp/TRT issue. It
// does not replace the proto policy tests above; it proves that the EP path
// handed to TRT can compile models whose source graph still contains the
// parser-hostile WebNN Clip encodings.
//
//   Clip(finite,+inf) and Clip(-inf,finite)
//        wrapper preprocessing
//   Max / Min
//
// The compiled model should contain an EPContext, which means TRT-RTX actually
// accepted a partition after preprocessing.
TEST(TensorRTRTXProtoPreprocessingTest, TrtCompile_WebNNClampMinOnlyAndMaxOnly_Succeeds)
{
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty()) << "No TRT RTX EP devices found.";

    {
        auto model = BuildClipModel({"X", "min", "max"});
        AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, -1.0f);
        AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, std::numeric_limits<float>::infinity());
        CompileModelWithTrtRtx(model, "trt_compile_webnn_clamp_min_only");
    }
    {
        auto model = BuildClipModel({"X", "min", "max"});
        AddFloatScalarInitializer(*model.mutable_graph(), "min", kFp32, -std::numeric_limits<float>::infinity());
        AddFloatScalarInitializer(*model.mutable_graph(), "max", kFp32, 1.0f);
        CompileModelWithTrtRtx(model, "trt_compile_webnn_clamp_max_only");
    }
}

// Intent:
// This smoke test exercises the exact required WebNN pooling dilation shape
// that failed before the lowering:
//
//   input=[1,2,5,5], window=[3,3], dilations=[2,2]
//
// The source model still contains native AveragePool/MaxPool dilation. Success
// plus EPContext in the compiled output confirms that production preprocessing
// lowered the graph before TRT-RTX parsed it.
TEST(TensorRTRTXProtoPreprocessingTest, TrtCompile_WebNNDilatedAveragePoolAndMaxPool_Succeeds)
{
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty()) << "No TRT RTX EP devices found.";

    CompileModelWithTrtRtx(
        BuildPoolModel("AveragePool", kFp32, {1, 2, 5, 5}, {1, 2, 1, 1}, {3, 3}, {2, 2}),
        "trt_compile_webnn_average_pool_dilation");
    CompileModelWithTrtRtx(BuildPoolModel("MaxPool", kFp32, {1, 2, 5, 5}, {1, 2, 1, 1}, {3, 3}, {2, 2}),
                           "trt_compile_webnn_max_pool_dilation");
}
