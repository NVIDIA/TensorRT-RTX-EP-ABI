// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pooling_dilation_compatibility.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_rtx_ep
{
namespace
{

// Implementation overview:
//
// ONNX/WebNN dilated pooling does not change the number of kernel taps; it
// changes the spacing between taps. For a 3x3 kernel with dilations=[2,2], the
// operator samples nine values spread across an effective 5x5 input region.
// TRT-RTX rejects AveragePool/MaxPool when any dilation is not 1, even though
// the same math can be expressed with primitive tensor operations.
//
// This pass decomposes a supported dilated pool into:
//   1. One Slice per kernel tap. Each Slice gathers that tap for every output
//      location using starts = tap * dilation and steps = original strides.
//   2. AveragePool: add all tap tensors and divide by the number of taps.
//   3. MaxPool: chain elementwise Max across all tap tensors.
//
// Correctness policy: lower only when every window is fully inside the input.
// That keeps AveragePool's divisor equal to kernel tap count and avoids padding
// masks. Cases involving padding, auto_pad, partial ceil/output windows, or
// dynamic shapes are left native until the pass grows explicit mask/divisor
// support.
constexpr const char* kMainOnnxDomain = "";
constexpr const char* kAiOnnxDomain = "ai.onnx";

// Guardrail for graph growth. Dilated pooling lowers to one Slice per kernel
// tap, so very large kernels should not explode the proto in a compatibility
// pass. The WebNN conformance cases that motivated this are 3x3.
constexpr size_t kMaxLoweredTaps = 256;

struct TensorMetadata
{
    std::optional<int32_t> element_type;
    std::vector<int64_t> shape;
};

struct GraphIndex
{
    explicit GraphIndex(const onnx::GraphProto& graph)
    {
        for (const auto& initializer : graph.initializer())
        {
            auto& metadata = tensor_metadata[initializer.name()];
            metadata.element_type = initializer.data_type();
            metadata.shape.assign(initializer.dims().begin(), initializer.dims().end());
        }

        for (const auto& value_info : graph.input())
        {
            AddValueInfo(value_info);
        }
        for (const auto& value_info : graph.value_info())
        {
            AddValueInfo(value_info);
        }
        for (const auto& value_info : graph.output())
        {
            AddValueInfo(value_info);
        }
    }

    std::optional<int32_t> FindTensorElementType(const std::string& name) const
    {
        const auto it = tensor_metadata.find(name);
        return it == tensor_metadata.end() ? std::nullopt : it->second.element_type;
    }

    bool TryGetStaticShape(const std::string& name, std::vector<int64_t>& shape) const
    {
        shape.clear();
        const auto it = tensor_metadata.find(name);
        if (it == tensor_metadata.end() || it->second.shape.empty())
        {
            return false;
        }

        for (const auto dim : it->second.shape)
        {
            if (dim <= 0)
            {
                shape.clear();
                return false;
            }
        }

        shape = it->second.shape;
        return true;
    }

    std::unordered_map<std::string, TensorMetadata> tensor_metadata;

private:
    void AddValueInfo(const onnx::ValueInfoProto& value_info)
    {
        if (value_info.name().empty() || !value_info.has_type() || !value_info.type().has_tensor_type())
        {
            return;
        }

        const auto& tensor_type = value_info.type().tensor_type();
        auto& metadata = tensor_metadata[value_info.name()];
        metadata.element_type = tensor_type.elem_type();
        metadata.shape.clear();
        if (!tensor_type.has_shape())
        {
            return;
        }

        metadata.shape.reserve(static_cast<size_t>(tensor_type.shape().dim_size()));
        for (const auto& dim : tensor_type.shape().dim())
        {
            metadata.shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
        }
    }
};

std::optional<int64_t> FindIntAttribute(const onnx::NodeProto& node, const std::string& name)
{
    for (const auto& attr : node.attribute())
    {
        if (attr.name() == name && attr.type() == onnx::AttributeProto_AttributeType_INT)
        {
            return attr.i();
        }
    }
    return std::nullopt;
}

std::optional<std::string> FindStringAttribute(const onnx::NodeProto& node, const std::string& name)
{
    for (const auto& attr : node.attribute())
    {
        if (attr.name() == name && attr.type() == onnx::AttributeProto_AttributeType_STRING)
        {
            return attr.s();
        }
    }
    return std::nullopt;
}

bool TryFindIntVectorAttribute(const onnx::NodeProto& node, const std::string& name, std::vector<int64_t>& values)
{
    for (const auto& attr : node.attribute())
    {
        if (attr.name() == name && attr.type() == onnx::AttributeProto_AttributeType_INTS)
        {
            values.assign(attr.ints().begin(), attr.ints().end());
            return true;
        }
    }
    values.clear();
    return false;
}

bool HasMainOnnxDomain(const onnx::NodeProto& node)
{
    return node.domain().empty() || node.domain() == kMainOnnxDomain || node.domain() == kAiOnnxDomain;
}

bool IsSupportedPoolOp(const onnx::NodeProto& node)
{
    return HasMainOnnxDomain(node) && (node.op_type() == "AveragePool" || node.op_type() == "MaxPool");
}

bool IsSupportedElementType(int32_t element_type)
{
    return element_type == onnx::TensorProto_DataType_FLOAT || element_type == onnx::TensorProto_DataType_FLOAT16;
}

std::vector<int64_t> DefaultVector(size_t size, int64_t value)
{
    return std::vector<int64_t>(size, value);
}

bool HasNonZeroPadding(const std::vector<int64_t>& pads)
{
    return std::any_of(pads.begin(), pads.end(),
                       [](int64_t value)
                       {
                           return value != 0;
                       });
}

bool HasNonUnitDilation(const std::vector<int64_t>& dilations)
{
    return std::any_of(dilations.begin(), dilations.end(),
                       [](int64_t value)
                       {
                           return value != 1;
                       });
}

bool AllPositive(const std::vector<int64_t>& values)
{
    return std::all_of(values.begin(), values.end(),
                       [](int64_t value)
                       {
                           return value > 0;
                       });
}

bool CheckedMultiply(size_t lhs, int64_t rhs, size_t& result)
{
    if (rhs <= 0)
    {
        return false;
    }

    const auto rhs_size = static_cast<size_t>(rhs);
    if (lhs > std::numeric_limits<size_t>::max() / rhs_size)
    {
        return false;
    }

    result = lhs * rhs_size;
    return true;
}

std::string MakeLoweredName(const std::string& base, const char* suffix, size_t unique_id)
{
    std::ostringstream oss;
    oss << "__trt_rtx_pool_dilation_lowered_" << unique_id << "_" << base << "_" << suffix;
    return oss.str();
}

void AddInt64VectorInitializer(onnx::GraphProto& graph, const std::string& name, const std::vector<int64_t>& values)
{
    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    tensor_proto->add_dims(static_cast<int64_t>(values.size()));
    tensor_proto->set_raw_data(values.data(), values.size() * sizeof(int64_t));
}

uint16_t Float32ToFloat16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;

    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return static_cast<uint16_t>(sign);
        }

        mantissa |= 0x800000;
        const uint32_t shifted = mantissa >> static_cast<uint32_t>(1 - exponent);
        return static_cast<uint16_t>(sign | ((shifted + 0x1000) >> 13));
    }

    if (exponent >= 31)
    {
        return static_cast<uint16_t>(sign | 0x7c00);
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000) >> 13));
}

void AddFloatScalarInitializer(onnx::GraphProto& graph, const std::string& name, int32_t element_type, float value)
{
    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(element_type);
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);

    if (element_type == onnx::TensorProto_DataType_FLOAT16)
    {
        const uint16_t half_bits = Float32ToFloat16Bits(value);
        tensor_proto->set_raw_data(&half_bits, sizeof(half_bits));
    }
    else
    {
        tensor_proto->set_raw_data(&value, sizeof(value));
    }
}

onnx::NodeProto* AppendNode(google::protobuf::RepeatedPtrField<onnx::NodeProto>& nodes,
                            const onnx::NodeProto& source_node, const std::string& op_type, const std::string& name,
                            const std::vector<std::string>& inputs, const std::vector<std::string>& outputs)
{
    auto* new_node = nodes.Add();
    new_node->set_name(name);
    new_node->set_domain(kMainOnnxDomain);
    new_node->set_op_type(op_type);
    // Preserve the original ORT node id doc_string so capability bookkeeping
    // can still attribute the lowered helper nodes to the source pooling node.
    new_node->set_doc_string(source_node.doc_string());
    for (const auto& input : inputs)
    {
        new_node->add_input(input);
    }
    for (const auto& output : outputs)
    {
        new_node->add_output(output);
    }
    return new_node;
}

struct PoolLoweringPlan
{
    bool is_average_pool = false;
    int32_t element_type = 0;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    size_t tap_count = 0;
};

bool TryBuildPoolLoweringPlan(const onnx::NodeProto& node, const GraphIndex& index, PoolLoweringPlan& plan)
{
    plan = {};
    if (!IsSupportedPoolOp(node) || node.input_size() != 1 || node.input(0).empty() || node.output_size() != 1 ||
        node.output(0).empty())
    {
        return false;
    }

    const auto auto_pad = FindStringAttribute(node, "auto_pad");
    if (auto_pad.has_value() && *auto_pad != "NOTSET")
    {
        // SAME_* auto padding changes edge-window membership. Leave it native
        // until the lowering grows explicit padding/mask support.
        return false;
    }

    if (!TryFindIntVectorAttribute(node, "kernel_shape", plan.kernel_shape) || plan.kernel_shape.empty() ||
        !AllPositive(plan.kernel_shape))
    {
        // ONNX pooling stores the spatial window in kernel_shape. We need it
        // to enumerate the tap coordinates that will become Slice nodes.
        return false;
    }

    const size_t spatial_rank = plan.kernel_shape.size();
    if (spatial_rank > static_cast<size_t>(std::numeric_limits<int64_t>::max() - 2))
    {
        return false;
    }

    if (!TryFindIntVectorAttribute(node, "strides", plan.strides))
    {
        plan.strides = DefaultVector(spatial_rank, 1);
    }
    if (!TryFindIntVectorAttribute(node, "dilations", plan.dilations))
    {
        plan.dilations = DefaultVector(spatial_rank, 1);
    }

    std::vector<int64_t> pads;
    if (!TryFindIntVectorAttribute(node, "pads", pads))
    {
        pads = DefaultVector(spatial_rank * 2, 0);
    }

    if (plan.strides.size() != spatial_rank || plan.dilations.size() != spatial_rank ||
        pads.size() != spatial_rank * 2 || !AllPositive(plan.strides) || !AllPositive(plan.dilations) ||
        !HasNonUnitDilation(plan.dilations) || HasNonZeroPadding(pads))
    {
        // Policy: support only the equivalence we can prove cheaply:
        // no explicit padding and at least one non-unit dilation. Padding
        // would require per-output validity masks and, for AveragePool,
        // count_include_pad-aware divisors.
        return false;
    }

    if (!index.TryGetStaticShape(node.input(0), plan.input_shape) || plan.input_shape.size() != spatial_rank + 2 ||
        !index.TryGetStaticShape(node.output(0), plan.output_shape) ||
        plan.output_shape.size() != plan.input_shape.size())
    {
        // Static input/output shapes are required because Slice starts/ends are
        // emitted as initializers. Dynamic-shape lowering would need Shape,
        // Gather, arithmetic, and runtime-computed Slice parameters.
        return false;
    }

    if (plan.output_shape[0] != plan.input_shape[0] || plan.output_shape[1] != plan.input_shape[1])
    {
        // ONNX AveragePool/MaxPool preserve N and C. If metadata disagrees,
        // do not manufacture helper nodes around an inconsistent graph.
        return false;
    }

    const auto element_type = index.FindTensorElementType(node.input(0));
    if (!element_type.has_value() || !IsSupportedElementType(*element_type))
    {
        // WebNN exposes these failing cases as fp32/fp16. Keeping the pass to
        // floating types avoids integer average/division surprises and keeps
        // the scalar divisor representable in the same tensor type.
        return false;
    }

    plan.is_average_pool = node.op_type() == "AveragePool";
    plan.element_type = *element_type;
    plan.tap_count = 1;
    for (const auto kernel_dim : plan.kernel_shape)
    {
        if (!CheckedMultiply(plan.tap_count, kernel_dim, plan.tap_count) || plan.tap_count > kMaxLoweredTaps)
        {
            return false;
        }
    }

    for (size_t dim = 0; dim < spatial_rank; ++dim)
    {
        const auto output_dim = plan.output_shape[dim + 2];
        if (output_dim <= 0)
        {
            return false;
        }

        const auto last_input_index =
            (output_dim - 1) * plan.strides[dim] + (plan.kernel_shape[dim] - 1) * plan.dilations[dim];
        if (last_input_index >= plan.input_shape[dim + 2])
        {
            // Every lowered Slice tap must read real input elements for every
            // output coordinate. If ceil_mode/output sizing creates a partial
            // final window, native pooling semantics require masking instead.
            return false;
        }
    }

    return true;
}

std::vector<int64_t> DecodeTapCoordinates(size_t tap_index, const std::vector<int64_t>& kernel_shape)
{
    std::vector<int64_t> coordinates(kernel_shape.size(), 0);
    for (size_t dim = kernel_shape.size(); dim > 0; --dim)
    {
        const auto kernel_dim = static_cast<size_t>(kernel_shape[dim - 1]);
        coordinates[dim - 1] = static_cast<int64_t>(tap_index % kernel_dim);
        tap_index /= kernel_dim;
    }
    return coordinates;
}

void BuildSliceParameters(const PoolLoweringPlan& plan, const std::vector<int64_t>& tap_coordinates,
                          std::vector<int64_t>& starts, std::vector<int64_t>& ends, std::vector<int64_t>& axes,
                          std::vector<int64_t>& steps)
{
    const size_t rank = plan.input_shape.size();
    starts.assign(rank, 0);
    ends = plan.input_shape;
    axes.resize(rank);
    steps.assign(rank, 1);

    for (size_t dim = 0; dim < rank; ++dim)
    {
        axes[dim] = static_cast<int64_t>(dim);
    }

    for (size_t dim = 0; dim < plan.kernel_shape.size(); ++dim)
    {
        const size_t tensor_dim = dim + 2;

        // For output coordinate o and kernel coordinate k, dilated pooling
        // reads input index:
        //   input = o * stride + k * dilation
        // A Slice with:
        //   start = k * dilation
        //   step  = stride
        //   end   = start + (output_size - 1) * stride + 1
        // therefore gathers exactly this tap for every output coordinate.
        starts[tensor_dim] = tap_coordinates[dim] * plan.dilations[dim];
        ends[tensor_dim] = starts[tensor_dim] + (plan.output_shape[tensor_dim] - 1) * plan.strides[dim] + 1;
        steps[tensor_dim] = plan.strides[dim];
    }
}

bool TryLowerPoolNode(onnx::GraphProto& graph, const onnx::NodeProto& node,
                      google::protobuf::RepeatedPtrField<onnx::NodeProto>& lowered_nodes, const GraphIndex& index,
                      size_t& unique_id)
{
    PoolLoweringPlan plan;
    if (!TryBuildPoolLoweringPlan(node, index, plan))
    {
        return false;
    }

    std::vector<std::string> tap_outputs;
    tap_outputs.reserve(plan.tap_count);

    for (size_t tap_index = 0; tap_index < plan.tap_count; ++tap_index)
    {
        const bool single_tap = plan.tap_count == 1;
        const auto tap_coordinates = DecodeTapCoordinates(tap_index, plan.kernel_shape);

        std::vector<int64_t> starts;
        std::vector<int64_t> ends;
        std::vector<int64_t> axes;
        std::vector<int64_t> steps;
        BuildSliceParameters(plan, tap_coordinates, starts, ends, axes, steps);

        const auto base_name = MakeLoweredName(node.output(0), "tap", ++unique_id);
        const auto starts_name = base_name + "_starts";
        const auto ends_name = base_name + "_ends";
        const auto axes_name = base_name + "_axes";
        const auto steps_name = base_name + "_steps";
        const auto output_name = single_tap ? node.output(0) : base_name + "_output";

        AddInt64VectorInitializer(graph, starts_name, starts);
        AddInt64VectorInitializer(graph, ends_name, ends);
        AddInt64VectorInitializer(graph, axes_name, axes);
        AddInt64VectorInitializer(graph, steps_name, steps);

        AppendNode(lowered_nodes, node, "Slice", base_name + "_slice",
                   {node.input(0), starts_name, ends_name, axes_name, steps_name}, {output_name});

        if (!single_tap)
        {
            tap_outputs.push_back(output_name);
        }
    }

    if (plan.tap_count == 1)
    {
        return true;
    }

    if (plan.is_average_pool)
    {
        // With zero padding and fully-valid windows, AveragePool's divisor is
        // the kernel tap count. That lets a chain of Add nodes plus one scalar
        // Div reproduce the native operator exactly for fp32/fp16.
        std::string sum_output = tap_outputs[0];
        for (size_t i = 1; i < tap_outputs.size(); ++i)
        {
            const auto add_output = MakeLoweredName(node.output(0), "sum", ++unique_id);
            AppendNode(lowered_nodes, node, "Add", add_output + "_add", {sum_output, tap_outputs[i]}, {add_output});
            sum_output = add_output;
        }

        const auto divisor_name = MakeLoweredName(node.output(0), "divisor", ++unique_id);
        AddFloatScalarInitializer(graph, divisor_name, plan.element_type, static_cast<float>(plan.tap_count));
        AppendNode(lowered_nodes, node, "Div", MakeLoweredName(node.output(0), "average", ++unique_id),
                   {sum_output, divisor_name}, {node.output(0)});
    }
    else
    {
        // MaxPool over a dilated window is just the elementwise maximum across
        // the same tap tensors. ONNX Max accepts variadic inputs, but a binary
        // chain keeps the lowered graph simple for parser compatibility.
        std::string max_output = tap_outputs[0];
        for (size_t i = 1; i < tap_outputs.size(); ++i)
        {
            const bool is_last = i == tap_outputs.size() - 1;
            const auto next_output = is_last ? node.output(0) : MakeLoweredName(node.output(0), "max", ++unique_id);
            AppendNode(lowered_nodes, node, "Max", next_output + "_max", {max_output, tap_outputs[i]}, {next_output});
            max_output = next_output;
        }
    }

    return true;
}

void RewritePoolingDilations(onnx::GraphProto& graph)
{
    const GraphIndex index(graph);
    google::protobuf::RepeatedPtrField<onnx::NodeProto> lowered_nodes;
    size_t unique_id = 0;

    for (const auto& node : graph.node())
    {
        if (TryLowerPoolNode(graph, node, lowered_nodes, index, unique_id))
        {
            continue;
        }

        *lowered_nodes.Add() = node;
    }

    graph.mutable_node()->Swap(&lowered_nodes);
}

}  // namespace

void RunPoolingDilationCompatibilityForTensorRt(onnx::ModelProto& model_proto)
{
    RewritePoolingDilations(*model_proto.mutable_graph());
}

}  // namespace trt_rtx_ep
