// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "clip_bound_compatibility.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_rtx_ep
{
namespace
{

// Implementation overview:
//
// WebNN clamp(minValue/maxValue omitted) means "do not apply that bound".
// Chromium's ONNX conversion can encode this as Clip with -inf/+inf scalar
// inputs. That is legal ONNX math, but TRT-RTX imports Clip through an
// activation-style path where alpha/beta must be finite, so TRT rejects a
// graph that is otherwise executable.
//
// This pass rewrites only the unbounded sides away:
//   Clip(x, -inf, +inf)       -> Identity(x)
//   Clip(x, finite_min, +inf) -> Max(x, finite_min)
//   Clip(x, -inf, finite_max) -> Min(x, finite_max)
//
// The pass is intentionally not a general Clip decomposition. If both bounds
// are finite, native Clip is already the best representation for TRT. If an
// ONNX optional bound is omitted, the spec-defined default is a finite dtype
// limit rather than infinity, so that case stays native too. If a bound is
// dynamic/unknown/non-scalar/NaN or explicitly asks to produce an infinity,
// preserving the original node is safer than guessing.
constexpr const char* kExternalMemAddrLocation = "_MEM_ADDR_";
constexpr const char* kMainOnnxDomain = "";
constexpr const char* kAiOnnxDomain = "ai.onnx";

enum class BoundKind
{
    kMissing,
    kFinite,
    kNegativeInfinity,
    kPositiveInfinity,
    kOtherNonFinite,
    kUnknown,
};

struct TensorDataView
{
    const void* data = nullptr;
    size_t bytes = 0;
};

size_t GetTensorElementCount(const onnx::TensorProto& tensor)
{
    if (tensor.dims_size() == 0)
    {
        return 1;
    }

    size_t count = 1;
    for (const auto dim : tensor.dims())
    {
        if (dim <= 0)
        {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

size_t GetTensorElementSize(int32_t data_type)
{
    switch (data_type)
    {
        case onnx::TensorProto_DataType_FLOAT:
        case onnx::TensorProto_DataType_INT32:
        case onnx::TensorProto_DataType_UINT32:
            return 4;
        case onnx::TensorProto_DataType_DOUBLE:
        case onnx::TensorProto_DataType_INT64:
        case onnx::TensorProto_DataType_UINT64:
            return 8;
        case onnx::TensorProto_DataType_FLOAT16:
        case onnx::TensorProto_DataType_BFLOAT16:
        case onnx::TensorProto_DataType_INT16:
        case onnx::TensorProto_DataType_UINT16:
            return 2;
        case onnx::TensorProto_DataType_INT8:
        case onnx::TensorProto_DataType_UINT8:
        case onnx::TensorProto_DataType_BOOL:
            return 1;
        default:
            return 0;
    }
}

bool TryGetExternalDataView(const onnx::TensorProto& tensor, TensorDataView& data_view)
{
    if (tensor.data_location() != onnx::TensorProto_DataLocation_EXTERNAL)
    {
        return false;
    }

    std::string location;
    std::string offset;
    std::string length;
    for (const auto& entry : tensor.external_data())
    {
        if (entry.key() == "location")
        {
            location = entry.value();
        }
        else if (entry.key() == "offset")
        {
            offset = entry.value();
        }
        else if (entry.key() == "length")
        {
            length = entry.value();
        }
    }

    if (location != kExternalMemAddrLocation || offset.empty())
    {
        return false;
    }

    try
    {
        const uintptr_t ptr_value = static_cast<uintptr_t>(std::stoull(offset));
        data_view.data = reinterpret_cast<const void*>(ptr_value);
        data_view.bytes = length.empty() ? GetTensorElementCount(tensor) * GetTensorElementSize(tensor.data_type())
                                         : static_cast<size_t>(std::stoull(length));
    }
    catch (const std::exception&)
    {
        return false;
    }

    return data_view.data != nullptr && data_view.bytes > 0;
}

bool TryGetRawDataView(const onnx::TensorProto& tensor, TensorDataView& data_view)
{
    data_view = {};
    if (tensor.has_raw_data())
    {
        data_view.data = tensor.raw_data().data();
        data_view.bytes = static_cast<size_t>(tensor.raw_data().size());
        return true;
    }

    return TryGetExternalDataView(tensor, data_view);
}

template <typename T>
bool TryReadScalarBytes(const onnx::TensorProto& tensor, T& value)
{
    TensorDataView data_view;
    if (!TryGetRawDataView(tensor, data_view) || data_view.bytes < sizeof(T))
    {
        return false;
    }

    std::memcpy(&value, data_view.data, sizeof(T));
    return true;
}

BoundKind ClassifyFloatValue(float value)
{
    if (std::isfinite(value))
    {
        return BoundKind::kFinite;
    }
    if (value == std::numeric_limits<float>::infinity())
    {
        return BoundKind::kPositiveInfinity;
    }
    if (value == -std::numeric_limits<float>::infinity())
    {
        return BoundKind::kNegativeInfinity;
    }
    return BoundKind::kOtherNonFinite;
}

BoundKind ClassifyDoubleValue(double value)
{
    if (std::isfinite(value))
    {
        return BoundKind::kFinite;
    }
    if (value == std::numeric_limits<double>::infinity())
    {
        return BoundKind::kPositiveInfinity;
    }
    if (value == -std::numeric_limits<double>::infinity())
    {
        return BoundKind::kNegativeInfinity;
    }
    return BoundKind::kOtherNonFinite;
}

BoundKind ClassifyFloat16Bits(uint16_t bits)
{
    const uint16_t exponent = bits & 0x7c00;
    const uint16_t mantissa = bits & 0x03ff;
    if (exponent != 0x7c00)
    {
        return BoundKind::kFinite;
    }
    if (mantissa != 0)
    {
        return BoundKind::kOtherNonFinite;
    }
    return (bits & 0x8000) ? BoundKind::kNegativeInfinity : BoundKind::kPositiveInfinity;
}

BoundKind ClassifyBFloat16Bits(uint16_t bits)
{
    const uint16_t exponent = bits & 0x7f80;
    const uint16_t mantissa = bits & 0x007f;
    if (exponent != 0x7f80)
    {
        return BoundKind::kFinite;
    }
    if (mantissa != 0)
    {
        return BoundKind::kOtherNonFinite;
    }
    return (bits & 0x8000) ? BoundKind::kNegativeInfinity : BoundKind::kPositiveInfinity;
}

BoundKind ClassifyScalarTensor(const onnx::TensorProto& tensor)
{
    // Policy: only rewrite bounds that are compile-time scalar constants.
    // Dynamic, vector, or unreadable bounds may carry user/runtime semantics
    // that cannot be proven equivalent here, so they stay native and let TRT
    // make the final parser decision.
    if (GetTensorElementCount(tensor) != 1)
    {
        return BoundKind::kUnknown;
    }

    switch (tensor.data_type())
    {
        case onnx::TensorProto_DataType_FLOAT:
        {
            float value = 0.0f;
            if (TryReadScalarBytes(tensor, value))
            {
                return ClassifyFloatValue(value);
            }
            return tensor.float_data_size() == 1 ? ClassifyFloatValue(tensor.float_data(0)) : BoundKind::kUnknown;
        }
        case onnx::TensorProto_DataType_DOUBLE:
        {
            double value = 0.0;
            if (TryReadScalarBytes(tensor, value))
            {
                return ClassifyDoubleValue(value);
            }
            return tensor.double_data_size() == 1 ? ClassifyDoubleValue(tensor.double_data(0)) : BoundKind::kUnknown;
        }
        case onnx::TensorProto_DataType_FLOAT16:
        {
            uint16_t bits = 0;
            if (TryReadScalarBytes(tensor, bits))
            {
                return ClassifyFloat16Bits(bits);
            }
            return tensor.int32_data_size() == 1 ? ClassifyFloat16Bits(static_cast<uint16_t>(tensor.int32_data(0)))
                                                 : BoundKind::kUnknown;
        }
        case onnx::TensorProto_DataType_BFLOAT16:
        {
            uint16_t bits = 0;
            if (TryReadScalarBytes(tensor, bits))
            {
                return ClassifyBFloat16Bits(bits);
            }
            return tensor.int32_data_size() == 1 ? ClassifyBFloat16Bits(static_cast<uint16_t>(tensor.int32_data(0)))
                                                 : BoundKind::kUnknown;
        }
        case onnx::TensorProto_DataType_INT8:
        case onnx::TensorProto_DataType_UINT8:
        case onnx::TensorProto_DataType_INT16:
        case onnx::TensorProto_DataType_UINT16:
        case onnx::TensorProto_DataType_INT32:
        case onnx::TensorProto_DataType_UINT32:
        case onnx::TensorProto_DataType_INT64:
        case onnx::TensorProto_DataType_UINT64:
            return BoundKind::kFinite;
        default:
            return BoundKind::kUnknown;
    }
}

const onnx::TensorProto* FindConstantTensorAttribute(const onnx::NodeProto& node)
{
    if (node.op_type() != "Constant" ||
        (!node.domain().empty() && node.domain() != kMainOnnxDomain && node.domain() != kAiOnnxDomain))
    {
        return nullptr;
    }

    for (const auto& attr : node.attribute())
    {
        if (attr.name() == "value" && attr.type() == onnx::AttributeProto_AttributeType_TENSOR)
        {
            return &attr.t();
        }
    }
    return nullptr;
}

bool HasLegacyClipBoundsAttributes(const onnx::NodeProto& node)
{
    for (const auto& attr : node.attribute())
    {
        if (attr.name() == "min" || attr.name() == "max")
        {
            return true;
        }
    }
    return false;
}

struct GraphIndex
{
    explicit GraphIndex(const onnx::GraphProto& graph_proto)
    {
        for (const auto& initializer : graph_proto.initializer())
        {
            initializers[initializer.name()] = &initializer;
        }

        for (int node_index = 0; node_index < graph_proto.node_size(); ++node_index)
        {
            const auto& node = graph_proto.node(node_index);
            for (const auto& output : node.output())
            {
                if (!output.empty())
                {
                    producers[output] = &node;
                }
            }
        }
    }

    BoundKind ResolveBound(const onnx::NodeProto& node, int input_index) const
    {
        // ONNX Clip input 1 is min and input 2 is max. Missing optional
        // inputs are not the same as explicit +/-infinity constants: ONNX
        // defines omitted bounds as finite dtype limits, so those cases must
        // stay native unless we explicitly materialize the finite default.
        if (node.input_size() <= input_index || node.input(input_index).empty())
        {
            return BoundKind::kMissing;
        }

        // WebNN-generated models usually materialize clamp bounds either as
        // graph initializers or Constant nodes. We intentionally do not chase
        // through arithmetic, casts, or other producers because that would turn
        // this parser-compat pass into partial constant propagation.
        const auto& input_name = node.input(input_index);
        const auto initializer_it = initializers.find(input_name);
        if (initializer_it != initializers.end())
        {
            return ClassifyScalarTensor(*initializer_it->second);
        }

        const auto producer_it = producers.find(input_name);
        if (producer_it != producers.end())
        {
            if (const auto* tensor = FindConstantTensorAttribute(*producer_it->second))
            {
                return ClassifyScalarTensor(*tensor);
            }
        }

        return BoundKind::kUnknown;
    }

    std::unordered_map<std::string, const onnx::TensorProto*> initializers;
    std::unordered_map<std::string, const onnx::NodeProto*> producers;
};

bool IsLowerUnbounded(BoundKind kind)
{
    // Only explicit WebNN-style -inf is unbounded for this rewrite. A missing
    // ONNX Clip min input means the finite lowest value for the input dtype,
    // which is observably different for -inf inputs.
    return kind == BoundKind::kNegativeInfinity;
}

bool IsUpperUnbounded(BoundKind kind)
{
    // Only explicit WebNN-style +inf is unbounded for this rewrite. A missing
    // ONNX Clip max input means the finite max value for the input dtype, which
    // is observably different for +inf inputs.
    return kind == BoundKind::kPositiveInfinity;
}

void ReplaceNode(onnx::NodeProto& node, const std::string& op_type, const std::vector<std::string>& inputs)
{
    // Keep the original output name and doc_string. Downstream edges and the
    // provider's proto-node bookkeeping still see the same logical node result;
    // only the parser-hostile Clip opcode/inputs are changed.
    node.set_op_type(op_type);
    node.clear_attribute();
    node.mutable_input()->Clear();
    for (const auto& input : inputs)
    {
        node.add_input(input);
    }
}

void RewriteClipBounds(onnx::GraphProto& graph)
{
    const GraphIndex index(graph);

    for (auto& node : *graph.mutable_node())
    {
        if (node.op_type() != "Clip" ||
            (!node.domain().empty() && node.domain() != kMainOnnxDomain && node.domain() != kAiOnnxDomain) ||
            node.input_size() < 1 || node.input(0).empty() || node.output_size() != 1)
        {
            continue;
        }

        // Older ONNX Clip opsets encoded min/max as attributes instead of
        // optional inputs. Missing input 1/2 is unbounded only for modern
        // input-form Clip, so leave legacy attribute-form Clip untouched.
        if (HasLegacyClipBoundsAttributes(node))
        {
            continue;
        }

        const auto lower_bound = index.ResolveBound(node, 1);
        const auto upper_bound = index.ResolveBound(node, 2);

        // Safe rewrites only for explicit +/-inf bounds:
        //   * explicit -inf lower and +inf upper -> Identity
        //   * finite lower and explicit +inf     -> Max
        //   * explicit -inf and finite upper     -> Min
        //
        // Missing ONNX Clip bounds, explicit non-finite producing cases such
        // as min=+inf/max=-inf, NaN bounds, or unknown dynamic bounds are
        // intentionally left untouched.
        if (IsLowerUnbounded(lower_bound) && IsUpperUnbounded(upper_bound))
        {
            ReplaceNode(node, "Identity", {node.input(0)});
        }
        else if (lower_bound == BoundKind::kFinite && IsUpperUnbounded(upper_bound))
        {
            ReplaceNode(node, "Max", {node.input(0), node.input(1)});
        }
        else if (IsLowerUnbounded(lower_bound) && upper_bound == BoundKind::kFinite)
        {
            ReplaceNode(node, "Min", {node.input(0), node.input(2)});
        }
    }
}

}  // namespace

void RunClipBoundCompatibilityForTensorRt(onnx::ModelProto& model_proto)
{
    RewriteClipBounds(*model_proto.mutable_graph());
}

}  // namespace trt_rtx_ep
