// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Q/DQ lowering pass for the TensorRT-RTX EP.
//
// Design summary (see also qdq_lowering.h):
//
//   * The policy layer (Evaluate*Lowering, ShouldLowerQuantizeLinear,
//     BaseMustLower*, IsSymmetricConstantDequantizeCandidate, ...) decides
//     *whether* a Q/DQ node must be rewritten. It is intentionally separated
//     from the rewrite layer so rule changes never touch the arithmetic.
//
//   * The rewrite layer (driver loop in RunQdqLoweringForTensorRt +
//     TryFoldConstantDequantizeLinear + Append* helpers) materializes the
//     ONNX math for DQ / Q in terms of Cast / Sub / Mul / Div / Add / Round
//     / Min / Max / Cast, with a few numerical refinements (integer-first
//     subtraction for int32/uint32, fp32 promotion for int16/uint16 with
//     fp16/bf16 scales, zero_point dtype promotion for safe subtraction).
//
//   * Every emitted helper node inherits the *original* ORT node id via
//     NodeProto::doc_string so the provider's proto->ORT remap can still
//     credit ownership to the correct ORT node (see provider comments near
//     build_proto_to_ort_index).
//
//   * Graph ownership invariant: every Q/DQ node in the input proto ends up
//     represented in the output proto either verbatim (kept native) or as a
//     set of lowered nodes sharing its doc_string (and, for folded constant
//     DQ, an entry in LoweredQdqInfo::folded_constant_nodes). The driver
//     *never* drops a node on the floor.

#include "qdq_lowering.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "proto_node_id_utils.h"

namespace trt_rtx_ep
{
namespace
{
// Sentinel location string used by the provider's OrtGraphToProto handler.
// When a TensorProto carries this location, `offset` is NOT a file offset;
// it is a raw pointer to ORT-owned initializer bytes, reinterpret_cast to
// int64_t. TryGetTensorDataView knows how to decode that convention.
// MUST match the constant in tensorrt_rtx_execution_provider.cc.
constexpr const char* kExternalMemAddrLocation = "_MEM_ADDR_";
constexpr const char* kMainOnnxDomain = "";

struct TensorDataView
{
    const void* data = nullptr;
    size_t bytes = 0;
    std::vector<uint8_t> owned_bytes;
};

struct TensorMetadata
{
    std::optional<int32_t> element_type;
    std::vector<int64_t> shape;
    bool is_graph_input = false;
};

struct GraphIndex
{
    explicit GraphIndex(onnx::GraphProto& graph_proto)
        : graph(graph_proto)
    {
        Build();
    }

    void RebuildTensorMetadata(const std::string& name)
    {
        tensor_metadata.erase(name);
        AddValueInfo(name, nullptr, true);
        for (const auto& value_info : graph.input())
        {
            if (value_info.name() == name)
            {
                AddValueInfo(name, &value_info, true);
            }
        }
        for (const auto& value_info : graph.value_info())
        {
            if (value_info.name() == name)
            {
                AddValueInfo(name, &value_info, false);
            }
        }
        for (const auto& value_info : graph.output())
        {
            if (value_info.name() == name)
            {
                AddValueInfo(name, &value_info, false);
            }
        }
        if (const auto* initializer = FindInitializer(name))
        {
            auto& metadata = tensor_metadata[name];
            metadata.element_type = initializer->data_type();
            metadata.shape.assign(initializer->dims().begin(), initializer->dims().end());
        }
    }

    const onnx::TensorProto* FindInitializer(const std::string& name) const
    {
        auto it = initializer_map.find(name);
        return it == initializer_map.end() ? nullptr : it->second;
    }

    std::optional<int32_t> FindTensorElementType(const std::string& name) const
    {
        auto it = tensor_metadata.find(name);
        if (it == tensor_metadata.end())
        {
            return std::nullopt;
        }
        return it->second.element_type;
    }

    bool IsGraphInput(const std::string& name) const
    {
        auto it = tensor_metadata.find(name);
        return it != tensor_metadata.end() && it->second.is_graph_input;
    }

    std::optional<int64_t> FindTensorRank(const std::string& name) const
    {
        auto it = tensor_metadata.find(name);
        if (it == tensor_metadata.end() || it->second.shape.empty())
        {
            return std::nullopt;
        }
        return static_cast<int64_t>(it->second.shape.size());
    }

    bool TryGetTensorShape(const std::string& name, std::vector<int64_t>& shape) const
    {
        shape.clear();
        auto it = tensor_metadata.find(name);
        if (it == tensor_metadata.end())
        {
            return false;
        }

        if (it->second.shape.empty())
        {
            return true;
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

    void NoteInitializer(const onnx::TensorProto& initializer)
    {
        initializer_map[initializer.name()] = &initializer;
        auto& metadata = tensor_metadata[initializer.name()];
        metadata.element_type = initializer.data_type();
        metadata.shape.assign(initializer.dims().begin(), initializer.dims().end());
    }

    const onnx::NodeProto* GetNode(size_t index) const
    {
        if (index >= static_cast<size_t>(graph.node_size()))
        {
            return nullptr;
        }
        return &graph.node(static_cast<int>(index));
    }

    const onnx::NodeProto* FindProducer(const std::string& name) const
    {
        auto it = producer_node_indices.find(name);
        return it == producer_node_indices.end() ? nullptr : GetNode(it->second);
    }

    const std::vector<size_t>* FindConsumerNodeIndices(const std::string& name) const
    {
        auto it = consumer_node_indices.find(name);
        return it == consumer_node_indices.end() ? nullptr : &it->second;
    }

    onnx::GraphProto& graph;
    std::unordered_map<std::string, const onnx::TensorProto*> initializer_map;
    std::unordered_map<std::string, TensorMetadata> tensor_metadata;
    std::unordered_map<std::string, size_t> producer_node_indices;
    std::unordered_map<std::string, std::vector<size_t>> consumer_node_indices;

private:
    void Build()
    {
        initializer_map.clear();
        tensor_metadata.clear();
        producer_node_indices.clear();
        consumer_node_indices.clear();
        for (const auto& initializer : graph.initializer())
        {
            initializer_map[initializer.name()] = &initializer;
            auto& metadata = tensor_metadata[initializer.name()];
            metadata.element_type = initializer.data_type();
            metadata.shape.assign(initializer.dims().begin(), initializer.dims().end());
        }
        for (const auto& value_info : graph.input())
        {
            AddValueInfo(value_info.name(), &value_info, true);
        }
        for (const auto& value_info : graph.value_info())
        {
            AddValueInfo(value_info.name(), &value_info, false);
        }
        for (const auto& value_info : graph.output())
        {
            AddValueInfo(value_info.name(), &value_info, false);
        }
        for (int node_index = 0; node_index < graph.node_size(); ++node_index)
        {
            const auto& node = graph.node(node_index);
            for (const auto& input_name : node.input())
            {
                if (!input_name.empty())
                {
                    consumer_node_indices[input_name].push_back(static_cast<size_t>(node_index));
                }
            }
            for (const auto& output_name : node.output())
            {
                if (!output_name.empty())
                {
                    producer_node_indices[output_name] = static_cast<size_t>(node_index);
                }
            }
        }
    }

    void AddValueInfo(const std::string& name, const onnx::ValueInfoProto* value_info, bool is_input)
    {
        auto& metadata = tensor_metadata[name];
        metadata.is_graph_input = metadata.is_graph_input || is_input;
        if (value_info == nullptr || !value_info->has_type() || !value_info->type().has_tensor_type())
        {
            return;
        }

        const auto& tensor_type = value_info->type().tensor_type();
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

bool TryReadIntegerTensorData(const onnx::TensorProto& tensor, std::vector<int64_t>& values);
bool TryReadFloatingTensorData(const onnx::TensorProto& tensor, std::vector<double>& values);

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
        return 1;
    default:
        return 0;
    }
}

float Float16BitsToFloat(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1f;
    const uint32_t mantissa = bits & 0x03ff;

    uint32_t result_bits = 0;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            result_bits = sign;
        }
        else
        {
            uint32_t normalized_mantissa = mantissa;
            uint32_t normalized_exponent = 0;
            while ((normalized_mantissa & 0x0400) == 0)
            {
                normalized_mantissa <<= 1;
                ++normalized_exponent;
            }
            normalized_mantissa &= 0x03ff;
            result_bits = sign | ((127 - 15 - normalized_exponent) << 23) | (normalized_mantissa << 13);
        }
    }
    else if (exponent == 0x1f)
    {
        result_bits = sign | 0x7f800000 | (mantissa << 13);
    }
    else
    {
        result_bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &result_bits, sizeof(value));
    return value;
}

float BFloat16BitsToFloat(uint16_t bits)
{
    const uint32_t raw = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

template <typename T>
bool CopyTypedValues(const void* source, size_t source_count, std::vector<uint8_t>& dest)
{
    if (source_count == 0)
    {
        dest.clear();
        return true;
    }
    dest.resize(source_count * sizeof(T));
    std::memcpy(dest.data(), source, dest.size());
    return true;
}

bool PackTensorFromTypedFields(const onnx::TensorProto& tensor, TensorDataView& data_view)
{
    const size_t element_count = GetTensorElementCount(tensor);
    if (element_count == 0)
    {
        return false;
    }

    switch (tensor.data_type())
    {
    case onnx::TensorProto_DataType_FLOAT:
        if (static_cast<size_t>(tensor.float_data_size()) == element_count)
        {
            return CopyTypedValues<float>(tensor.float_data().data(), element_count, data_view.owned_bytes);
        }
        break;
    case onnx::TensorProto_DataType_DOUBLE:
        if (static_cast<size_t>(tensor.double_data_size()) == element_count)
        {
            return CopyTypedValues<double>(tensor.double_data().data(), element_count, data_view.owned_bytes);
        }
        break;
    case onnx::TensorProto_DataType_INT64:
        if (static_cast<size_t>(tensor.int64_data_size()) == element_count)
        {
            return CopyTypedValues<int64_t>(tensor.int64_data().data(), element_count, data_view.owned_bytes);
        }
        break;
    case onnx::TensorProto_DataType_UINT64:
        if (static_cast<size_t>(tensor.uint64_data_size()) == element_count)
        {
            return CopyTypedValues<uint64_t>(tensor.uint64_data().data(), element_count, data_view.owned_bytes);
        }
        break;
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_BFLOAT16:
    case onnx::TensorProto_DataType_INT8:
    case onnx::TensorProto_DataType_UINT8:
    case onnx::TensorProto_DataType_INT16:
    case onnx::TensorProto_DataType_UINT16:
    case onnx::TensorProto_DataType_INT32:
    {
        if (static_cast<size_t>(tensor.int32_data_size()) != element_count)
        {
            break;
        }
        data_view.owned_bytes.resize(element_count * GetTensorElementSize(tensor.data_type()));
        for (size_t i = 0; i < element_count; ++i)
        {
            const int32_t value = tensor.int32_data(static_cast<int>(i));
            switch (tensor.data_type())
            {
            case onnx::TensorProto_DataType_FLOAT16:
            case onnx::TensorProto_DataType_BFLOAT16:
            {
                const uint16_t narrowed = static_cast<uint16_t>(value & 0xffff);
                std::memcpy(data_view.owned_bytes.data() + i * sizeof(uint16_t), &narrowed, sizeof(uint16_t));
                break;
            }
            case onnx::TensorProto_DataType_INT8:
            {
                const int8_t narrowed = static_cast<int8_t>(value);
                std::memcpy(data_view.owned_bytes.data() + i, &narrowed, sizeof(narrowed));
                break;
            }
            case onnx::TensorProto_DataType_UINT8:
            {
                const uint8_t narrowed = static_cast<uint8_t>(value);
                std::memcpy(data_view.owned_bytes.data() + i, &narrowed, sizeof(narrowed));
                break;
            }
            case onnx::TensorProto_DataType_INT16:
            {
                const int16_t narrowed = static_cast<int16_t>(value);
                std::memcpy(data_view.owned_bytes.data() + i * sizeof(int16_t), &narrowed, sizeof(narrowed));
                break;
            }
            case onnx::TensorProto_DataType_UINT16:
            {
                const uint16_t narrowed = static_cast<uint16_t>(value);
                std::memcpy(data_view.owned_bytes.data() + i * sizeof(uint16_t), &narrowed, sizeof(narrowed));
                break;
            }
            case onnx::TensorProto_DataType_INT32:
            {
                std::memcpy(data_view.owned_bytes.data() + i * sizeof(int32_t), &value, sizeof(value));
                break;
            }
            default:
                break;
            }
        }
        break;
    }
    case onnx::TensorProto_DataType_UINT32:
        if (static_cast<size_t>(tensor.uint64_data_size()) == element_count)
        {
            data_view.owned_bytes.resize(element_count * sizeof(uint32_t));
            for (size_t i = 0; i < element_count; ++i)
            {
                const uint32_t value = static_cast<uint32_t>(tensor.uint64_data(static_cast<int>(i)));
                std::memcpy(data_view.owned_bytes.data() + i * sizeof(uint32_t), &value, sizeof(value));
            }
            break;
        }
        return false;
    default:
        return false;
    }

    if (data_view.owned_bytes.empty())
    {
        return false;
    }

    data_view.data = data_view.owned_bytes.data();
    data_view.bytes = data_view.owned_bytes.size();
    return true;
}

// Uniform accessor for initializer bytes regardless of how the serializer
// emitted them. Three supported layouts, in priority order:
//   1. Inline raw_data           - returned as a pointer to the proto's bytes.
//   2. External with location
//      == "_MEM_ADDR_"           - `offset` is a pointer into ORT-owned memory
//                                   (see kExternalMemAddrLocation). We never
//                                   copy; the view aliases ORT's buffer for
//                                   the lifetime of this lowering pass.
//   3. Typed fields (int32_data,
//      float_data, ...)          - packed into owned_bytes so callers can
//                                   treat all tensors uniformly.
// Anything else (e.g. external on-disk) is intentionally unsupported here
// because ORT never hands us those during EP lowering.
bool TryGetTensorDataView(const onnx::TensorProto& tensor, TensorDataView& data_view)
{
    data_view = {};
    if (tensor.has_raw_data())
    {
        data_view.data = tensor.raw_data().data();
        data_view.bytes = static_cast<size_t>(tensor.raw_data().size());
        return true;
    }

    if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL)
    {
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

        if (location == kExternalMemAddrLocation && !offset.empty())
        {
            const uintptr_t ptr_value = static_cast<uintptr_t>(std::stoull(offset));
            data_view.data = reinterpret_cast<const void*>(ptr_value);
            if (!length.empty())
            {
                data_view.bytes = static_cast<size_t>(std::stoull(length));
            }
            else
            {
                data_view.bytes = GetTensorElementCount(tensor) * GetTensorElementSize(tensor.data_type());
            }
            return data_view.data != nullptr && data_view.bytes > 0;
        }
    }

    return PackTensorFromTypedFields(tensor, data_view);
}

template <typename T>
bool TensorBytesHaveAnyNonZero(const TensorDataView& data_view, size_t element_count)
{
    if (data_view.bytes < element_count * sizeof(T))
    {
        return false;
    }
    const auto* values = static_cast<const T*>(data_view.data);
    for (size_t i = 0; i < element_count; ++i)
    {
        if (values[i] != static_cast<T>(0))
        {
            return true;
        }
    }
    return false;
}

bool TensorHasAnyNonZeroValue(const onnx::TensorProto& tensor)
{
    TensorDataView data_view;
    if (!TryGetTensorDataView(tensor, data_view))
    {
        return false;
    }

    const size_t element_count = GetTensorElementCount(tensor);
    switch (tensor.data_type())
    {
    case onnx::TensorProto_DataType_INT8:
        return TensorBytesHaveAnyNonZero<int8_t>(data_view, element_count);
    case onnx::TensorProto_DataType_UINT8:
        return TensorBytesHaveAnyNonZero<uint8_t>(data_view, element_count);
    case onnx::TensorProto_DataType_INT16:
        return TensorBytesHaveAnyNonZero<int16_t>(data_view, element_count);
    case onnx::TensorProto_DataType_UINT16:
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_BFLOAT16:
        return TensorBytesHaveAnyNonZero<uint16_t>(data_view, element_count);
    case onnx::TensorProto_DataType_INT32:
        return TensorBytesHaveAnyNonZero<int32_t>(data_view, element_count);
    case onnx::TensorProto_DataType_UINT32:
        return TensorBytesHaveAnyNonZero<uint32_t>(data_view, element_count);
    case onnx::TensorProto_DataType_INT64:
        return TensorBytesHaveAnyNonZero<int64_t>(data_view, element_count);
    case onnx::TensorProto_DataType_UINT64:
        return TensorBytesHaveAnyNonZero<uint64_t>(data_view, element_count);
    case onnx::TensorProto_DataType_FLOAT:
        return TensorBytesHaveAnyNonZero<float>(data_view, element_count);
    case onnx::TensorProto_DataType_DOUBLE:
        return TensorBytesHaveAnyNonZero<double>(data_view, element_count);
    default:
        return false;
    }
}

// TRT-RTX native DequantizeLinear accepts low-bit integer inputs
// (int8/uint8/int16/uint16). It rejects 32-bit integer DQ inputs, which ONNX
// spec allows but TRT's Q/DQ path does not model. Those must be lowered.
//
// Policy knob: if TRT widens its DQ input support in a future release, revisit
// this predicate together with GetDequantizeIntegerMathType.
bool IsTensorRtUnsupportedQdqIntegerType(int32_t data_type)
{
    return data_type == onnx::TensorProto_DataType_INT32 || data_type == onnx::TensorProto_DataType_UINT32;
}

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

std::string MakeLoweredName(const std::string& base, const char* suffix, size_t unique_id)
{
    std::ostringstream oss;
    oss << "__trt_rtx_qdq_lowered_" << unique_id << "_" << base << "_" << suffix;
    return oss.str();
}

void SetScalarInitializerData(onnx::TensorProto& tensor_proto, int32_t data_type, int64_t int_value)
{
    tensor_proto.set_data_type(data_type);
    tensor_proto.set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    switch (data_type)
    {
    case onnx::TensorProto_DataType_INT8:
    {
        const int8_t value = static_cast<int8_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_UINT8:
    {
        const uint8_t value = static_cast<uint8_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_INT16:
    {
        const int16_t value = static_cast<int16_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_UINT16:
    {
        const uint16_t value = static_cast<uint16_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_INT32:
    {
        const int32_t value = static_cast<int32_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_UINT32:
    {
        const uint32_t value = static_cast<uint32_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_INT64:
    {
        const int64_t value = int_value;
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    case onnx::TensorProto_DataType_UINT64:
    {
        const uint64_t value = static_cast<uint64_t>(int_value);
        tensor_proto.set_raw_data(&value, sizeof(value));
        break;
    }
    default:
        break;
    }
}

onnx::TensorProto* AddScalarInitializer(onnx::GraphProto& graph, GraphIndex& index, const std::string& name,
                                        int32_t data_type, int64_t int_value)
{
    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    SetScalarInitializerData(*tensor_proto, data_type, int_value);
    index.NoteInitializer(*tensor_proto);
    return tensor_proto;
}

onnx::TensorProto* AddInt64VectorInitializer(onnx::GraphProto& graph, GraphIndex& index, const std::string& name,
                                             const std::vector<int64_t>& values)
{
    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(onnx::TensorProto_DataType_INT64);
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    tensor_proto->add_dims(static_cast<int64_t>(values.size()));
    tensor_proto->set_raw_data(values.data(), values.size() * sizeof(int64_t));
    index.NoteInitializer(*tensor_proto);
    return tensor_proto;
}

onnx::TensorProto* AddCopiedInitializerWithShape(onnx::GraphProto& graph, GraphIndex& index, const std::string& name,
                                                 const onnx::TensorProto& source_tensor,
                                                 const std::vector<int64_t>& new_shape)
{
    TensorDataView data_view;
    if (!TryGetTensorDataView(source_tensor, data_view))
    {
        return nullptr;
    }

    const size_t element_size = GetTensorElementSize(source_tensor.data_type());
    const size_t element_count = GetTensorElementCount(source_tensor);
    if (element_size == 0 || data_view.bytes < element_size * element_count)
    {
        return nullptr;
    }

    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(source_tensor.data_type());
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    for (const auto dim : new_shape)
    {
        tensor_proto->add_dims(dim);
    }
    tensor_proto->set_raw_data(data_view.data, element_size * element_count);
    index.NoteInitializer(*tensor_proto);
    return tensor_proto;
}

onnx::TensorProto* AddPromotedIntegerInitializer(onnx::GraphProto& graph, GraphIndex& index, const std::string& name,
                                                 const onnx::TensorProto& source_tensor, int32_t promoted_type)
{
    std::vector<int64_t> values;
    if (!TryReadIntegerTensorData(source_tensor, values))
    {
        return nullptr;
    }

    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(promoted_type);
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    for (const auto dim : source_tensor.dims())
    {
        tensor_proto->add_dims(dim);
    }

    if (promoted_type == onnx::TensorProto_DataType_INT32)
    {
        std::vector<int32_t> promoted(values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            promoted[i] = static_cast<int32_t>(values[i]);
        }
        tensor_proto->set_raw_data(promoted.data(), promoted.size() * sizeof(int32_t));
    }
    else
    {
        std::vector<int64_t> promoted(values.begin(), values.end());
        tensor_proto->set_raw_data(promoted.data(), promoted.size() * sizeof(int64_t));
    }
    index.NoteInitializer(*tensor_proto);
    return tensor_proto;
}

bool TryReadIntegerTensorData(const onnx::TensorProto& tensor, std::vector<int64_t>& values)
{
    TensorDataView data_view;
    if (!TryGetTensorDataView(tensor, data_view))
    {
        return false;
    }

    const size_t element_count = GetTensorElementCount(tensor);
    values.resize(element_count);
    switch (tensor.data_type())
    {
    case onnx::TensorProto_DataType_INT8:
    {
        const auto* source = static_cast<const int8_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = source[i];
        return true;
    }
    case onnx::TensorProto_DataType_UINT8:
    {
        const auto* source = static_cast<const uint8_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = source[i];
        return true;
    }
    case onnx::TensorProto_DataType_INT16:
    {
        const auto* source = static_cast<const int16_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = source[i];
        return true;
    }
    case onnx::TensorProto_DataType_UINT16:
    {
        const auto* source = static_cast<const uint16_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = source[i];
        return true;
    }
    case onnx::TensorProto_DataType_INT32:
    {
        const auto* source = static_cast<const int32_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = source[i];
        return true;
    }
    case onnx::TensorProto_DataType_UINT32:
    {
        const auto* source = static_cast<const uint32_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = static_cast<int64_t>(source[i]);
        return true;
    }
    case onnx::TensorProto_DataType_INT64:
    {
        const auto* source = static_cast<const int64_t*>(data_view.data);
        values.assign(source, source + element_count);
        return true;
    }
    case onnx::TensorProto_DataType_UINT64:
    {
        const auto* source = static_cast<const uint64_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = static_cast<int64_t>(source[i]);
        return true;
    }
    default:
        return false;
    }
}

bool TryReadFloatingTensorData(const onnx::TensorProto& tensor, std::vector<double>& values)
{
    TensorDataView data_view;
    if (!TryGetTensorDataView(tensor, data_view))
    {
        return false;
    }

    const size_t element_count = GetTensorElementCount(tensor);
    values.resize(element_count);
    switch (tensor.data_type())
    {
    case onnx::TensorProto_DataType_FLOAT:
    {
        const auto* source = static_cast<const float*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = static_cast<double>(source[i]);
        return true;
    }
    case onnx::TensorProto_DataType_DOUBLE:
    {
        const auto* source = static_cast<const double*>(data_view.data);
        values.assign(source, source + element_count);
        return true;
    }
    case onnx::TensorProto_DataType_FLOAT16:
    {
        const auto* source = static_cast<const uint16_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = static_cast<double>(Float16BitsToFloat(source[i]));
        return true;
    }
    case onnx::TensorProto_DataType_BFLOAT16:
    {
        const auto* source = static_cast<const uint16_t*>(data_view.data);
        for (size_t i = 0; i < element_count; ++i)
            values[i] = static_cast<double>(BFloat16BitsToFloat(source[i]));
        return true;
    }
    default:
        return false;
    }
}

// ONNX requires Q/DQ scales to be strictly positive and finite. Zero or
// negative scales make the lowered arithmetic ill-defined (Div->inf/NaN,
// clamp bounds flip), and NaN/Inf contaminate everything downstream.
//
// We don't try to "rescue" bad scales -- the caller bails out of lowering
// and lets TRT (or ORT validation) be the one to reject a malformed node.
bool IsPositiveScaleTensor(const onnx::TensorProto& tensor)
{
    std::vector<double> values;
    if (!TryReadFloatingTensorData(tensor, values) || values.empty())
    {
        return false;
    }
    return std::all_of(values.begin(), values.end(),
                       [](double value)
                       {
                           return value > 0.0 && std::isfinite(value);
                       });
}

// Given a flat output element index, work out which flat index of the
// scale/zp parameter tensor it should read from.
//
// Handles all three Q/DQ broadcasting flavors the ONNX spec defines:
//   * per-tensor              : parameter rank 0 or 1 element -> index 0
//   * per-axis (1-D)          : parameter matches one input axis (default
//                               axis=1, negative axes normalized)
//   * block quantization      : parameter along `axis` is (input_dim /
//                               block_size); floor-divide the input axis
//                               coordinate by block_size to find the
//                               parameter coordinate.
//   * full-rank broadcast     : parameter rank == input rank with each dim
//                               either equal to the input or 1.
//
// Returns false if shapes are inconsistent with any of the above; the caller
// treats that as "unsupported layout, leave node native" rather than guessing.
bool TryResolveBroadcastIndex(size_t element_index, const onnx::TensorProto& input_tensor,
                              const onnx::TensorProto& parameter_tensor, std::optional<int64_t> axis_attr,
                              int64_t block_size, size_t& parameter_index)
{
    std::vector<int64_t> input_shape(input_tensor.dims().begin(), input_tensor.dims().end());
    const size_t input_rank = input_shape.size();
    const size_t parameter_rank = static_cast<size_t>(parameter_tensor.dims_size());
    const size_t parameter_elements = GetTensorElementCount(parameter_tensor);
    if (parameter_rank == 0 || parameter_elements == 1)
    {
        parameter_index = 0;
        return true;
    }

    int64_t axis = axis_attr.value_or(1);
    if (axis < 0)
    {
        axis += static_cast<int64_t>(input_rank);
    }
    if ((parameter_rank == 1 && input_rank > 1) && (axis < 0 || axis >= static_cast<int64_t>(input_rank)))
    {
        return false;
    }

    std::vector<size_t> input_strides(input_rank, 1);
    for (size_t dim = input_rank; dim-- > 1;)
    {
        input_strides[dim - 1] = input_strides[dim] * static_cast<size_t>(input_shape[dim]);
    }

    std::vector<size_t> parameter_strides(parameter_rank == 0 ? 1 : parameter_rank, 1);
    if (parameter_rank > 1)
    {
        for (size_t dim = parameter_rank; dim-- > 1;)
        {
            parameter_strides[dim - 1] =
                parameter_strides[dim] * static_cast<size_t>(parameter_tensor.dims(static_cast<int>(dim)));
        }
    }

    std::vector<size_t> input_indices(input_rank, 0);
    size_t remaining_index = element_index;
    for (size_t dim = 0; dim < input_rank; ++dim)
    {
        const size_t stride = input_strides[dim];
        input_indices[dim] = remaining_index / stride;
        remaining_index %= stride;
    }

    if (parameter_rank == 1)
    {
        if (input_rank <= 1)
        {
            const int64_t input_dim = input_shape.empty() ? 1 : input_shape[0];
            const int64_t parameter_dim = parameter_tensor.dims(0);
            if (parameter_dim == input_dim)
            {
                parameter_index = input_indices.empty() ? 0 : input_indices[0];
                return true;
            }
            if (parameter_dim == 1)
            {
                parameter_index = 0;
                return true;
            }
            if (block_size > 1 && parameter_dim > 0 && parameter_dim * block_size == input_dim)
            {
                parameter_index = (input_indices.empty() ? 0 : input_indices[0]) / static_cast<size_t>(block_size);
                return parameter_index < parameter_elements;
            }
            return false;
        }

        const size_t axis_index = static_cast<size_t>(axis);
        const int64_t parameter_dim = parameter_tensor.dims(0);
        const int64_t input_dim = input_shape[axis_index];
        size_t logical_index = input_indices[axis_index];
        if (block_size > 1)
        {
            if (parameter_dim <= 0 || parameter_dim * block_size != input_dim)
            {
                return false;
            }
            logical_index /= static_cast<size_t>(block_size);
        }
        else if (parameter_dim != 1 && parameter_dim != input_dim)
        {
            return false;
        }

        parameter_index = parameter_dim == 1 ? 0 : logical_index;
        return parameter_index < parameter_elements;
    }

    if (parameter_rank != input_rank)
    {
        return false;
    }

    parameter_index = 0;
    for (size_t dim = 0; dim < input_rank; ++dim)
    {
        const int64_t parameter_dim = parameter_tensor.dims(static_cast<int>(dim));
        const int64_t input_dim = input_shape[dim];
        size_t logical_index = input_indices[dim];

        if (dim == static_cast<size_t>(axis) && block_size > 1)
        {
            if (parameter_dim > 0 && parameter_dim * block_size == input_dim)
            {
                logical_index /= static_cast<size_t>(block_size);
            }
            else if (parameter_dim != 1 && parameter_dim != input_dim)
            {
                return false;
            }
        }
        else if (parameter_dim != 1 && parameter_dim != input_dim)
        {
            return false;
        }

        parameter_index += (parameter_dim == 1 ? 0 : logical_index) * parameter_strides[dim];
    }

    return parameter_index < parameter_elements;
}

// Emits a lowered helper node inheriting the source Q/DQ node's doc_string.
//
// CRITICAL INVARIANT: doc_string carries the original ORT node id (written
// by the provider's OrtGraphToProto handler). Every lowered helper MUST
// propagate it so the provider's build_proto_to_ort_index remap can credit
// supported-set membership back to the correct ORT node. Initializers we add
// (qmin, qmax, reshaped scale, promoted zp) are not NodeProtos and do not
// carry doc_string, which is intentional -- only executable nodes are
// subject to the parser's support decision.
onnx::NodeProto* AppendNode(google::protobuf::RepeatedPtrField<onnx::NodeProto>& nodes,
                            const onnx::NodeProto& source_node, const std::string& op_type, const std::string& name,
                            const std::vector<std::string>& inputs, const std::vector<std::string>& outputs)
{
    auto* new_node = nodes.Add();
    new_node->set_name(name);
    new_node->set_domain(kMainOnnxDomain);
    new_node->set_op_type(op_type);
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

void CopyNodeAttributes(const onnx::NodeProto& source_node, onnx::NodeProto& target_node)
{
    target_node.mutable_attribute()->CopyFrom(source_node.attribute());
}

bool TryGetQuantizedValueRange(int32_t data_type, int64_t& min_value, int64_t& max_value)
{
    switch (data_type)
    {
    case onnx::TensorProto_DataType_INT8:
        min_value = -128;
        max_value = 127;
        return true;
    case onnx::TensorProto_DataType_UINT8:
        min_value = 0;
        max_value = 255;
        return true;
    case onnx::TensorProto_DataType_INT16:
        min_value = std::numeric_limits<int16_t>::min();
        max_value = std::numeric_limits<int16_t>::max();
        return true;
    case onnx::TensorProto_DataType_UINT16:
        min_value = 0;
        max_value = std::numeric_limits<uint16_t>::max();
        return true;
    case onnx::TensorProto_DataType_INT32:
        min_value = std::numeric_limits<int32_t>::min();
        max_value = std::numeric_limits<int32_t>::max();
        return true;
    case onnx::TensorProto_DataType_UINT32:
        min_value = 0;
        max_value = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        return true;
    default:
        return false;
    }
}

// Pre-expand a per-axis or block-quantized scale/zp initializer to the
// input's full shape.
//
// Why we expand at compile time rather than emit a runtime Expand:
//   * Block-quant broadcasting (param_dim * block_size == input_dim along
//     `axis`) is NOT a standard ONNX Expand pattern; no TRT-supported op
//     encodes "repeat each row block_size times along axis N".
//   * Expanded initializer bytes are small for the Q/DQ cases we lower
//     (weights scales, channel/block scales), so the memory cost is modest.
//   * Keeping it as an initializer lets TRT constant-fold into weights
//     without any runtime cost.
onnx::TensorProto* AddExpandedInitializer(onnx::GraphProto& graph, GraphIndex& index, const std::string& name,
                                          const onnx::TensorProto& source_tensor,
                                          const std::vector<int64_t>& input_shape, std::optional<int64_t> axis_attr,
                                          int64_t block_size)
{
    TensorDataView data_view;
    if (!TryGetTensorDataView(source_tensor, data_view))
    {
        return nullptr;
    }

    const size_t element_size = GetTensorElementSize(source_tensor.data_type());
    if (element_size == 0)
    {
        return nullptr;
    }

    onnx::TensorProto input_shape_tensor;
    input_shape_tensor.set_data_type(source_tensor.data_type());
    for (const auto dim : input_shape)
    {
        input_shape_tensor.add_dims(dim);
    }

    const size_t output_count = GetTensorElementCount(input_shape_tensor);
    std::string expanded_raw_data;
    expanded_raw_data.resize(output_count * element_size);
    for (size_t i = 0; i < output_count; ++i)
    {
        size_t parameter_index = 0;
        if (!TryResolveBroadcastIndex(i, input_shape_tensor, source_tensor, axis_attr, block_size, parameter_index))
        {
            return nullptr;
        }
        std::memcpy(expanded_raw_data.data() + i * element_size,
                    static_cast<const uint8_t*>(data_view.data) + parameter_index * element_size, element_size);
    }

    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(source_tensor.data_type());
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    for (const auto dim : input_shape)
    {
        tensor_proto->add_dims(dim);
    }
    tensor_proto->set_raw_data(expanded_raw_data.data(), expanded_raw_data.size());
    index.NoteInitializer(*tensor_proto);
    return tensor_proto;
}

// Prepare the scale / zero_point input for the lowered Mul / Div / Sub / Add
// so that standard ONNX right-aligned broadcasting reproduces Q/DQ semantics:
//
//   * Per-tensor (scalar / single-element parameter): no reshape needed.
//   * Block quantization (block_size > 1): eagerly expand the initializer
//     via AddExpandedInitializer (standard broadcast cannot express it).
//   * Per-axis on the *last* axis: no reshape -- ONNX broadcasting already
//     aligns a 1-D tensor to the last axis.
//   * Per-axis on any other axis: emit a Reshape that places the parameter
//     dim at `axis` and 1s elsewhere, so downstream broadcasts correctly.
//   * Full-rank parameter: accepted as-is only when rank matches input rank;
//     unusual ONNX layouts fall through as "unsupported, keep native".
bool MaybeAddAxisReshape(onnx::GraphProto& graph, GraphIndex& index,
                         google::protobuf::RepeatedPtrField<onnx::NodeProto>& nodes, const onnx::NodeProto& source_node,
                         const std::string& input_name, const std::string& base_name, std::optional<int64_t> input_rank,
                         std::optional<int64_t> axis_attr, const onnx::TensorProto& parameter_tensor,
                         int64_t block_size, size_t unique_id, std::string& broadcast_name)
{
    broadcast_name = input_name;
    const size_t parameter_rank = static_cast<size_t>(parameter_tensor.dims_size());
    const size_t parameter_elements = GetTensorElementCount(parameter_tensor);
    if (parameter_rank == 0 || parameter_elements == 1)
    {
        return true;
    }

    if (!input_rank)
    {
        return false;
    }

    if (block_size > 1)
    {
        std::vector<int64_t> input_shape;
        if (!index.TryGetTensorShape(source_node.input(0), input_shape))
        {
            return false;
        }
        const std::string expanded_name = MakeLoweredName(base_name, "expanded", unique_id);
        if (AddExpandedInitializer(graph, index, expanded_name, parameter_tensor, input_shape, axis_attr, block_size) ==
            nullptr)
        {
            return false;
        }
        broadcast_name = expanded_name;
        return true;
    }

    if (parameter_rank == 1)
    {
        if (*input_rank <= 1)
        {
            return true;
        }

        int64_t axis = axis_attr.value_or(1);
        if (axis < 0)
        {
            axis += *input_rank;
        }
        if (axis < 0 || axis >= *input_rank)
        {
            return false;
        }
        if (axis == *input_rank - 1)
        {
            return true;
        }

        std::vector<int64_t> reshape_dims(static_cast<size_t>(*input_rank), 1);
        reshape_dims[static_cast<size_t>(axis)] = parameter_tensor.dims(0);
        const std::string shape_name = MakeLoweredName(base_name, "reshape_shape", unique_id);
        const std::string reshaped_name = MakeLoweredName(base_name, "reshaped", unique_id);
        AddInt64VectorInitializer(graph, index, shape_name, reshape_dims);
        AppendNode(nodes, source_node, "Reshape", MakeLoweredName(base_name, "reshape", unique_id),
                   {input_name, shape_name}, {reshaped_name});
        broadcast_name = reshaped_name;
        return true;
    }

    return parameter_rank == static_cast<size_t>(*input_rank);
}

// Constant-weight DQ folding.
//
// When the DQ input is itself an initializer, we precompute y = (x - zp) *
// scale offline and emit a float initializer carrying those bytes. We then
// append a trivial Identity(folded_initializer -> original_output_name) so
// downstream consumers don't need to be rewired.
//
// Why keep the Identity instead of renaming the initializer:
//   1. The original output name may appear elsewhere in the graph (graph
//      outputs, other consumers); renaming would require a full sweep.
//   2. The Identity carries the original ORT node's doc_string, so the
//      provider's proto->ORT remap can still credit this ORT node. If TRT
//      prunes the Identity as trivial, we also record the fold in
//      LoweredQdqInfo::folded_constant_nodes so the provider can reattach
//      the original ORT node to whatever parser subgraph still references
//      the folded tensor (see the reattachment loop in GetSupportedList).
bool TryFoldConstantDequantizeLinear(onnx::GraphProto& graph, GraphIndex& index,
                                     google::protobuf::RepeatedPtrField<onnx::NodeProto>& nodes,
                                     const onnx::NodeProto& node, const onnx::TensorProto& input_tensor,
                                     const onnx::TensorProto& scale_tensor, const onnx::TensorProto* zero_point_tensor,
                                     int32_t arithmetic_type, std::optional<int64_t> axis_attr, int64_t block_size,
                                     size_t& unique_id, LoweredQdqInfo& lowered_qdq_info)
{
    // Only fp32/fp64 folding is safe; fp16/bf16 folding would lose precision
    // relative to what the runtime path would have produced.
    if (arithmetic_type != onnx::TensorProto_DataType_FLOAT && arithmetic_type != onnx::TensorProto_DataType_DOUBLE)
    {
        return false;
    }

    std::vector<int64_t> input_values;
    std::vector<double> scale_values;
    std::vector<int64_t> zero_point_values;
    if (!TryReadIntegerTensorData(input_tensor, input_values) || !TryReadFloatingTensorData(scale_tensor, scale_values))
    {
        return false;
    }
    if (zero_point_tensor != nullptr && !TryReadIntegerTensorData(*zero_point_tensor, zero_point_values))
    {
        return false;
    }

    const size_t element_count = input_values.size();
    const std::string folded_output_name = MakeLoweredName(node.output(0), "folded_constant", ++unique_id);
    auto* tensor_proto = graph.add_initializer();
    tensor_proto->set_name(folded_output_name);
    tensor_proto->set_data_type(arithmetic_type);
    tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    for (const auto dim : input_tensor.dims())
    {
        tensor_proto->add_dims(dim);
    }

    if (arithmetic_type == onnx::TensorProto_DataType_FLOAT)
    {
        std::vector<float> output_values(element_count);
        for (size_t i = 0; i < element_count; ++i)
        {
            size_t scale_index = 0;
            size_t zero_index = 0;
            if (!TryResolveBroadcastIndex(i, input_tensor, scale_tensor, axis_attr, block_size, scale_index))
            {
                return false;
            }
            if (zero_point_tensor != nullptr &&
                !TryResolveBroadcastIndex(i, input_tensor, *zero_point_tensor, axis_attr, block_size, zero_index))
            {
                return false;
            }
            const double zero_point =
                zero_point_tensor != nullptr ? static_cast<double>(zero_point_values[zero_index]) : 0.0;
            output_values[i] =
                static_cast<float>((static_cast<double>(input_values[i]) - zero_point) * scale_values[scale_index]);
        }
        tensor_proto->set_raw_data(output_values.data(), output_values.size() * sizeof(float));
    }
    else
    {
        std::vector<double> output_values(element_count);
        for (size_t i = 0; i < element_count; ++i)
        {
            size_t scale_index = 0;
            size_t zero_index = 0;
            if (!TryResolveBroadcastIndex(i, input_tensor, scale_tensor, axis_attr, block_size, scale_index))
            {
                return false;
            }
            if (zero_point_tensor != nullptr &&
                !TryResolveBroadcastIndex(i, input_tensor, *zero_point_tensor, axis_attr, block_size, zero_index))
            {
                return false;
            }
            const double zero_point =
                zero_point_tensor != nullptr ? static_cast<double>(zero_point_values[zero_index]) : 0.0;
            output_values[i] = (static_cast<double>(input_values[i]) - zero_point) * scale_values[scale_index];
        }
        tensor_proto->set_raw_data(output_values.data(), output_values.size() * sizeof(double));
    }
    index.NoteInitializer(*tensor_proto);

    AppendNode(nodes, node, "Identity", MakeLoweredName(node.output(0), "folded_identity", ++unique_id),
               {folded_output_name}, {node.output(0)});
    if (auto node_id = TryParseNodeId(node.doc_string()))
    {
        lowered_qdq_info.folded_constant_nodes.push_back({*node_id, node.output(0)});
    }
    return true;
}

// Policy: the lowered arithmetic carries the scale's own floating dtype.
// Rationale:
//   * The final Mul/Div requires same-type operands as the scale, so any
//     other choice forces an extra Cast with no numerical benefit.
//   * Model authors pick scale dtype deliberately (fp16 scales for fp16
//     pipelines, fp32 otherwise); respect that choice.
// Exceptions are handled in GetQuantizeArithmeticType (int16/uint16 outputs
// need fp32 even when the scale is fp16/bf16).
std::optional<int32_t> GetArithmeticTypeForScale(const onnx::TensorProto& scale_tensor)
{
    switch (scale_tensor.data_type())
    {
    case onnx::TensorProto_DataType_FLOAT:
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_BFLOAT16:
    case onnx::TensorProto_DataType_DOUBLE:
        return scale_tensor.data_type();
    default:
        return std::nullopt;
    }
}

// "Asymmetric" is only asymmetric if the zp tensor exists AND carries at
// least one non-zero element. An absent zp (no input or empty-string input)
// and a zp initializer full of zeros are both treated as symmetric (zp=0).
bool HasExplicitNonZeroZeroPoint(const onnx::TensorProto* zero_point_tensor)
{
    return zero_point_tensor != nullptr && TensorHasAnyNonZeroValue(*zero_point_tensor);
}

// Native ONNX integer zero_point types (int8/uint8/int16/uint16) cannot be
// Sub'd from a widened integer (or Cast'd to a float) directly in a way that
// both TRT and ORT accept as input dtypes to Sub/Cast. Promote to int32 (or
// int64 when the arithmetic is double-precision) so the lowered Sub/Add is
// always in a widely-supported dtype.
bool NeedsPromotedZeroPoint(int32_t zero_point_type)
{
    return zero_point_type != onnx::TensorProto_DataType_INT32 && zero_point_type != onnx::TensorProto_DataType_INT64;
}

// Choose the promoted zero_point container type:
//   * INT64 zp stays INT64 (already widest).
//   * UINT32 zp -> INT64 so the signed Sub cannot overflow at the high end.
//   * INT32 zp stays INT32.
//   * Low-bit integer zp -> INT32 normally, INT64 only when arithmetic is
//     double (to keep the Cast chain precision-consistent).
int32_t GetPromotedZeroPointType(int32_t zero_point_type, int32_t arithmetic_type)
{
    if (zero_point_type == onnx::TensorProto_DataType_INT64)
    {
        return onnx::TensorProto_DataType_INT64;
    }
    if (zero_point_type == onnx::TensorProto_DataType_UINT32)
    {
        return onnx::TensorProto_DataType_INT64;
    }
    if (zero_point_type == onnx::TensorProto_DataType_INT32)
    {
        return onnx::TensorProto_DataType_INT32;
    }
    if (zero_point_type == onnx::TensorProto_DataType_UINT16 || zero_point_type == onnx::TensorProto_DataType_UINT8 ||
        zero_point_type == onnx::TensorProto_DataType_INT16 || zero_point_type == onnx::TensorProto_DataType_INT8)
    {
        return arithmetic_type == onnx::TensorProto_DataType_DOUBLE ? onnx::TensorProto_DataType_INT64
                                                                    : onnx::TensorProto_DataType_INT32;
    }
    return zero_point_type;
}

// For int32/uint32 DequantizeLinear, do the (x - zp) subtraction in INT64
// *before* casting to float. Converting uint32 values straight to fp32 loses
// up to 8 bits of precision and makes the subsequent Sub inexact; widening to
// int64 first preserves the exact integer difference, and only then we cast
// to the scale's floating dtype for the final Mul.
//
// Low-bit integers (int8/uint8/int16/uint16) fit exactly in fp32, so they
// skip this stage and cast directly to arithmetic_type in one step.
std::optional<int32_t> GetDequantizeIntegerMathType(int32_t input_type)
{
    switch (input_type)
    {
    case onnx::TensorProto_DataType_INT32:
    case onnx::TensorProto_DataType_UINT32:
        return onnx::TensorProto_DataType_INT64;
    default:
        return std::nullopt;
    }
}

// Choose the arithmetic dtype for a lowered QuantizeLinear.
//
// Default: inherit the scale's dtype (same rationale as GetArithmeticTypeForScale).
//
// Special case: int16/uint16 output with fp16/bf16 scale MUST be promoted
// to fp32. The 16-bit integer clamp bounds are +/-32768, far outside the
// exactly-representable range of fp16 (max ~65504 but with spacing >32 near
// the top) and bf16 (spacing 256 near the top). Doing round(x/scale)+zp and
// the final Max/Min in fp16/bf16 would round qmax to inf or snap rounded
// values to the wrong integer, producing silently-wrong quantization.
std::optional<int32_t> GetQuantizeArithmeticType(const onnx::TensorProto& scale_tensor, int32_t output_type)
{
    auto arithmetic_type = GetArithmeticTypeForScale(scale_tensor);
    if (!arithmetic_type)
    {
        return std::nullopt;
    }

    if ((output_type == onnx::TensorProto_DataType_INT16 || output_type == onnx::TensorProto_DataType_UINT16) &&
        (*arithmetic_type == onnx::TensorProto_DataType_FLOAT16 ||
         *arithmetic_type == onnx::TensorProto_DataType_BFLOAT16))
    {
        return onnx::TensorProto_DataType_FLOAT;
    }

    return arithmetic_type;
}

struct QdqLoweringDecision
{
    bool should_lower = false;
    std::string reason;
};

bool ShouldLowerQuantizeLinear(const onnx::NodeProto& node, const onnx::TensorProto& scale_tensor,
                               const onnx::TensorProto* zero_point_tensor, int32_t output_type);

std::string NodeDisplayName(const onnx::NodeProto& node)
{
    if (!node.name().empty())
    {
        return node.name();
    }
    if (node.output_size() > 0 && !node.output(0).empty())
    {
        return node.output(0);
    }
    return node.op_type();
}

std::string TensorDataTypeToString(int32_t data_type)
{
    switch (data_type)
    {
    case onnx::TensorProto_DataType_INT8:
        return "int8";
    case onnx::TensorProto_DataType_UINT8:
        return "uint8";
    case onnx::TensorProto_DataType_INT16:
        return "int16";
    case onnx::TensorProto_DataType_UINT16:
        return "uint16";
    case onnx::TensorProto_DataType_INT32:
        return "int32";
    case onnx::TensorProto_DataType_UINT32:
        return "uint32";
    case onnx::TensorProto_DataType_FLOAT16:
        return "float16";
    case onnx::TensorProto_DataType_BFLOAT16:
        return "bfloat16";
    case onnx::TensorProto_DataType_FLOAT:
        return "float32";
    case onnx::TensorProto_DataType_DOUBLE:
        return "float64";
    default:
        return std::to_string(data_type);
    }
}

// Ops that TRT treats as Q/DQ fusion anchors. The harmonization rule only
// kicks in around these, because those are the ops where TRT's Q/DQ fuser
// has a meaningful "either fully-native or fully-lowered" preference. For
// other ops (e.g. Add, Concat, Transpose) leaving a mixed neighborhood is
// harmless -- no fusion opportunity is lost.
bool IsComputeAnchorOp(const onnx::NodeProto& node)
{
    return node.op_type() == "Gemm" || node.op_type() == "MatMul" || node.op_type() == "Conv" ||
           node.op_type() == "ConvTranspose";
}

// Base (unconditional) DQ lowering rules. These fire regardless of whether
// the DQ is around a compute anchor; harmonization is layered on top.
//
//   DQ-1: explicit non-zero zero_point  -> TRT native DQ is symmetric-only.
//   DQ-2: int32/uint32 input            -> TRT native DQ does not accept
//                                          those integer widths.
//
// `reason` is populated so the harmonization layer and diagnostics can
// attribute decisions back to a specific rule.
bool BaseMustLowerDequantize(const onnx::TensorProto* zero_point_tensor, int32_t input_type, std::string& reason)
{
    if (HasExplicitNonZeroZeroPoint(zero_point_tensor))
    {
        reason = "explicit non-zero zeroPoint";
        return true;
    }
    if (IsTensorRtUnsupportedQdqIntegerType(input_type))
    {
        reason = "native TensorRT DequantizeLinear does not support input type " + TensorDataTypeToString(input_type);
        return true;
    }
    return false;
}

QdqLoweringDecision EvaluateBaseDequantizeLowering(const onnx::NodeProto& node, const GraphIndex& index)
{
    if (node.input_size() < 2)
    {
        return {};
    }

    const auto* scale_tensor = index.FindInitializer(node.input(1));
    if (scale_tensor == nullptr || !IsPositiveScaleTensor(*scale_tensor))
    {
        return {};
    }

    const auto input_type = index.FindTensorElementType(node.input(0));
    if (!input_type)
    {
        return {};
    }

    const auto* zero_point_tensor =
        (node.input_size() > 2 && !node.input(2).empty()) ? index.FindInitializer(node.input(2)) : nullptr;

    QdqLoweringDecision decision;
    decision.should_lower = BaseMustLowerDequantize(zero_point_tensor, *input_type, decision.reason);
    return decision;
}

QdqLoweringDecision EvaluateBaseQuantizeLowering(const onnx::NodeProto& node, const GraphIndex& index)
{
    if (node.input_size() < 2)
    {
        return {};
    }

    const auto* scale_tensor = index.FindInitializer(node.input(1));
    if (scale_tensor == nullptr || !IsPositiveScaleTensor(*scale_tensor))
    {
        return {};
    }

    const auto* zero_point_tensor =
        (node.input_size() > 2 && !node.input(2).empty()) ? index.FindInitializer(node.input(2)) : nullptr;

    int32_t output_type =
        zero_point_tensor != nullptr ? zero_point_tensor->data_type() : onnx::TensorProto_DataType_UINT8;
    if (zero_point_tensor == nullptr)
    {
        if (auto output_dtype = FindIntAttribute(node, "output_dtype"))
        {
            output_type = static_cast<int32_t>(*output_dtype);
        }
    }

    QdqLoweringDecision decision;
    if (ShouldLowerQuantizeLinear(node, *scale_tensor, zero_point_tensor, output_type))
    {
        decision.should_lower = true;
        decision.reason = "explicit non-zero zeroPoint";
    }
    return decision;
}

// A "symmetric constant DQ" is the shape TRT wants to keep native:
//   * zp is absent / all-zero,
//   * input type is one TRT accepts (no int32/uint32),
//   * the DQ input is a compile-time initializer (i.e. a weight, not an
//     activation -- graph inputs are explicitly excluded).
// Only candidates that pass this filter are eligible for the harmonization
// rule below. Everything else is already covered by the base rules.
bool IsSymmetricConstantDequantizeCandidate(const onnx::NodeProto& node, const GraphIndex& index,
                                            const onnx::TensorProto* zero_point_tensor, int32_t input_type)
{
    if (node.input_size() < 1 || node.input(0).empty())
    {
        return false;
    }
    if (HasExplicitNonZeroZeroPoint(zero_point_tensor) || IsTensorRtUnsupportedQdqIntegerType(input_type))
    {
        return false;
    }
    if (index.IsGraphInput(node.input(0)))
    {
        return false;
    }
    return index.FindInitializer(node.input(0)) != nullptr;
}

const onnx::NodeProto* FindSingleConsumerAnchor(const onnx::NodeProto& node, const GraphIndex& index)
{
    // Symmetric constant DQ stays native by default. We only harmonize it when a
    // single downstream compute anchor would otherwise mix native and lowered Q/DQ
    // paths in the same local neighborhood.
    if (node.output_size() == 0 || node.output(0).empty())
    {
        return nullptr;
    }

    const auto* consumer_indices = index.FindConsumerNodeIndices(node.output(0));
    if (consumer_indices == nullptr || consumer_indices->size() != 1)
    {
        return nullptr;
    }

    const auto* anchor = index.GetNode((*consumer_indices)[0]);
    if (anchor == nullptr || !IsComputeAnchorOp(*anchor))
    {
        return nullptr;
    }
    return anchor;
}

// Harmonization check around a compute anchor.
//
// Problem: a Gemm/MatMul/Conv typically sits at the center of a Q/DQ cluster:
//
//    weight_const --(sym DQ int8)----\
//                                      Gemm --(asym Q uint8)--> ...
//    input_act    --(asym DQ uint8)--/
//
// If we keep the symmetric weight DQ native but lower the asymmetric
// activation DQ and output Q, TRT's fuser is left with a mixed neighborhood
// (half native Q/DQ, half plain arithmetic) and cannot apply its integer
// Gemm fusion. Perf collapses and numerics become harder to reason about.
//
// Rule: if ANY sibling Q/DQ attached to the same anchor is already forced
// to lower by the base rules, the symmetric constant DQ is lowered too so
// the whole neighborhood is uniformly lowered arithmetic.
//
// We scan both sides of the anchor:
//   * Other inputs' producers: other DQ/Q feeding the anchor.
//   * Each output's consumers: Q ops that re-quantize the anchor's result.
//
// Importantly, we call the *base* evaluators on siblings, not the full
// harmonization evaluator, to avoid mutual recursion between neighbors.
bool AnchorHasForcedLoweringSibling(const onnx::NodeProto& candidate_node, const onnx::NodeProto& anchor,
                                    const GraphIndex& index, std::string& reason)
{
    for (const auto& anchor_input : anchor.input())
    {
        if (anchor_input.empty())
        {
            continue;
        }

        const auto* producer = index.FindProducer(anchor_input);
        if (producer == nullptr || producer == &candidate_node)
        {
            continue;
        }

        if (producer->op_type() == "DequantizeLinear")
        {
            auto sibling_decision = EvaluateBaseDequantizeLowering(*producer, index);
            if (sibling_decision.should_lower)
            {
                reason = "anchor input producer '" + NodeDisplayName(*producer) + "' already lowers (" +
                         sibling_decision.reason + ")";
                return true;
            }
        }
        else if (producer->op_type() == "QuantizeLinear")
        {
            auto sibling_decision = EvaluateBaseQuantizeLowering(*producer, index);
            if (sibling_decision.should_lower)
            {
                reason = "anchor input producer '" + NodeDisplayName(*producer) + "' already lowers (" +
                         sibling_decision.reason + ")";
                return true;
            }
        }
    }

    for (const auto& anchor_output : anchor.output())
    {
        if (anchor_output.empty())
        {
            continue;
        }

        const auto* consumer_indices = index.FindConsumerNodeIndices(anchor_output);
        if (consumer_indices == nullptr)
        {
            continue;
        }

        for (const auto consumer_index : *consumer_indices)
        {
            const auto* consumer = index.GetNode(consumer_index);
            if (consumer == nullptr || consumer == &candidate_node)
            {
                continue;
            }

            if (consumer->op_type() == "QuantizeLinear")
            {
                auto sibling_decision = EvaluateBaseQuantizeLowering(*consumer, index);
                if (sibling_decision.should_lower)
                {
                    reason = "anchor output consumer '" + NodeDisplayName(*consumer) + "' already lowers (" +
                             sibling_decision.reason + ")";
                    return true;
                }
            }
        }
    }

    return false;
}

// DequantizeLinear lowering has two stages:
// 1. Lower asymmetric or TRT-unsupported integer types unconditionally.
// 2. Keep symmetric constant weights native unless the surrounding compute anchor
//    is already forced into lowered math by a sibling Q/DQ path. That avoids
//    leaving a mixed native/lowered neighborhood around one Gemm/MatMul/Conv op.
QdqLoweringDecision EvaluateDequantizeLowering(const onnx::NodeProto& node, const GraphIndex& index,
                                               const onnx::TensorProto* zero_point_tensor, int32_t input_type)
{
    auto base_decision = EvaluateBaseDequantizeLowering(node, index);
    if (base_decision.should_lower)
    {
        return base_decision;
    }

    if (!IsSymmetricConstantDequantizeCandidate(node, index, zero_point_tensor, input_type))
    {
        base_decision.reason = "keep native symmetric DequantizeLinear";
        return base_decision;
    }

    const auto* anchor = FindSingleConsumerAnchor(node, index);
    if (anchor == nullptr)
    {
        base_decision.reason =
            "keep native symmetric constant DequantizeLinear because there is no single compute anchor";
        return base_decision;
    }

    std::string harmonization_reason;
    if (AnchorHasForcedLoweringSibling(node, *anchor, index, harmonization_reason))
    {
        return {true, "harmonize symmetric constant DequantizeLinear around anchor '" + NodeDisplayName(*anchor) +
                          "' because " + harmonization_reason};
    }

    base_decision.reason = "keep native symmetric constant DequantizeLinear because anchor '" +
                           NodeDisplayName(*anchor) + "' has no sibling Q/DQ path that must lower";
    return base_decision;
}

// QuantizeLinear lowering policy.
//
//   Q-1 (keep native): zero_point is absent or all zeros. TRT's symmetric Q
//       path is the right fit; don't disturb it.
//   Q-2 (never lower): output dtype is int32/uint32. A fp32/fp64 rendering
//       of round(x/scale)+zp is NOT accurate over a 32-bit integer range
//       (ULP gap at the top exceeds 1), so we'd rather report the node
//       unsupported and let ORT fall back to CPU than silently miscompile.
//   Q-3 (lower):       asymmetric quantization into int8/uint8/int16/uint16.
//       These ranges are modeled exactly by our arithmetic (with the fp32
//       promotion from GetQuantizeArithmeticType for the 16-bit cases).
bool ShouldLowerQuantizeLinear(const onnx::NodeProto& node, const onnx::TensorProto& scale_tensor,
                               const onnx::TensorProto* zero_point_tensor, int32_t output_type)
{
    (void)node;
    (void)scale_tensor;
    if (output_type == onnx::TensorProto_DataType_INT32 || output_type == onnx::TensorProto_DataType_UINT32)
    {
        return false;
    }
    if (!HasExplicitNonZeroZeroPoint(zero_point_tensor))
    {
        return false;
    }
    return output_type == onnx::TensorProto_DataType_INT8 || output_type == onnx::TensorProto_DataType_UINT8 ||
           output_type == onnx::TensorProto_DataType_INT16 || output_type == onnx::TensorProto_DataType_UINT16;
}
}  // namespace

LoweredQdqInfo RunQdqLoweringForTensorRt(onnx::ModelProto& model_proto)
{
    LoweredQdqInfo lowered_qdq_info;
    auto& graph = *model_proto.mutable_graph();
    GraphIndex index(graph);
    google::protobuf::RepeatedPtrField<onnx::NodeProto> lowered_nodes;
    size_t unique_id = 0;

    // Rewrite only the Q/DQ patterns that TRT-RTX cannot consume natively.
    //
    // For DQ we materialize the WebNN / ONNX math explicitly:
    //   y = (x - zero_point) * scale
    //
    //   before: x -- DequantizeLinear --> y
    //   after : x -- Cast -- Sub(zp) -- Cast? -- Mul(scale) --> y
    //
    // Constant DQ can fold one step further to:
    //   constant_int --(fold)--> constant_float -- Identity --> y
    //
    // For Q we lower the asymmetric form to the corresponding arithmetic:
    //   y = cast(clamp(round(x / scale) + zero_point, qmin, qmax))
    //
    // We leave native symmetric paths in place unless harmonization is needed to
    // avoid mixing lowered and native Q/DQ around the same compute anchor.
    for (const auto& node : graph.node())
    {
        // Preflight gates. Any failure here means "this node's shape is
        // either not Q/DQ or is outside what the rewrite layer can safely
        // handle"; we copy it through verbatim and let TRT / ORT decide.
        // These gates must stay in sync with the ones inside
        // Evaluate{Base,}DequantizeLowering / EvaluateBaseQuantizeLowering,
        // otherwise the policy layer and the rewrite layer could disagree.

        if ((node.op_type() != "DequantizeLinear" && node.op_type() != "QuantizeLinear") || node.input_size() < 2)
        {
            *lowered_nodes.Add() = node;
            continue;
        }

        // Gate 1: scale must be a readable, spec-compliant constant
        // (positive and finite). See IsPositiveScaleTensor for rationale.
        const auto* scale_tensor = index.FindInitializer(node.input(1));
        if (scale_tensor == nullptr || !IsPositiveScaleTensor(*scale_tensor))
        {
            *lowered_nodes.Add() = node;
            continue;
        }

        // zero_point is optional (symmetric case). nullptr here means "treat
        // as zp=0"; it is NOT an error, so no bail-out on this line.
        const auto* zero_point_tensor =
            (node.input_size() > 2 && !node.input(2).empty()) ? index.FindInitializer(node.input(2)) : nullptr;

        // Gate 2: input dtype must be known. Without it we can't pick
        // integer-math widening (int32/uint32 -> int64) nor apply DQ-2.
        std::optional<int32_t> input_type = index.FindTensorElementType(node.input(0));
        if (!input_type)
        {
            *lowered_nodes.Add() = node;
            continue;
        }

        // Gate 3: scale dtype must be a supported floating type. This also
        // anchors the dtype of every arithmetic op we are about to emit.
        const auto arithmetic_type = GetArithmeticTypeForScale(*scale_tensor);
        if (!arithmetic_type)
        {
            *lowered_nodes.Add() = node;
            continue;
        }

        const auto axis_attr = FindIntAttribute(node, "axis");
        const int64_t block_size = FindIntAttribute(node, "block_size").value_or(0);
        const std::string node_base = !node.name().empty() ? node.name() : node.output(0);

        if (node.op_type() == "DequantizeLinear")
        {
            const auto dequantize_decision = EvaluateDequantizeLowering(node, index, zero_point_tensor, *input_type);
            if (!dequantize_decision.should_lower)
            {
                *lowered_nodes.Add() = node;
                continue;
            }

            if (const auto* input_initializer = index.FindInitializer(node.input(0)))
            {
                // Constant DQ is folded to a float initializer plus a trivial Identity so the
                // lowered graph still exposes the original tensor name to downstream nodes.
                if (TryFoldConstantDequantizeLinear(graph, index, lowered_nodes, node, *input_initializer,
                                                    *scale_tensor, zero_point_tensor, *arithmetic_type, axis_attr,
                                                    block_size, unique_id, lowered_qdq_info))
                {
                    ++lowered_qdq_info.lowered_node_count;
                    continue;
                }
            }

            // Lower DQ into standard ONNX arithmetic so TRT sees only supported
            // float/int ops:
            //
            //   x_q -- DQ(scale, zp) --> y_f
            //
            // becomes
            //
            //   x_q -- Cast(input_math_type)
            //       -- Sub(Cast(zp))
            //       -- Cast(arithmetic_type)?
            //       -- Mul(scale)
            //       --> y_f
            //
            // Integer-first subtraction preserves exact differences for int32/uint32
            // before we finally cast to the floating arithmetic type used by scale.
            std::string lowered_scale_name = node.input(1);
            std::string lowered_zero_name = zero_point_tensor != nullptr ? node.input(2) : std::string();

            ++unique_id;
            if (!MaybeAddAxisReshape(graph, index, lowered_nodes, node, lowered_scale_name,
                                     MakeLoweredName(node_base, "scale", unique_id),
                                     index.FindTensorRank(node.input(0)), axis_attr, *scale_tensor, block_size,
                                     unique_id, lowered_scale_name))
            {
                *lowered_nodes.Add() = node;
                continue;
            }

            if (zero_point_tensor != nullptr)
            {
                if (NeedsPromotedZeroPoint(zero_point_tensor->data_type()))
                {
                    lowered_zero_name = MakeLoweredName(node_base, "zero_promoted", ++unique_id);
                    if (AddPromotedIntegerInitializer(
                            graph, index, lowered_zero_name, *zero_point_tensor,
                            GetPromotedZeroPointType(zero_point_tensor->data_type(), *arithmetic_type)) == nullptr)
                    {
                        *lowered_nodes.Add() = node;
                        continue;
                    }
                }
                ++unique_id;
                if (!MaybeAddAxisReshape(
                        graph, index, lowered_nodes, node, lowered_zero_name,
                        MakeLoweredName(node_base, "zero", unique_id), index.FindTensorRank(node.input(0)), axis_attr,
                        *index.FindInitializer(lowered_zero_name), block_size, unique_id, lowered_zero_name))
                {
                    *lowered_nodes.Add() = node;
                    continue;
                }
            }

            const auto integer_math_type = GetDequantizeIntegerMathType(*input_type);
            const int32_t input_math_type = integer_math_type.value_or(*arithmetic_type);

            const std::string cast_input_name = MakeLoweredName(node_base, "input_cast", ++unique_id);
            auto* cast_input =
                AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_input", unique_id),
                           {node.input(0)}, {cast_input_name});
            auto* cast_input_attr = cast_input->add_attribute();
            cast_input_attr->set_name("to");
            cast_input_attr->set_type(onnx::AttributeProto_AttributeType_INT);
            cast_input_attr->set_i(input_math_type);

            std::string dequantized_name = cast_input_name;
            if (!lowered_zero_name.empty())
            {
                const std::string cast_zero_name = MakeLoweredName(node_base, "zero_cast", ++unique_id);
                auto* cast_zero =
                    AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_zero", unique_id),
                               {lowered_zero_name}, {cast_zero_name});
                auto* cast_zero_attr = cast_zero->add_attribute();
                cast_zero_attr->set_name("to");
                cast_zero_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                cast_zero_attr->set_i(input_math_type);

                dequantized_name = MakeLoweredName(node_base, "shifted", ++unique_id);
                AppendNode(lowered_nodes, node, "Sub", MakeLoweredName(node_base, "sub", unique_id),
                           {cast_input_name, cast_zero_name}, {dequantized_name});
            }

            if (input_math_type != *arithmetic_type)
            {
                const std::string cast_diff_name = MakeLoweredName(node_base, "diff_cast", ++unique_id);
                auto* cast_diff =
                    AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_diff", unique_id),
                               {dequantized_name}, {cast_diff_name});
                auto* cast_diff_attr = cast_diff->add_attribute();
                cast_diff_attr->set_name("to");
                cast_diff_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                cast_diff_attr->set_i(*arithmetic_type);
                dequantized_name = cast_diff_name;
            }

            AppendNode(lowered_nodes, node, "Mul", MakeLoweredName(node_base, "mul", ++unique_id),
                       {dequantized_name, lowered_scale_name}, {node.output(0)});
            ++lowered_qdq_info.lowered_node_count;
            continue;
        }

        if (node.op_type() == "QuantizeLinear")
        {
            int32_t output_type =
                zero_point_tensor != nullptr ? zero_point_tensor->data_type() : onnx::TensorProto_DataType_UINT8;
            if (zero_point_tensor == nullptr)
            {
                if (auto output_dtype = FindIntAttribute(node, "output_dtype"))
                {
                    output_type = static_cast<int32_t>(*output_dtype);
                }
            }

            if (ShouldLowerQuantizeLinear(node, *scale_tensor, zero_point_tensor, output_type))
            {
                // Lower asymmetric Q into explicit arithmetic:
                //
                //   x_f -- QuantizeLinear(scale, zp) --> y_q
                //
                // becomes
                //
                //   x_f -- Cast
                //       -- Div(scale)
                //       -- Add(zp)
                //       -- Round
                //       -- Clamp(qmin, qmax)
                //       -- Cast(output_type)
                //       --> y_q
                //
                // This mirrors the spec math while keeping TRT away from unsupported
                // native asymmetric Q paths.
                const auto quantize_arithmetic_type = GetQuantizeArithmeticType(*scale_tensor, output_type);
                if (!quantize_arithmetic_type)
                {
                    *lowered_nodes.Add() = node;
                    continue;
                }

                int64_t qmin = 0;
                int64_t qmax = 0;
                if (!TryGetQuantizedValueRange(output_type, qmin, qmax))
                {
                    *lowered_nodes.Add() = node;
                    continue;
                }

                std::string lowered_scale_name = node.input(1);
                std::string lowered_zero_name = zero_point_tensor != nullptr ? node.input(2) : std::string();
                ++unique_id;
                if (!MaybeAddAxisReshape(graph, index, lowered_nodes, node, lowered_scale_name,
                                         MakeLoweredName(node_base, "scale", unique_id),
                                         index.FindTensorRank(node.input(0)), axis_attr, *scale_tensor, block_size,
                                         unique_id, lowered_scale_name))
                {
                    *lowered_nodes.Add() = node;
                    continue;
                }

                if (*quantize_arithmetic_type != scale_tensor->data_type())
                {
                    const std::string cast_scale_name = MakeLoweredName(node_base, "scale_cast", ++unique_id);
                    auto* cast_scale =
                        AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_scale", unique_id),
                                   {lowered_scale_name}, {cast_scale_name});
                    auto* cast_scale_attr = cast_scale->add_attribute();
                    cast_scale_attr->set_name("to");
                    cast_scale_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                    cast_scale_attr->set_i(*quantize_arithmetic_type);
                    lowered_scale_name = cast_scale_name;
                }

                if (zero_point_tensor != nullptr)
                {
                    if (NeedsPromotedZeroPoint(zero_point_tensor->data_type()))
                    {
                        lowered_zero_name = MakeLoweredName(node_base, "zero_promoted", ++unique_id);
                        if (AddPromotedIntegerInitializer(
                                graph, index, lowered_zero_name, *zero_point_tensor,
                                GetPromotedZeroPointType(zero_point_tensor->data_type(), *quantize_arithmetic_type)) ==
                            nullptr)
                        {
                            *lowered_nodes.Add() = node;
                            continue;
                        }
                    }
                    ++unique_id;
                    if (!MaybeAddAxisReshape(graph, index, lowered_nodes, node, lowered_zero_name,
                                             MakeLoweredName(node_base, "zero", unique_id),
                                             index.FindTensorRank(node.input(0)), axis_attr,
                                             *index.FindInitializer(lowered_zero_name), block_size, unique_id,
                                             lowered_zero_name))
                    {
                        *lowered_nodes.Add() = node;
                        continue;
                    }
                }

                const std::string cast_input_name = MakeLoweredName(node_base, "input_cast", ++unique_id);
                auto* cast_input =
                    AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_input", unique_id),
                               {node.input(0)}, {cast_input_name});
                auto* cast_input_attr = cast_input->add_attribute();
                cast_input_attr->set_name("to");
                cast_input_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                cast_input_attr->set_i(*quantize_arithmetic_type);

                const std::string div_name = MakeLoweredName(node_base, "div", ++unique_id);
                AppendNode(lowered_nodes, node, "Div", MakeLoweredName(node_base, "div_node", unique_id),
                           {cast_input_name, lowered_scale_name}, {div_name});

                std::string shifted_name = div_name;
                if (!lowered_zero_name.empty())
                {
                    const std::string cast_zero_name = MakeLoweredName(node_base, "zero_cast", ++unique_id);
                    auto* cast_zero =
                        AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_zero", unique_id),
                                   {lowered_zero_name}, {cast_zero_name});
                    auto* cast_zero_attr = cast_zero->add_attribute();
                    cast_zero_attr->set_name("to");
                    cast_zero_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                    cast_zero_attr->set_i(*quantize_arithmetic_type);

                    shifted_name = MakeLoweredName(node_base, "shifted", ++unique_id);
                    AppendNode(lowered_nodes, node, "Add", MakeLoweredName(node_base, "add", unique_id),
                               {div_name, cast_zero_name}, {shifted_name});
                }

                const std::string rounded_name = MakeLoweredName(node_base, "rounded", ++unique_id);
                AppendNode(lowered_nodes, node, "Round", MakeLoweredName(node_base, "round", unique_id), {shifted_name},
                           {rounded_name});

                const int32_t qrange_constant_type = output_type == onnx::TensorProto_DataType_UINT32
                                                         ? onnx::TensorProto_DataType_INT64
                                                         : onnx::TensorProto_DataType_INT32;
                const std::string qmin_name = MakeLoweredName(node_base, "qmin", ++unique_id);
                const std::string qmax_name = MakeLoweredName(node_base, "qmax", ++unique_id);
                AddScalarInitializer(graph, index, qmin_name, qrange_constant_type, qmin);
                AddScalarInitializer(graph, index, qmax_name, qrange_constant_type, qmax);

                const std::string qmin_cast_name = MakeLoweredName(node_base, "qmin_cast", ++unique_id);
                const std::string qmax_cast_name = MakeLoweredName(node_base, "qmax_cast", ++unique_id);
                auto* qmin_cast =
                    AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_qmin", unique_id),
                               {qmin_name}, {qmin_cast_name});
                auto* qmin_cast_attr = qmin_cast->add_attribute();
                qmin_cast_attr->set_name("to");
                qmin_cast_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                qmin_cast_attr->set_i(*quantize_arithmetic_type);

                auto* qmax_cast =
                    AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_qmax", ++unique_id),
                               {qmax_name}, {qmax_cast_name});
                auto* qmax_cast_attr = qmax_cast->add_attribute();
                qmax_cast_attr->set_name("to");
                qmax_cast_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                qmax_cast_attr->set_i(*quantize_arithmetic_type);

                const std::string clamped_min_name = MakeLoweredName(node_base, "clamped_min", ++unique_id);
                AppendNode(lowered_nodes, node, "Max", MakeLoweredName(node_base, "max", unique_id),
                           {rounded_name, qmin_cast_name}, {clamped_min_name});
                const std::string clamped_name = MakeLoweredName(node_base, "clamped", ++unique_id);
                AppendNode(lowered_nodes, node, "Min", MakeLoweredName(node_base, "min", unique_id),
                           {clamped_min_name, qmax_cast_name}, {clamped_name});

                auto* final_cast =
                    AppendNode(lowered_nodes, node, "Cast", MakeLoweredName(node_base, "cast_output", ++unique_id),
                               {clamped_name}, {node.output(0)});
                auto* final_cast_attr = final_cast->add_attribute();
                final_cast_attr->set_name("to");
                final_cast_attr->set_type(onnx::AttributeProto_AttributeType_INT);
                final_cast_attr->set_i(output_type);

                ++lowered_qdq_info.lowered_node_count;
                continue;
            }
        }

        // Fall-through: policy said "keep native" OR a dtype/shape corner
        // case inside the rewrite declined. Either way, emit the original
        // node unchanged so ownership is preserved.
        *lowered_nodes.Add() = node;
    }

    // Invariant at this point: |lowered_nodes| >= |graph.node()|, every
    // original ORT node has either a verbatim copy or a set of helpers
    // sharing its doc_string, and any folded constant DQ has been logged
    // in lowered_qdq_info for the provider's capability remap.
    graph.mutable_node()->Swap(&lowered_nodes);
    return lowered_qdq_info;
}

}  // namespace trt_rtx_ep
