// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ort_model_dump.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#define ORT_EP_UTILS_C_RETURN_IF_ERROR(fn) \
    do                                     \
    {                                      \
        Ort::Status _status{(fn)};         \
        if (!_status.IsOK())               \
        {                                  \
            return _status;                \
        }                                  \
    } while (0)

#define ORT_EP_UTILS_CXX_RETURN_IF_ERROR(fn) ORT_EP_UTILS_C_RETURN_IF_ERROR(fn)

namespace OrtEpUtils
{
namespace
{

constexpr const char* kExternalMemAddrLocation = "_MEM_ADDR_";
constexpr size_t kExternalDataInlineThreshold = 1024;
#ifdef ORT_EP_UTILS_TEST_MAX_INLINE_PROTO_BYTES
constexpr size_t kMaxInlineProtoBytes = ORT_EP_UTILS_TEST_MAX_INLINE_PROTO_BYTES;
#else
constexpr size_t kMaxInlineProtoBytes = static_cast<size_t>((std::numeric_limits<int32_t>::max)());
#endif

bool TryGetExternalDataValue(const onnx::TensorProto& tensor, const std::string& key, std::string& value)
{
    for (const auto& entry : tensor.external_data())
    {
        if (entry.key() == key)
        {
            value = entry.value();
            return true;
        }
    }
    return false;
}

std::string BaseNameFromPath(const std::string& path)
{
    const auto slash_pos = path.find_last_of("/\\");
    return slash_pos == std::string::npos ? path : path.substr(slash_pos + 1);
}

Ort::Status WriteExternalDataBytes(std::ofstream& data_file, const void* data, size_t bytes, int64_t& offset_out)
{
    if (data == nullptr && bytes != 0)
    {
        return Ort::Status{"Cannot write external initializer data from a null pointer.", ORT_FAIL};
    }

    const auto offset = data_file.tellp();
    if (offset < 0)
    {
        return Ort::Status{"Failed to query external initializer data file offset.", ORT_FAIL};
    }

    offset_out = static_cast<int64_t>(offset);
    const char* current = static_cast<const char*>(data);
    size_t remaining = bytes;
    constexpr size_t kMaxWriteChunk = 1U << 30;  // 1 GiB

    while (remaining > 0)
    {
        const size_t chunk = (std::min)(remaining, kMaxWriteChunk);
        if (chunk > static_cast<size_t>((std::numeric_limits<std::streamsize>::max)()))
        {
            return Ort::Status{"External initializer write chunk exceeds streamsize limit.", ORT_FAIL};
        }

        data_file.write(current, static_cast<std::streamsize>(chunk));
        if (!data_file.good())
        {
            return Ort::Status{"Failed to write external initializer data.", ORT_FAIL};
        }

        current += chunk;
        remaining -= chunk;
    }

    return Ort::Status{nullptr};
}

void SetTensorExternalData(onnx::TensorProto& tensor, const std::string& location, int64_t offset, size_t bytes)
{
    tensor.clear_raw_data();
    tensor.clear_external_data();
    tensor.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);

    auto* location_entry = tensor.add_external_data();
    location_entry->set_key("location");
    location_entry->set_value(location);

    auto* offset_entry = tensor.add_external_data();
    offset_entry->set_key("offset");
    offset_entry->set_value(std::to_string(offset));

    auto* length_entry = tensor.add_external_data();
    length_entry->set_key("length");
    length_entry->set_value(std::to_string(bytes));
}

void SetTensorRawData(onnx::TensorProto& tensor, const void* data, size_t bytes)
{
    tensor.clear_external_data();
    tensor.set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
    tensor.set_raw_data(data, bytes);
}

Ort::Status GetTensorDataForDump(onnx::TensorProto& tensor, const void*& data, size_t& bytes)
{
    data = nullptr;
    bytes = 0;

    if (tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL)
    {
        std::string location;
        std::string offset;
        std::string length;
        (void)TryGetExternalDataValue(tensor, "location", location);
        (void)TryGetExternalDataValue(tensor, "offset", offset);
        (void)TryGetExternalDataValue(tensor, "length", length);

        if (location != kExternalMemAddrLocation)
        {
            return Ort::Status{nullptr};
        }

        if (offset.empty() || length.empty())
        {
            return Ort::Status{"_MEM_ADDR_ initializer is missing offset or length external_data.", ORT_FAIL};
        }

        const uintptr_t ptr_value = static_cast<uintptr_t>(std::stoull(offset));
        data = reinterpret_cast<const void*>(ptr_value);
        bytes = static_cast<size_t>(std::stoull(length));
    }
    else if (tensor.has_raw_data())
    {
        data = tensor.raw_data().data();
        bytes = tensor.raw_data().size();
    }

    return Ort::Status{nullptr};
}

Ort::Status DumpTensorWithExternalData(onnx::TensorProto& tensor, std::ofstream& data_file,
                                       const std::string& external_data_location)
{
    const void* data = nullptr;
    size_t bytes = 0;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetTensorDataForDump(tensor, data, bytes));

    if (data == nullptr || bytes == 0)
    {
        return Ort::Status{nullptr};
    }

    if (tensor.has_raw_data() && bytes <= kExternalDataInlineThreshold)
    {
        return Ort::Status{nullptr};
    }

    if (bytes <= kExternalDataInlineThreshold)
    {
        if (tensor.has_raw_data() && data == tensor.raw_data().data())
        {
            return Ort::Status{nullptr};
        }

        SetTensorRawData(tensor, data, bytes);
        return Ort::Status{nullptr};
    }

    int64_t file_offset = 0;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(WriteExternalDataBytes(data_file, data, bytes, file_offset));
    SetTensorExternalData(tensor, external_data_location, file_offset, bytes);
    return Ort::Status{nullptr};
}

Ort::Status DumpGraphWithInlineData(onnx::GraphProto& graph_proto);

Ort::Status DumpTensorWithInlineData(onnx::TensorProto& tensor)
{
    const void* data = nullptr;
    size_t bytes = 0;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetTensorDataForDump(tensor, data, bytes));

    if (data == nullptr || bytes == 0)
    {
        return Ort::Status{nullptr};
    }

    if (tensor.has_raw_data() && data == tensor.raw_data().data())
    {
        return Ort::Status{nullptr};
    }

    SetTensorRawData(tensor, data, bytes);
    return Ort::Status{nullptr};
}

Ort::Status DumpAttributeTensorsWithInlineData(onnx::AttributeProto& attr)
{
    if (attr.has_t())
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpTensorWithInlineData(*attr.mutable_t()));
    }

    for (int i = 0; i < attr.tensors_size(); ++i)
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpTensorWithInlineData(*attr.mutable_tensors(i)));
    }

    if (attr.has_g())
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpGraphWithInlineData(*attr.mutable_g()));
    }

    for (int i = 0; i < attr.graphs_size(); ++i)
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpGraphWithInlineData(*attr.mutable_graphs(i)));
    }

    return Ort::Status{nullptr};
}

Ort::Status DumpGraphWithInlineData(onnx::GraphProto& graph_proto)
{
    for (auto& tensor : *graph_proto.mutable_initializer())
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpTensorWithInlineData(tensor));
    }

    for (auto& node : *graph_proto.mutable_node())
    {
        for (auto& attr : *node.mutable_attribute())
        {
            ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpAttributeTensorsWithInlineData(attr));
        }
    }

    return Ort::Status{nullptr};
}

Ort::Status TryDumpModelWithInlineInitializers(onnx::ModelProto& model_proto, const std::string& onnx_path,
                                               bool& dumped_inline)
{
    dumped_inline = false;

    onnx::ModelProto inline_model_proto = model_proto;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpGraphWithInlineData(*inline_model_proto.mutable_graph()));

    if (inline_model_proto.ByteSizeLong() > kMaxInlineProtoBytes)
    {
        return Ort::Status{nullptr};
    }

    std::ofstream onnx_file(onnx_path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!onnx_file.good())
    {
        return Ort::Status{"Failed to open ONNX model file for writing.", ORT_FAIL};
    }

    if (!inline_model_proto.SerializeToOstream(&onnx_file))
    {
        return Ort::Status{nullptr};
    }

    model_proto = std::move(inline_model_proto);
    dumped_inline = true;
    return Ort::Status{nullptr};
}

Ort::Status DumpGraphWithExternalData(onnx::GraphProto& graph_proto, std::ofstream& data_file,
                                      const std::string& external_data_location);

Ort::Status DumpGraphInitializersWithExternalData(onnx::GraphProto& graph_proto, std::ofstream& data_file,
                                                  const std::string& external_data_location)
{
    for (auto& tensor : *graph_proto.mutable_initializer())
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(DumpTensorWithExternalData(tensor, data_file, external_data_location));
    }

    return Ort::Status{nullptr};
}

Ort::Status DumpAttributeTensorsWithExternalData(onnx::AttributeProto& attr, std::ofstream& data_file,
                                                 const std::string& external_data_location)
{
    if (attr.has_t())
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
            DumpTensorWithExternalData(*attr.mutable_t(), data_file, external_data_location));
    }

    for (int i = 0; i < attr.tensors_size(); ++i)
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
            DumpTensorWithExternalData(*attr.mutable_tensors(i), data_file, external_data_location));
    }

    if (attr.has_g())
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
            DumpGraphWithExternalData(*attr.mutable_g(), data_file, external_data_location));
    }

    for (int i = 0; i < attr.graphs_size(); ++i)
    {
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
            DumpGraphWithExternalData(*attr.mutable_graphs(i), data_file, external_data_location));
    }

    return Ort::Status{nullptr};
}

Ort::Status DumpGraphWithExternalData(onnx::GraphProto& graph_proto, std::ofstream& data_file,
                                      const std::string& external_data_location)
{
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
        DumpGraphInitializersWithExternalData(graph_proto, data_file, external_data_location));

    for (auto& node : *graph_proto.mutable_node())
    {
        for (auto& attr : *node.mutable_attribute())
        {
            ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
                DumpAttributeTensorsWithExternalData(attr, data_file, external_data_location));
        }
    }

    return Ort::Status{nullptr};
}

}  // namespace

Ort::Status DumpModelWithInitializers(onnx::ModelProto& model_proto, const std::string& onnx_path,
                                      const std::string& data_path)
{
    try
    {
        bool dumped_inline = false;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(TryDumpModelWithInlineInitializers(model_proto, onnx_path, dumped_inline));
        if (dumped_inline)
        {
            (void)std::remove(data_path.c_str());
            return Ort::Status{nullptr};
        }

        std::ofstream data_file(data_path, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!data_file.good())
        {
            return Ort::Status{"Failed to open external initializer data file for writing.", ORT_FAIL};
        }

        const std::string external_data_location = BaseNameFromPath(data_path);
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(
            DumpGraphWithExternalData(*model_proto.mutable_graph(), data_file, external_data_location));

        data_file.flush();
        if (!data_file.good())
        {
            return Ort::Status{"Failed to flush external initializer data file.", ORT_FAIL};
        }

        std::ofstream onnx_file(onnx_path, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!onnx_file.good())
        {
            return Ort::Status{"Failed to open ONNX model file for writing.", ORT_FAIL};
        }

        if (!model_proto.SerializeToOstream(&onnx_file))
        {
            return Ort::Status{"Failed to serialize ONNX model with external initializers.", ORT_FAIL};
        }

        return Ort::Status{nullptr};
    }
    catch (const std::exception& ex)
    {
        return Ort::Status{ex.what(), ORT_FAIL};
    }
}

}  // namespace OrtEpUtils
