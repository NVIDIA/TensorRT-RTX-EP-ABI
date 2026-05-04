// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "logical_output_compatibility.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_rtx_ep
{
namespace
{

struct TensorMetadata
{
    std::optional<int32_t> element_type;
};

struct GraphIndex
{
    explicit GraphIndex(const onnx::GraphProto& graph_proto)
        : graph(graph_proto)
    {
        Build();
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
        const auto it = producer_node_indices.find(name);
        return it == producer_node_indices.end() ? nullptr : GetNode(it->second);
    }

    const std::vector<size_t>* FindConsumerNodeIndices(const std::string& name) const
    {
        const auto it = consumer_node_indices.find(name);
        return it == consumer_node_indices.end() ? nullptr : &it->second;
    }

    std::optional<int32_t> FindTensorElementType(const std::string& name) const
    {
        const auto it = tensor_metadata.find(name);
        if (it == tensor_metadata.end())
        {
            return std::nullopt;
        }
        return it->second.element_type;
    }

    const onnx::GraphProto& graph;
    std::unordered_map<std::string, TensorMetadata> tensor_metadata;
    std::unordered_map<std::string, size_t> producer_node_indices;
    std::unordered_map<std::string, std::vector<size_t>> consumer_node_indices;

private:
    void Build()
    {
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

    void AddValueInfo(const onnx::ValueInfoProto& value_info)
    {
        if (!value_info.has_type() || !value_info.type().has_tensor_type())
        {
            return;
        }

        tensor_metadata[value_info.name()].element_type = value_info.type().tensor_type().elem_type();
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

onnx::AttributeProto* FindMutableAttribute(onnx::NodeProto& node, const std::string& name)
{
    for (auto& attribute : *node.mutable_attribute())
    {
        if (attribute.name() == name)
        {
            return &attribute;
        }
    }
    return nullptr;
}

void UpdateTensorElementType(google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>* value_infos,
                             const std::string& tensor_name, int32_t element_type)
{
    for (auto& value_info : *value_infos)
    {
        if (value_info.name() != tensor_name)
        {
            continue;
        }

        value_info.mutable_type()->mutable_tensor_type()->set_elem_type(element_type);
    }
}

bool IsKnownBoolProducer(const onnx::NodeProto& node)
{
    return node.op_type() == "Equal" || node.op_type() == "NotEqual" || node.op_type() == "Greater" ||
           node.op_type() == "GreaterOrEqual" || node.op_type() == "Less" || node.op_type() == "LessOrEqual" ||
           node.op_type() == "And" || node.op_type() == "Or" || node.op_type() == "Xor" || node.op_type() == "Not" ||
           node.op_type() == "IsInf" || node.op_type() == "IsNaN";
}

bool IsBoolTensor(const GraphIndex& index, const std::string& tensor_name)
{
    const auto tensor_type = index.FindTensorElementType(tensor_name);
    if (tensor_type.has_value())
    {
        return *tensor_type == onnx::TensorProto_DataType_BOOL;
    }

    const auto* producer = index.FindProducer(tensor_name);
    return producer != nullptr && IsKnownBoolProducer(*producer);
}

// Canonicalize Chromium's bool -> uint8 -> Cast(T) bridge to
// bool -> Cast(int32) -> Cast(T) when every downstream use is already a cast.
// This removes the unsupported UINT8 intermediate while keeping a single
// parser-friendly bridge type that can fan out to different final cast types.
void CanonicalizeBoolBridgeCastChain(onnx::GraphProto& graph)
{
    const GraphIndex index(graph);

    std::unordered_set<std::string> graph_outputs;
    for (const auto& output : graph.output())
    {
        if (!output.name().empty())
        {
            graph_outputs.insert(output.name());
        }
    }

    for (int node_index = 0; node_index < graph.node_size(); ++node_index)
    {
        auto* uint8_bridge = graph.mutable_node(node_index);
        if (uint8_bridge->op_type() != "Cast" || uint8_bridge->input_size() != 1 || uint8_bridge->output_size() != 1 ||
            uint8_bridge->input(0).empty() || uint8_bridge->output(0).empty())
        {
            continue;
        }

        const auto bridge_type = FindIntAttribute(*uint8_bridge, "to");
        if (!bridge_type || *bridge_type != onnx::TensorProto_DataType_UINT8)
        {
            continue;
        }

        if (graph_outputs.find(uint8_bridge->output(0)) != graph_outputs.end())
        {
            continue;
        }

        if (!IsBoolTensor(index, uint8_bridge->input(0)))
        {
            continue;
        }

        const auto* consumer_indices = index.FindConsumerNodeIndices(uint8_bridge->output(0));
        if (consumer_indices == nullptr || consumer_indices->empty())
        {
            continue;
        }

        bool can_retarget_bridge = true;
        for (const size_t consumer_index : *consumer_indices)
        {
            const auto* consumer = index.GetNode(consumer_index);
            if (consumer == nullptr || consumer->op_type() != "Cast" || consumer->input_size() != 1 ||
                consumer->input(0) != uint8_bridge->output(0))
            {
                can_retarget_bridge = false;
                break;
            }

            const auto consumer_type = FindIntAttribute(*consumer, "to");
            if (!consumer_type || *consumer_type == onnx::TensorProto_DataType_UINT8)
            {
                can_retarget_bridge = false;
                break;
            }
        }

        if (!can_retarget_bridge)
        {
            continue;
        }

        auto* bridge_attr = FindMutableAttribute(*uint8_bridge, "to");
        if (bridge_attr == nullptr)
        {
            continue;
        }

        bridge_attr->set_i(onnx::TensorProto_DataType_INT32);
        UpdateTensorElementType(graph.mutable_value_info(), uint8_bridge->output(0), onnx::TensorProto_DataType_INT32);
        UpdateTensorElementType(graph.mutable_output(), uint8_bridge->output(0), onnx::TensorProto_DataType_INT32);
    }
}

}  // namespace

void RunLogicalOutputCompatibilityForTensorRt(onnx::ModelProto& model_proto)
{
    CanonicalizeBoolBridgeCastChain(*model_proto.mutable_graph());
}

}  // namespace trt_rtx_ep
