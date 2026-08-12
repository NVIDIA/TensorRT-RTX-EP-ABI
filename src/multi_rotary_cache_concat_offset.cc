// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "multi_rotary_cache_concat_offset.h"

#include <algorithm>

namespace trt_rtx_ep
{
namespace
{

constexpr const char* kGroupQueryAttentionOpType = "GroupQueryAttention";
constexpr const char* kMultiRotaryCacheConcatOffsetAttribute = "multiRotaryCacheConcatOffset";

bool HasAttribute(const onnx::NodeProto& node, const char* attribute_name)
{
    return std::any_of(node.attribute().begin(), node.attribute().end(),
                       [attribute_name](const onnx::AttributeProto& attribute)
                       {
                           return attribute.name() == attribute_name;
                       });
}

void ApplyMultiRotaryCacheConcatOffset(onnx::GraphProto& graph, int64_t offset)
{
    for (auto& node : *graph.mutable_node())
    {
        if (node.op_type() == kGroupQueryAttentionOpType && !HasAttribute(node, kMultiRotaryCacheConcatOffsetAttribute))
        {
            auto* attribute = node.add_attribute();
            attribute->set_name(kMultiRotaryCacheConcatOffsetAttribute);
            attribute->set_type(onnx::AttributeProto_AttributeType_INT);
            attribute->set_i(offset);
        }

        for (auto& attribute : *node.mutable_attribute())
        {
            if (attribute.has_g())
            {
                ApplyMultiRotaryCacheConcatOffset(*attribute.mutable_g(), offset);
            }
            for (auto& nested_graph : *attribute.mutable_graphs())
            {
                ApplyMultiRotaryCacheConcatOffset(nested_graph, offset);
            }
        }
    }
}

}  // namespace

void RunMultiRotaryCacheConcatOffsetForTensorRt(onnx::ModelProto& model_proto, int64_t offset)
{
    if (offset == 0 || !model_proto.has_graph())
    {
        return;
    }

    ApplyMultiRotaryCacheConcatOffset(*model_proto.mutable_graph(), offset);
}

}  // namespace trt_rtx_ep
