// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_rtx_profiler_metadata.h"

#include <string>

#include <gtest/gtest.h>

namespace trt_rtx_ep::profiler_metadata
{

TEST(TensorRTRTXEpTest_ProfilerMetadata, DecodesEscapedDelimiterBeforeSplitting)
{
    constexpr std::string_view json =
        R"({"Name": "layer", "Metadata": "[ONNX Layer: /conv/Conv]\u001f[ONNX Layer: /relu/Relu]"})";

    std::string metadata;
    ASSERT_TRUE(GetJsonStringField(json, "Metadata", metadata));
    EXPECT_EQ(metadata, std::string("[ONNX Layer: /conv/Conv]\x1f[ONNX Layer: /relu/Relu]"));
    EXPECT_EQ(ExtractOnnxNodeNames(metadata), "/conv/Conv, /relu/Relu");
}

TEST(TensorRTRTXEpTest_ProfilerMetadata, SplitsDecodedUnitAndRecordSeparators)
{
    const std::string metadata = "[ONNX Layer: /conv/Conv]\x1e[ONNX Layer: /relu/Relu]\x1f[ONNX Layer: /add/Add]";

    EXPECT_EQ(ExtractOnnxNodeNames(metadata), "/conv/Conv, /relu/Relu, /add/Add");
}

TEST(TensorRTRTXEpTest_ProfilerMetadata, HandlesIndentedJsonWhitespace)
{
    constexpr std::string_view json =
        "{\n"
        "\t\"Name\":\n\t\t\"layer\",\n"
        "\t\"Metadata\":\r\n\t\t\"[ONNX Layer: /conv/Conv]\\u001f[ONNX Layer: /relu/Relu]\"\n"
        "}";

    std::string layer_name;
    std::string metadata;
    ASSERT_TRUE(GetJsonStringField(json, "Name", layer_name));
    ASSERT_TRUE(GetJsonStringField(json, "Metadata", metadata));
    EXPECT_EQ(layer_name, "layer");
    EXPECT_EQ(ExtractOnnxNodeNames(metadata), "/conv/Conv, /relu/Relu");
}

}  // namespace trt_rtx_ep::profiler_metadata
