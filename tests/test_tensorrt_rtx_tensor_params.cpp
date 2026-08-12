// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// End-to-end guard-rail test for the TensorParams input-rank check.
//
// TensorRT stores at most nvinfer1::Dims::MAX_DIMS (8) dimensions in a fixed
// inline array. The TensorParams(const void*, const std::vector<int64_t>&)
// constructor guards against a model/EPContext presenting a higher-rank input,
// rejecting it instead of writing past dims.d[]. This test feeds the EP a
// rank-9 model and asserts the EP-backed session is rejected gracefully with an
// Ort::Exception rather than crashing the process.
//
// Note: this exercises the behaviour end-to-end through public ORT APIs only.
// The rank-9 graph is refused during engine build, so this is a guard-rail /
// no-crash regression check; the exact constructor bound is additionally
// covered by code review.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{

// Append the TRT RTX EP using the global registration (mirrors the helper in
// test_tensorrt_rtx_basic.cpp).
void AppendTrtRtxEp(Ort::SessionOptions& so)
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv_options;
    so.AppendExecutionProvider_V2(*ort_env, devices, kv_options);
}

}  // namespace

TEST(TensorRTRTXEpTest_TensorParams, HighRankInputRejected)
{
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty());

    const std::string model_name = "nv_execution_provider_high_rank.onnx";
    clearFileIfExists(model_name);

    // Rank 9 (> TensorRT's MAX_DIMS of 8).
    const std::vector<int> dims(9, 1);
    model_builder::CreateBaseModel(model_name, "high_rank", dims);

    Ort::SessionOptions so;
    // Disable CPU fallback so a rank the EP cannot handle surfaces as an error
    // instead of silently running on the CPU EP and masking the rejection.
    so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    AppendTrtRtxEp(so);

    EXPECT_THROW({ Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so); }, Ort::Exception);

    clearFileIfExists(model_name);
}
