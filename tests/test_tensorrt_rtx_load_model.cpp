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

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>

#include "test_tensorrt_rtx_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

TEST(TensorRTRTXEpTest_LoadModel, ModelFileExists) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;
}

TEST(TensorRTRTXEpTest_LoadModel, CreatesSessionSuccessfully) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    Ort::SessionOptions session_options;
    Ort::KeyValuePairs ep_options;
    const auto trt_rtx_devices = get_trt_rtx_devices(*ort_env);
    session_options.AppendExecutionProvider_V2(*ort_env, trt_rtx_devices, ep_options);
    Ort::Session session(*ort_env, toOrtString(kModelPath).c_str(), session_options);

    EXPECT_EQ(session.GetInputCount(), 1);
    EXPECT_EQ(session.GetOutputCount(), 1);
}
