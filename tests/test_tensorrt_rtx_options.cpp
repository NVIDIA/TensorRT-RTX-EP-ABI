// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from onnxruntime/test/providers/nv_tensorrt_rtx/nv_options_test.cc
// Uses only public ORT SDK APIs.

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <filesystem>
#include <string>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

// Helper: append TRT RTX EP to session options.
static void AppendTrtRtxEp(
    Ort::SessionOptions& so,
    const std::unordered_map<std::string, std::string>& options = {}) {
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv_options;
    for (auto& [k, v] : options) {
        kv_options.Add(k.c_str(), v.c_str());
    }
    so.AppendExecutionProvider_V2(*ort_env, devices, kv_options);
}

static size_t countFilesInDirectory(const std::string& dir_path) {
    return static_cast<size_t>(std::distance(
        std::filesystem::directory_iterator(dir_path),
        std::filesystem::directory_iterator{}));
}

TEST(TensorRTRTXEpTest_Options, RuntimeCaching) {
    const std::string model_name = "nv_execution_provider_runtime_caching.onnx";
    const std::string model_name_ctx = "nv_execution_provider_runtime_caching_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    const std::string runtime_cache_dir = "./runtime_cache/";
    if (std::filesystem::exists(runtime_cache_dir)) {
        std::filesystem::remove_all(runtime_cache_dir);
    }

    model_builder::CreateBaseModel(model_name, "test", {1, 3, 2});

    // AOT: compile with runtime cache enabled
    {
        Ort::SessionOptions so;
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath,
                          model_name_ctx.c_str());
        AppendTrtRtxEp(so, {{"nv_runtime_cache_path", runtime_cache_dir}});
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }
    // Cache dumped on session destruction
    ASSERT_TRUE(std::filesystem::exists(runtime_cache_dir));
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir), 1u);

    // Use existing cache
    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_runtime_cache_path", runtime_cache_dir}});
        Ort::Session session(*ort_env, toOrtString(model_name_ctx).c_str(), so);
    }
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir), 1u);

    // Create new cache in different directory
    {
        const std::string new_cache_dir = "./runtime_cache_new/";
        if (std::filesystem::exists(new_cache_dir)) {
            std::filesystem::remove_all(new_cache_dir);
        }

        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_runtime_cache_path", new_cache_dir}});
        {
            Ort::Session session(*ort_env, toOrtString(model_name_ctx).c_str(), so);
        }
        // Cache dumped on session destruction
        ASSERT_TRUE(std::filesystem::exists(new_cache_dir));
        ASSERT_EQ(countFilesInDirectory(new_cache_dir), 1u);
    }
}
