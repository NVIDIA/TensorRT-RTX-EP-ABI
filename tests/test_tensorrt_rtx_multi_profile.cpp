// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <filesystem>
#include <future>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{
constexpr const char* kProfileMinShapes = "X:1x1x1,Y:1x1x1,Z:1x1x1,X:1x5x5,Y:1x5x5,Z:1x5x5";
constexpr const char* kProfileOptShapes = "X:1x1x1,Y:1x1x1,Z:1x1x1,X:1x5x5,Y:1x5x5,Z:1x5x5";
constexpr const char* kProfileMaxShapes = "X:1x1x1,Y:1x1x1,Z:1x1x1,X:1x5x5,Y:1x5x5,Z:1x5x5";
constexpr const char* kOverlappingProfileShapes = "X:1x3x3,Y:1x3x3,Z:1x3x3,X:1x3x3,Y:1x3x3,Z:1x3x3";

void AppendTrtRtxEp(Ort::SessionOptions& session_options,
                    const std::unordered_map<std::string, std::string>& options = {})
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";

    Ort::KeyValuePairs provider_options;
    for (const auto& [key, value] : options)
    {
        provider_options.Add(key.c_str(), value.c_str());
    }
    session_options.AppendExecutionProvider_V2(*ort_env, devices, provider_options);
}

std::map<std::string, std::vector<int64_t>> InputShapes(int64_t spatial_size)
{
    const std::vector<int64_t> shape = {1, spatial_size, spatial_size};
    return {{"X", shape}, {"Y", shape}, {"Z", shape}};
}

void RunWithProfile(Ort::Session& session, int profile_index, Ort::IoBinding& io_binding)
{
    Ort::RunOptions run_options;
    const std::string profile_index_string = std::to_string(profile_index);
    run_options.AddConfigEntry("nv_profile_index", profile_index_string.c_str());
    session.Run(run_options, io_binding);
}

void RunWithProfile(Ort::Session& session, int profile_index, int64_t spatial_size)
{
    auto io_binding = generate_io_binding(session, InputShapes(spatial_size));
    RunWithProfile(session, profile_index, io_binding);
}

std::unordered_map<std::string, std::string> MultiProfileProviderOptions()
{
    return {
        {"nv_profile_min_shapes", kProfileMinShapes},
        {"nv_profile_opt_shapes", kProfileOptShapes},
        {"nv_profile_max_shapes", kProfileMaxShapes},
        {"nv_multi_profile_enable", "1"},
    };
}
}  // namespace

TEST(TensorRTRTXEpTest_MultiProfile, DirectEngineSelectsRequestedProfileForEachRun)
{
    const std::string model_path = "nv_execution_provider_multi_profile.onnx";
    model_builder::CreateBaseModel(model_path, "multi_profile", {1, -1, -1});

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options, MultiProfileProviderOptions());
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);

    ASSERT_NO_THROW(RunWithProfile(session, 0, 1));
    ASSERT_NO_THROW(RunWithProfile(session, 1, 5));
    ASSERT_NO_THROW(RunWithProfile(session, 0, 1));
}

TEST(TensorRTRTXEpTest_MultiProfile, ProfileChangeRebindsUnchangedInputs)
{
    const std::string model_path = "nv_execution_provider_overlapping_multi_profile.onnx";
    model_builder::CreateBaseModel(model_path, "overlapping_multi_profile", {1, -1, -1});

    Ort::SessionOptions session_options;
    auto provider_options = MultiProfileProviderOptions();
    provider_options["nv_profile_min_shapes"] = kOverlappingProfileShapes;
    provider_options["nv_profile_opt_shapes"] = kOverlappingProfileShapes;
    provider_options["nv_profile_max_shapes"] = kOverlappingProfileShapes;
    AppendTrtRtxEp(session_options, provider_options);
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);

    auto io_binding = generate_io_binding(session, InputShapes(3));
    ASSERT_NO_THROW(RunWithProfile(session, 0, io_binding));
    ASSERT_NO_THROW(RunWithProfile(session, 1, io_binding));
    ASSERT_NO_THROW(RunWithProfile(session, 0, io_binding));
}

TEST(TensorRTRTXEpTest_MultiProfile, ConcurrentCallsWithDifferentProfilesAreSerialized)
{
    const std::string model_path = "nv_execution_provider_concurrent_multi_profile.onnx";
    model_builder::CreateBaseModel(model_path, "concurrent_multi_profile", {1, -1, -1});

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options, MultiProfileProviderOptions());
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    auto run_profile = [&](int profile_index, int64_t spatial_size)
    {
        ++ready;
        while (!start.load())
        {
            std::this_thread::yield();
        }
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            RunWithProfile(session, profile_index, spatial_size);
        }
    };

    auto profile_zero = std::async(std::launch::async, run_profile, 0, 1);
    auto profile_one = std::async(std::launch::async, run_profile, 1, 5);
    while (ready.load() != 2)
    {
        std::this_thread::yield();
    }
    start = true;

    ASSERT_NO_THROW(profile_zero.get());
    ASSERT_NO_THROW(profile_one.get());
}

TEST(TensorRTRTXEpTest_MultiProfile, EpContextEngineEnablesAndSelectsProfilesWithoutProviderOption)
{
    const std::string model_path = "nv_execution_provider_multi_profile_ep_context.onnx";
    const std::string ep_context_path = "nv_execution_provider_multi_profile_ep_context_ctx.onnx";
    clearFileIfExists(ep_context_path);
    model_builder::CreateBaseModel(model_path, "multi_profile_ep_context", {1, -1, -1});

    // Build and serialize an engine containing two optimization profiles.
    {
        Ort::SessionOptions session_options;
        session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_path.c_str());
        AppendTrtRtxEp(session_options, MultiProfileProviderOptions());
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);

        ASSERT_NO_THROW(RunWithProfile(session, 0, 1));
        ASSERT_TRUE(std::filesystem::is_regular_file(ep_context_path));
    }

    // Do not pass nv_multi_profile_enable here. The EP must discover profile support from
    // ICudaEngine::getNbOptimizationProfiles() while loading the EP-context engine.
    {
        Ort::SessionOptions session_options;
        AppendTrtRtxEp(session_options);
        Ort::Session session(*ort_env, toOrtString(ep_context_path).c_str(), session_options);

        ASSERT_NO_THROW(RunWithProfile(session, 0, 1));
        ASSERT_NO_THROW(RunWithProfile(session, 1, 5));
        ASSERT_NO_THROW(RunWithProfile(session, 0, 1));
    }
}
