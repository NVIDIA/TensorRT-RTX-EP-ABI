// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace {

constexpr size_t kMB = 1024ull * 1024;
constexpr float kDefaultAbsTolerance = 1e-4f;
constexpr float kDefaultRelTolerance = 1e-4f;
constexpr float kIndependentSessionAbsTolerance = 3e-3f;
constexpr float kIndependentSessionRelTolerance = 1e-4f;
constexpr const char* kWeightStreamingBudgetLogMarker =
    "Weight streaming budget applied";

struct LogCapture {
    mutable std::mutex mutex;
    std::vector<std::string> messages;
};

struct WeightStreamingBudgetLog {
    std::string requested;
    int64_t resolved;
    int64_t actual;
    int64_t streamable_size;
    int64_t scratch_size;
};

void ORT_API_CALL CaptureWeightStreamingLog(void* param,
                                            OrtLoggingLevel /*severity*/,
                                            const char* /*category*/,
                                            const char* /*logid*/,
                                            const char* /*code_location*/,
                                            const char* message) {
    if (param == nullptr || message == nullptr ||
        std::strstr(message, kWeightStreamingBudgetLogMarker) == nullptr) {
        return;
    }

    auto* capture = static_cast<LogCapture*>(param);
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->messages.emplace_back(message);
}

void AttachLogCapture(Ort::SessionOptions& session_options,
                      LogCapture& log_capture) {
    session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
    Ort::ThrowOnError(Ort::GetApi().SetUserLoggingFunction(
        session_options, CaptureWeightStreamingLog, &log_capture));
}

std::vector<std::string> CopyCapturedMessages(const LogCapture& log_capture) {
    std::lock_guard<std::mutex> lock(log_capture.mutex);
    return log_capture.messages;
}

std::optional<WeightStreamingBudgetLog> TryParseWeightStreamingBudgetLog(
    const std::string& message) {
    static const std::regex log_pattern(
        R"(requested=([^,]+), resolved=(-?\d+), actual=(-?\d+), streamable_weights_size=(-?\d+), scratch_memory_size=(-?\d+))");
    std::smatch match;
    if (!std::regex_search(message, match, log_pattern)) {
        return std::nullopt;
    }

    return WeightStreamingBudgetLog{
        match[1].str(),
        std::stoll(match[2].str()),
        std::stoll(match[3].str()),
        std::stoll(match[4].str()),
        std::stoll(match[5].str())};
}

std::vector<WeightStreamingBudgetLog> GetCapturedBudgetLogs(
    const LogCapture& log_capture) {
    std::vector<WeightStreamingBudgetLog> logs;
    for (const auto& message : CopyCapturedMessages(log_capture)) {
        auto parsed = TryParseWeightStreamingBudgetLog(message);
        if (parsed) {
            logs.push_back(*parsed);
        }
    }
    return logs;
}

std::vector<WeightStreamingBudgetLog> ExpectBudgetLogs(
    const LogCapture& log_capture,
    const std::string& requested_budget) {
    auto logs = GetCapturedBudgetLogs(log_capture);
    EXPECT_FALSE(logs.empty()) << "Expected weight streaming budget log.";
    for (const auto& log : logs) {
        EXPECT_EQ(log.requested, requested_budget);
        EXPECT_GE(log.actual, 0);
        EXPECT_GE(log.streamable_size, 0);
        EXPECT_LE(log.actual, log.streamable_size);
        EXPECT_GE(log.scratch_size, 0);
    }
    return logs;
}

void ExpectNoBudgetLogs(const LogCapture& log_capture) {
    const auto logs = GetCapturedBudgetLogs(log_capture);
    EXPECT_TRUE(logs.empty()) << "Unexpected weight streaming budget log.";
}

void ExpectAutomaticBudgetApplied(const LogCapture& log_capture) {
    for (const auto& log : ExpectBudgetLogs(log_capture, "-1")) {
        EXPECT_EQ(log.actual, log.resolved);
    }
}

void ExpectExplicitBudgetApplied(const LogCapture& log_capture,
                                 const std::string& requested_text,
                                 int64_t requested_bytes) {
    for (const auto& log : ExpectBudgetLogs(log_capture, requested_text)) {
        EXPECT_EQ(log.resolved, requested_bytes);
        EXPECT_EQ(log.actual, std::min(requested_bytes, log.streamable_size));
    }
}

void ExpectMinimumVramBudgetApplied(const LogCapture& log_capture) {
    for (const auto& log : ExpectBudgetLogs(log_capture, "1")) {
        EXPECT_EQ(log.resolved, 0);
        EXPECT_EQ(log.actual, 0);
        if (log.streamable_size > 0) {
            EXPECT_LT(log.actual, log.streamable_size);
        }
    }
}

void ExpectStreamingDisabledByBudget(const LogCapture& log_capture,
                                     int64_t requested_budget) {
    for (const auto& log : ExpectBudgetLogs(log_capture, std::to_string(requested_budget))) {
        EXPECT_EQ(log.resolved, requested_budget);
        EXPECT_EQ(log.actual, log.streamable_size);
    }
}

void ExpectPercentBudgetApplied(const LogCapture& log_capture,
                                const std::string& requested_budget,
                                double percent) {
    for (const auto& log : ExpectBudgetLogs(log_capture, requested_budget)) {
        const auto expected_resolved =
            static_cast<int64_t>((percent / 100.0) * log.streamable_size);
        EXPECT_EQ(log.resolved, expected_resolved);
        EXPECT_EQ(log.actual, std::min(expected_resolved, log.streamable_size));
    }
}

void AppendTrtRtxEp(
    Ort::SessionOptions& session_options,
    const std::unordered_map<std::string, std::string>& options = {}) {
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv_options;
    for (const auto& [key, value] : options) {
        kv_options.Add(key.c_str(), value.c_str());
    }
    session_options.AppendExecutionProvider_V2(*ort_env, devices, kv_options);
}

size_t CountFilesInDirectory(const std::filesystem::path& dir_path) {
    if (!std::filesystem::exists(dir_path)) {
        return 0;
    }

    return static_cast<size_t>(std::distance(
        std::filesystem::directory_iterator(dir_path),
        std::filesystem::directory_iterator{}));
}

void RemoveDirectoryIfExists(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove_all(path);
    }
}

void ExpectSessionCreationFailsWith(
    const std::filesystem::path& model_path,
    Ort::SessionOptions& session_options,
    const std::string& expected_substring) {
    try {
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);
        FAIL() << "Expected session creation to fail.";
    } catch (const std::exception& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find(expected_substring), std::string::npos) << message;
    }
}

void ExpectProviderOptionsRejectedWith(
    const std::filesystem::path& model_path,
    const std::unordered_map<std::string, std::string>& ep_options,
    const std::string& expected_substring) {
    try {
        Ort::SessionOptions session_options;
        AppendTrtRtxEp(session_options, ep_options);
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);
        FAIL() << "Expected provider options or session creation to fail.";
    } catch (const std::exception& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find(expected_substring), std::string::npos) << message;
    }
}

std::unique_ptr<Ort::Session> CreateSession(
    const std::filesystem::path& model_path,
    const std::unordered_map<std::string, std::string>& ep_options = {},
    LogCapture* log_capture = nullptr) {
    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options, ep_options);
    if (log_capture != nullptr) {
        AttachLogCapture(session_options, *log_capture);
    }
    return std::make_unique<Ort::Session>(
        *ort_env, toOrtString(model_path).c_str(), session_options);
}

void RunOnce(Ort::Session& session) {
    auto io_binding = generate_io_binding(session);
    Ort::RunOptions run_options;
    session.Run(run_options, io_binding);
}

std::vector<float> RunAndCollectFloatOutputs(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::AllocatedStringPtr> input_name_storage;
    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_values;
    input_name_storage.reserve(session.GetInputCount());
    input_names.reserve(session.GetInputCount());
    input_values.reserve(session.GetInputCount());

    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        input_name_storage.emplace_back(session.GetInputNameAllocated(i, allocator));
        input_names.push_back(input_name_storage.back().get());

        auto type_info = session.GetInputTypeInfo(i);
        auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        for (auto& dim : shape) {
            if (dim < 0) {
                dim = 1;
            }
        }

        auto value = Ort::Value::CreateTensor(
            allocator, shape.data(), shape.size(), shape_info.GetElementType());
        std::memset(value.GetTensorMutableData<uint8_t>(), 0, ORTValueToBytes(value));
        input_values.push_back(std::move(value));
    }

    std::vector<Ort::AllocatedStringPtr> output_name_storage;
    std::vector<const char*> output_names;
    output_name_storage.reserve(session.GetOutputCount());
    output_names.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        output_name_storage.emplace_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_storage.back().get());
    }

    Ort::RunOptions run_options;
    auto outputs = session.Run(run_options,
                               input_names.data(),
                               input_values.data(),
                               input_values.size(),
                               output_names.data(),
                               output_names.size());

    std::vector<float> result;
    for (auto& output : outputs) {
        const auto shape_info = output.GetTensorTypeAndShapeInfo();
        if (shape_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Expected float output from weight streaming test model.");
        }

        const size_t element_count = shape_info.GetElementCount();
        const float* data = output.GetTensorData<float>();
        result.insert(result.end(), data, data + element_count);
    }

    return result;
}

void ExpectCloseOutputs(const std::vector<float>& expected,
                        const std::vector<float>& actual,
                        const std::string& comparison = {},
                        float abs_tolerance = kDefaultAbsTolerance,
                        float rel_tolerance = kDefaultRelTolerance) {
    ASSERT_EQ(expected.size(), actual.size())
        << (comparison.empty() ? "" : comparison);
    for (size_t i = 0; i < expected.size(); ++i) {
        const float tolerance = abs_tolerance + rel_tolerance * std::abs(expected[i]);
        ASSERT_NEAR(expected[i], actual[i], tolerance)
            << (comparison.empty() ? "" : comparison + ": ")
            << "Mismatch at output element " << i;
    }
}

void BuildStreamingEpContext(const std::filesystem::path& source_model,
                             const std::filesystem::path& ep_context_model,
                             const std::string& budget,
                             bool embed_mode = false,
                             std::unordered_map<std::string, std::string> ep_options = {},
                             LogCapture* log_capture = nullptr) {
    clearFileIfExists(ep_context_model);

    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    if (embed_mode) {
        session_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    }
    const auto ep_context_path = ep_context_model.string();
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath,
                                   ep_context_path.c_str());
    ep_options["nv_weight_streaming_budget"] = budget;
    AppendTrtRtxEp(session_options, ep_options);
    if (log_capture != nullptr) {
        AttachLogCapture(session_options, *log_capture);
    }

    Ort::Session session(*ort_env, toOrtString(source_model).c_str(), session_options);
    RunOnce(session);
}

}  // namespace

TEST(WeightStreamingTest, DefaultBudgetRunsWithoutStreamingOption) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    auto session = CreateSession(kModelPath, {}, &log_capture);
    RunOnce(*session);
    ExpectNoBudgetLogs(log_capture);
}

TEST(WeightStreamingTest, RejectsBudgetBelowMinusOne) {
    const std::string model_name = "nv_weight_streaming_invalid_budget.onnx";
    model_builder::CreateBaseModel(model_name, "weight_streaming_invalid_budget", {1, 3, 2});

    ExpectProviderOptionsRejectedWith(
        model_name, {{"nv_weight_streaming_budget", "-2"}},
        "Invalid nv_weight_streaming_budget: -2");
}

TEST(WeightStreamingTest, RejectsInvalidBudgetUnitSuffix) {
    const std::string model_name = "nv_weight_streaming_invalid_unit_budget.onnx";
    model_builder::CreateBaseModel(model_name, "weight_streaming_invalid_unit_budget", {1, 3, 2});

    ExpectProviderOptionsRejectedWith(
        model_name, {{"nv_weight_streaming_budget", "1KB"}},
        "Valid base-2 unit suffixes include: B, K, M, G");
}

TEST(WeightStreamingTest, RejectsBudgetPercentOutsideRange) {
    const std::string model_name = "nv_weight_streaming_invalid_percent_budget.onnx";
    model_builder::CreateBaseModel(model_name, "weight_streaming_invalid_percent_budget", {1, 3, 2});

    ExpectProviderOptionsRejectedWith(
        model_name, {{"nv_weight_streaming_budget", "101%"}},
        "weight streaming percentage must be between 0 and 100");
}

TEST(WeightStreamingTest, AutomaticBudgetRunsModel) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    auto session = CreateSession(
        kModelPath, {{"nv_weight_streaming_budget", "-1"}}, &log_capture);
    RunOnce(*session);
    ExpectAutomaticBudgetApplied(log_capture);
}

TEST(WeightStreamingTest, ExplicitByteBudgetRunsModel) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    auto session = CreateSession(
        kModelPath, {{"nv_weight_streaming_budget", std::to_string(kMB)}}, &log_capture);
    RunOnce(*session);
    ExpectExplicitBudgetApplied(log_capture, std::to_string(kMB), kMB);
}

TEST(WeightStreamingTest, ExplicitByteBudgetWithUnitSuffixRunsModel) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    auto session = CreateSession(
        kModelPath, {{"nv_weight_streaming_budget", "1B"}}, &log_capture);
    RunOnce(*session);
    ExpectExplicitBudgetApplied(log_capture, "1B", 1);
}

TEST(WeightStreamingTest, MebibyteBudgetWithUnitSuffixRunsModel) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    auto session = CreateSession(
        kModelPath, {{"nv_weight_streaming_budget", "1M"}}, &log_capture);
    RunOnce(*session);
    ExpectExplicitBudgetApplied(log_capture, "1M", kMB);
}

TEST(WeightStreamingTest, PercentBudgetRunsModel) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    auto session = CreateSession(
        kModelPath, {{"nv_weight_streaming_budget", "50%"}}, &log_capture);
    RunOnce(*session);
    ExpectPercentBudgetApplied(log_capture, "50%", 50.0);
}

TEST(WeightStreamingTest, BudgetAboveStreamableDisablesStreaming) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    LogCapture log_capture;
    constexpr int64_t kOversizedBudget = 1099511627776;
    auto session = CreateSession(
        kModelPath, {{"nv_weight_streaming_budget", std::to_string(kOversizedBudget)}},
        &log_capture);
    RunOnce(*session);
    ExpectStreamingDisabledByBudget(log_capture, kOversizedBudget);
}

TEST(WeightStreamingTest, EpContextNonStreamingEngineRejectsBudget) {
    const std::string model_name = "nv_weight_streaming_non_streaming_ctx.onnx";
    const std::string model_name_ctx = "nv_weight_streaming_non_streaming_ctx_cached.onnx";
    clearFileIfExists(model_name_ctx);

    model_builder::CreateBaseModel(model_name, "weight_streaming_non_streaming_ctx", {1, 3, 2});

    {
        Ort::SessionOptions session_options;
        session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx.c_str());
        AppendTrtRtxEp(session_options);
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), session_options);
        RunOnce(session);
    }

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options, {{"nv_weight_streaming_budget", "-1"}});

    ExpectSessionCreationFailsWith(
        model_name_ctx, session_options,
        "Failed to set nv_weight_streaming_budget=-1");
}

TEST(WeightStreamingTest, EpContextStreamingEngineRoundTripsWithBudget) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    const std::filesystem::path model_name_ctx =
        "nv_weight_streaming_streaming_ctx.onnx";

    BuildStreamingEpContext(kModelPath, model_name_ctx, "-1");

    LogCapture log_capture;
    auto session = CreateSession(
        model_name_ctx, {{"nv_weight_streaming_budget", "-1"}}, &log_capture);
    RunOnce(*session);
    ExpectAutomaticBudgetApplied(log_capture);
}

TEST(WeightStreamingTest, EmbeddedEpContextStreamingEngineRunsWithBudget) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    const std::filesystem::path model_name_ctx =
        "nv_weight_streaming_embedded_streaming_ctx.onnx";

    BuildStreamingEpContext(kModelPath, model_name_ctx, "-1", true);

    LogCapture log_capture;
    auto session = CreateSession(
        model_name_ctx, {{"nv_weight_streaming_budget", "-1"}}, &log_capture);
    RunOnce(*session);
    ExpectAutomaticBudgetApplied(log_capture);
}

TEST(WeightStreamingTest, EpContextStreamingCapableEngineRunsWithoutBudget) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    const std::filesystem::path model_name_ctx =
        "nv_weight_streaming_streaming_capable_budget_unset_ctx.onnx";

    BuildStreamingEpContext(kModelPath, model_name_ctx, "-1");

    LogCapture log_capture;
    auto session = CreateSession(model_name_ctx, {}, &log_capture);
    RunOnce(*session);
    ExpectNoBudgetLogs(log_capture);
}

TEST(WeightStreamingTest, RuntimeCacheComposesWithWeightStreaming) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    const std::filesystem::path runtime_cache_dir =
        "nv_weight_streaming_runtime_cache";
    RemoveDirectoryIfExists(runtime_cache_dir);

    {
        LogCapture log_capture;
        auto session = CreateSession(
            kModelPath,
            {{"nv_weight_streaming_budget", "-1"},
             {"nv_runtime_cache_path", runtime_cache_dir.string()}},
            &log_capture);
        RunOnce(*session);
        ExpectAutomaticBudgetApplied(log_capture);
    }

    ASSERT_TRUE(std::filesystem::exists(runtime_cache_dir));
    ASSERT_GE(CountFilesInDirectory(runtime_cache_dir), 1u);

    {
        LogCapture log_capture;
        auto session = CreateSession(
            kModelPath,
            {{"nv_weight_streaming_budget", "-1"},
             {"nv_runtime_cache_path", runtime_cache_dir.string()}},
            &log_capture);
        RunOnce(*session);
        ExpectAutomaticBudgetApplied(log_capture);
    }

    ASSERT_GE(CountFilesInDirectory(runtime_cache_dir), 1u);
}

TEST(WeightStreamingTest, RuntimeCacheAllowsBudgetChangeAcrossSessions) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    const std::filesystem::path runtime_cache_dir =
        "nv_weight_streaming_runtime_cache_budget_change";
    RemoveDirectoryIfExists(runtime_cache_dir);

    std::vector<float> automatic_budget_outputs;
    {
        LogCapture log_capture;
        auto session = CreateSession(
            kModelPath,
            {{"nv_weight_streaming_budget", "-1"},
             {"nv_runtime_cache_path", runtime_cache_dir.string()}},
            &log_capture);
        automatic_budget_outputs = RunAndCollectFloatOutputs(*session);
        ExpectAutomaticBudgetApplied(log_capture);
    }

    ASSERT_TRUE(std::filesystem::exists(runtime_cache_dir));
    ASSERT_GE(CountFilesInDirectory(runtime_cache_dir), 1u);

    std::vector<float> minimum_vram_outputs;
    {
        LogCapture log_capture;
        auto session = CreateSession(
            kModelPath,
            {{"nv_weight_streaming_budget", "1"},
             {"nv_runtime_cache_path", runtime_cache_dir.string()}},
            &log_capture);
        minimum_vram_outputs = RunAndCollectFloatOutputs(*session);
        ExpectMinimumVramBudgetApplied(log_capture);
    }

    ASSERT_GE(CountFilesInDirectory(runtime_cache_dir), 1u);
    ExpectCloseOutputs(automatic_budget_outputs, minimum_vram_outputs,
                       "runtime-cache automatic-budget vs minimum-vram",
                       kIndependentSessionAbsTolerance, kIndependentSessionRelTolerance);
}
