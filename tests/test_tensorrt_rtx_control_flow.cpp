// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "test_tensorrt_rtx_control_flow_models.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void AppendTrtRtxEp(Ort::SessionOptions& session_options,
                    const std::unordered_map<std::string, std::string>& options = {})
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs provider_options;
    for (const auto& [name, value] : options)
        provider_options.Add(name.c_str(), value.c_str());
    session_options.AppendExecutionProvider_V2(*ort_env, devices, provider_options);
}

Ort::SessionOptions MakeSessionOptions(const std::unordered_map<std::string, std::string>& provider_options = {},
                                       bool disable_cpu_fallback = false,
                                       const std::filesystem::path& profile_prefix = {})
{
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    if (disable_cpu_fallback)
        session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    AppendTrtRtxEp(session_options, provider_options);
    if (!profile_prefix.empty())
        session_options.EnableProfiling(toOrtString(profile_prefix.string()).c_str());
    return session_options;
}

std::string EndAndReadProfile(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto path = session.EndProfilingAllocated(allocator);
    EXPECT_NE(path.get(), nullptr);
    return path ? ReadTextFile(std::filesystem::path{path.get()}) : std::string{};
}

void ExpectFloatTensor(const Ort::Value& value, const std::vector<int64_t>& expected_shape,
                       const std::vector<float>& expected_values, float tolerance = 1e-6f)
{
    ASSERT_TRUE(value.IsTensor());
    const auto info = value.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(info.GetShape(), expected_shape);
    ASSERT_EQ(info.GetElementCount(), expected_values.size());
    const float* actual = value.GetTensorData<float>();
    for (size_t i = 0; i < expected_values.size(); ++i)
        EXPECT_NEAR(actual[i], expected_values[i], tolerance) << "Mismatch at output index " << i;
}

void ExpectProfileUsesProvider(const std::string& profile, const std::string& provider_name)
{
    EXPECT_NE(profile.find("\"provider\" : \"" + provider_name + "\""), std::string::npos);
}

void ExpectProfileUsesTrtWithoutCpuFallback(const std::string& profile)
{
    ExpectProfileUsesProvider(profile, kEpName);
    EXPECT_EQ(profile.find("\"provider\" : \"CPUExecutionProvider\""), std::string::npos);
}

void ExpectProfileNodeUsesProvider(const std::string& profile, const std::string& node_name,
                                   const std::string& provider_name)
{
    const std::string event_name = "\"name\" :\"" + node_name + "_kernel_time\"";
    const size_t event_position = profile.find(event_name);
    ASSERT_NE(event_position, std::string::npos) << "Missing profile event for node " << node_name;
    const size_t event_end = profile.find('\n', event_position);
    const std::string event = profile.substr(event_position, event_end - event_position);
    EXPECT_NE(event.find("\"provider\" : \"" + provider_name + "\""), std::string::npos)
        << "Unexpected provider for node " << node_name;
}

std::vector<Ort::Value> RunIfModel(Ort::Session& session, const std::vector<float>& input, bool condition,
                                   const std::vector<int64_t>& shape = {2, 3})
{
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(), shape.data(),
                                                   shape.size());
    Ort::Value cond = Ort::Value::CreateTensor<bool>(memory, &condition, 1, nullptr, 0);
    std::array<Ort::Value, 2> values = {std::move(x), std::move(cond)};
    const char* input_names[] = {"X", "cond"};
    const char* output_names[] = {"Y"};
    return session.Run(Ort::RunOptions{nullptr}, input_names, values.data(), values.size(), output_names, 1);
}

}  // namespace

TEST(TensorRTRTXControlFlowTest, ParserSupportedIfKeepsOriginalFullTrtPath)
{
    const std::string model_path = "control_flow_simple_if.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateSimpleIfModel(model_path);
    auto options = MakeSessionOptions({}, true);
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
    const std::vector<float> input = {1.0f, -2.0f, 3.0f, 0.25f, -0.5f, 2.5f};
    auto then_outputs = RunIfModel(session, input, true);
    ExpectFloatTensor(then_outputs[0], {2, 3}, {1.0f, 2.0f, 3.0f, 0.25f, 0.5f, 2.5f});
    auto else_outputs = RunIfModel(session, input, false);
    ExpectFloatTensor(else_outputs[0], {2, 3}, {-1.0f, 2.0f, -3.0f, -0.25f, 0.5f, -2.5f});
}

TEST(TensorRTRTXControlFlowTest, ParentProducedShapeIfRetriesWithIntactGraphContext)
{
    const std::vector<float> input = {1.0f, -2.0f, 3.0f, 0.25f, -0.5f, 2.5f};
    for (const bool take_then_branch : {false, true})
    {
        const std::string model_path =
            take_then_branch ? "control_flow_parent_context_then.onnx" : "control_flow_parent_context_else.onnx";
        clearFileIfExists(model_path);
        control_flow_models::CreateParentProducedShapeIfModel(model_path, take_then_branch);
        auto options = MakeSessionOptions({}, true,
                                          take_then_branch ? "control_flow_parent_context_then_profile"
                                                           : "control_flow_parent_context_else_profile");
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> shape = {2, 3};
        Ort::Value x = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(),
                                                       shape.data(), shape.size());
        const char* input_names[] = {"X"};
        const char* output_names[] = {"Y"};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &x, 1, output_names, 1);
        ExpectFloatTensor(outputs[0], take_then_branch ? std::vector<int64_t>{1, 2, 3} : std::vector<int64_t>{2, 3},
                          input);
        const std::string profile = EndAndReadProfile(session);
        ExpectProfileUsesTrtWithoutCpuFallback(profile);
    }
}

TEST(TensorRTRTXControlFlowTest, ExcludedBranchUsesOriginalMixedFallbackPath)
{
    const std::vector<float> input = {1.0f, -2.0f, 3.0f, 0.25f, -0.5f, 2.5f};
    for (const bool exclude_then_branch : {false, true})
    {
        const std::string suffix = exclude_then_branch ? "then" : "else";
        const std::string model_path = "control_flow_excluded_" + suffix + "_branch_if.onnx";
        clearFileIfExists(model_path);
        control_flow_models::CreateExcludedBranchIfModel(model_path, exclude_then_branch);
        auto options = MakeSessionOptions({{"nv_op_types_to_exclude", "Sin"}}, false,
                                          "control_flow_excluded_" + suffix + "_branch_profile");
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);

        // Execute the supported sibling branch. The If remains on CPU, while
        // the supported operation inside the selected branch remains on TRT.
        const bool run_then_branch = !exclude_then_branch;
        auto outputs = RunIfModel(session, input, run_then_branch);
        if (exclude_then_branch)
            ExpectFloatTensor(outputs[0], {2, 3}, {-1.0f, 2.0f, -3.0f, -0.25f, 0.5f, -2.5f});
        else
            ExpectFloatTensor(outputs[0], {2, 3}, {1.0f, 2.0f, 3.0f, 0.25f, 0.5f, 2.5f});

        const std::string profile = EndAndReadProfile(session);
        ExpectProfileUsesProvider(profile, "CPUExecutionProvider");
        ExpectProfileUsesProvider(profile, kEpName);
    }
}

TEST(TensorRTRTXControlFlowTest, UnsupportedBranchDataTypeUsesOriginalMixedFallbackPath)
{
    const std::string model_path = "control_flow_unsupported_dtype_branch_if.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateUnsupportedDataTypeBranchIfModel(model_path);
    auto options = MakeSessionOptions({}, false, "control_flow_unsupported_dtype_branch_profile");
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
    const std::vector<float> input = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
    auto outputs = RunIfModel(session, input, true);
    ExpectFloatTensor(outputs[0], {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    const std::string profile = EndAndReadProfile(session);
    ExpectProfileUsesProvider(profile, "CPUExecutionProvider");
    ExpectProfileUsesProvider(profile, kEpName);
}

TEST(TensorRTRTXControlFlowTest, UnsupportedOperatorUsesOriginalMixedFallbackPath)
{
    const std::string model_path = "control_flow_parser_rejected_if.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateParserRejectedIfModel(model_path);
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    for (const bool run_det_branch : {true, false})
    {
        SCOPED_TRACE(run_det_branch ? "then branch: Det" : "else branch: ReduceSum");
        const std::string suffix = run_det_branch ? "det" : "reduce_sum";
        auto options = MakeSessionOptions({}, false, "control_flow_parent_retry_failure_" + suffix + "_profile");
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
        auto outputs = RunIfModel(session, input, run_det_branch, {2, 2});
        ExpectFloatTensor(outputs[0], {}, {run_det_branch ? -2.0f : 10.0f});
        const std::string profile = EndAndReadProfile(session);
        ExpectProfileUsesProvider(profile, "CPUExecutionProvider");
        if (run_det_branch)
            EXPECT_EQ(profile.find("\"provider\" : \"" + std::string(kEpName) + "\""), std::string::npos);
        else
            ExpectProfileUsesProvider(profile, kEpName);
    }
}

TEST(TensorRTRTXControlFlowTest, NestedExcludedIfIsCaughtByRecursiveCoarseCheck)
{
    const std::string model_path = "control_flow_nested_excluded_if.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateNestedExcludedIfModel(model_path);
    const std::vector<float> input = {1.0f, -2.0f, 3.0f, 0.25f, -0.5f, 2.5f};
    for (const bool inner_condition_value : {true, false})
    {
        SCOPED_TRACE(inner_condition_value ? "selected supported nested Abs branch"
                                           : "selected excluded nested Sin branch");
        const std::string suffix = inner_condition_value ? "supported" : "excluded";
        auto options = MakeSessionOptions({{"nv_op_types_to_exclude", "Sin"}}, false,
                                          "control_flow_nested_" + suffix + "_profile");
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> shape = {2, 3};
        Ort::Value x = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(),
                                                       shape.data(), shape.size());
        bool outer_condition = true;
        bool inner_condition = inner_condition_value;
        Ort::Value outer = Ort::Value::CreateTensor<bool>(memory, &outer_condition, 1, nullptr, 0);
        Ort::Value inner = Ort::Value::CreateTensor<bool>(memory, &inner_condition, 1, nullptr, 0);
        std::array<Ort::Value, 3> values = {std::move(x), std::move(outer), std::move(inner)};
        const char* input_names[] = {"X", "outer_cond", "inner_cond"};
        const char* output_names[] = {"Y"};
        auto outputs =
            session.Run(Ort::RunOptions{nullptr}, input_names, values.data(), values.size(), output_names, 1);
        std::vector<float> expected(input.size());
        std::transform(input.begin(), input.end(), expected.begin(),
                       [inner_condition_value](float value)
                       {
                           return inner_condition_value ? std::abs(value) : std::sin(value);
                       });
        ExpectFloatTensor(outputs[0], {2, 3}, expected);
        const std::string profile = EndAndReadProfile(session);
        ExpectProfileNodeUsesProvider(profile, "outer_if", "CPUExecutionProvider");
        ExpectProfileNodeUsesProvider(profile, "inner_if", "CPUExecutionProvider");
        if (inner_condition_value)
        {
            // This branch deliberately needs parent producer context. The nested Sin exclusion must be found
            // recursively before an intact-parent retry can absorb this branch into a TRT control-flow partition.
            ExpectProfileNodeUsesProvider(profile, "inner_abs", "CPUExecutionProvider");
        }
        else
        {
            ExpectProfileNodeUsesProvider(profile, "nested_excluded_sin", "CPUExecutionProvider");
        }
    }
}

TEST(TensorRTRTXControlFlowTest, LoopSupportedBodyRunsWithoutCpuFallback)
{
    const std::string model_path = "control_flow_loop_supported.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateLoopModel(model_path, false);
    auto options = MakeSessionOptions({}, true);
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
    float input = 2.0f;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(memory, &input, 1, nullptr, 0);
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &x, 1, output_names, 1);
    ExpectFloatTensor(outputs[0], {}, {5.0f});
}

TEST(TensorRTRTXControlFlowTest, ParentProducedShapeLoopRetriesWithIntactGraphContext)
{
    const std::string model_path = "control_flow_parent_context_loop.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateParentProducedShapeLoopModel(model_path);
    auto options = MakeSessionOptions({}, true, "control_flow_parent_context_loop_profile");
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);

    std::vector<float> input = {1.0f, -2.0f, 3.0f, 0.25f, -0.5f, 2.5f};
    const std::vector<int64_t> shape = {2, 3};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &x, 1, output_names, 1);
    ExpectFloatTensor(outputs[0], {2, 3}, {4.0f, 1.0f, 6.0f, 3.25f, 2.5f, 5.5f});
    const std::string profile = EndAndReadProfile(session);
    ExpectProfileUsesTrtWithoutCpuFallback(profile);
}

TEST(TensorRTRTXControlFlowTest, LoopExcludedBodyUsesSafeFallback)
{
    const std::string model_path = "control_flow_loop_excluded.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateLoopModel(model_path, true);
    auto options = MakeSessionOptions({{"nv_op_types_to_exclude", "Sin"}}, false, "control_flow_loop_excluded_profile");
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
    float input = 0.25f;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(memory, &input, 1, nullptr, 0);
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &x, 1, output_names, 1);
    const float expected = std::sin(std::sin(std::sin(input + 1.0f) + 1.0f) + 1.0f);
    ExpectFloatTensor(outputs[0], {}, {expected});
    const std::string profile = EndAndReadProfile(session);
    ExpectProfileUsesProvider(profile, "CPUExecutionProvider");
    ExpectProfileUsesProvider(profile, kEpName);
}

TEST(TensorRTRTXControlFlowTest, ScanPreservesStateAndSequenceSemantics)
{
    const std::string model_path = "control_flow_scan.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateScanModel(model_path);
    auto options = MakeSessionOptions({}, true);
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
    float initial_state = 1.0f;
    std::vector<float> scan_input = {2.0f, 3.0f, 4.0f};
    const std::vector<int64_t> scan_shape = {3};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value state = Ort::Value::CreateTensor<float>(memory, &initial_state, 1, nullptr, 0);
    Ort::Value sequence = Ort::Value::CreateTensor<float>(memory, scan_input.data(), scan_input.size(),
                                                          scan_shape.data(), scan_shape.size());
    std::array<Ort::Value, 2> values = {std::move(state), std::move(sequence)};
    const char* input_names[] = {"initial_state", "scan_input"};
    const char* output_names[] = {"final_state", "scan_output"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, values.data(), values.size(), output_names, 2);
    ASSERT_EQ(outputs.size(), 2u);
    ExpectFloatTensor(outputs[0], {}, {10.0f});
    ExpectFloatTensor(outputs[1], {3}, {3.0f, 6.0f, 10.0f});
}

TEST(TensorRTRTXControlFlowTest, ParentProducedShapeScanRetriesWithIntactGraphContext)
{
    const std::string model_path = "control_flow_parent_context_scan.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateParentProducedShapeScanModel(model_path);
    auto options = MakeSessionOptions({}, true, "control_flow_parent_context_scan_profile");
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);

    std::vector<float> initial_state(6, 0.0f);
    std::vector<float> scan_input = {
        1.0f,  2.0f,  3.0f,  4.0f,   5.0f,   6.0f,   10.0f,  20.0f,  30.0f,
        40.0f, 50.0f, 60.0f, 100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f,
    };
    const std::vector<int64_t> state_shape = {2, 3};
    const std::vector<int64_t> scan_shape = {3, 2, 3};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value state = Ort::Value::CreateTensor<float>(memory, initial_state.data(), initial_state.size(),
                                                       state_shape.data(), state_shape.size());
    Ort::Value sequence = Ort::Value::CreateTensor<float>(memory, scan_input.data(), scan_input.size(),
                                                          scan_shape.data(), scan_shape.size());
    std::array<Ort::Value, 2> values = {std::move(state), std::move(sequence)};
    const char* input_names[] = {"initial_state", "scan_input"};
    const char* output_names[] = {"final_state", "scan_output"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, values.data(), values.size(), output_names, 2);

    ExpectFloatTensor(outputs[0], {2, 3}, {111.0f, 222.0f, 333.0f, 444.0f, 555.0f, 666.0f});
    ExpectFloatTensor(outputs[1], {3, 2, 3},
                      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 111.0f, 222.0f,
                       333.0f, 444.0f, 555.0f, 666.0f});
    const std::string profile = EndAndReadProfile(session);
    ExpectProfileUsesTrtWithoutCpuFallback(profile);
}

TEST(TensorRTRTXControlFlowTest, ScanExcludedBodyUsesSafeMixedFallback)
{
    const std::string model_path = "control_flow_scan_excluded.onnx";
    clearFileIfExists(model_path);
    control_flow_models::CreateScanModel(model_path, true);
    auto options = MakeSessionOptions({{"nv_op_types_to_exclude", "Sin"}}, false, "control_flow_scan_excluded_profile");
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), options);
    float initial_state = 1.0f;
    std::vector<float> scan_input = {2.0f, 3.0f, 4.0f};
    const std::vector<int64_t> scan_shape = {3};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value state = Ort::Value::CreateTensor<float>(memory, &initial_state, 1, nullptr, 0);
    Ort::Value sequence = Ort::Value::CreateTensor<float>(memory, scan_input.data(), scan_input.size(),
                                                          scan_shape.data(), scan_shape.size());
    std::array<Ort::Value, 2> values = {std::move(state), std::move(sequence)};
    const char* input_names[] = {"initial_state", "scan_input"};
    const char* output_names[] = {"final_state", "scan_output"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, values.data(), values.size(), output_names, 2);
    ASSERT_EQ(outputs.size(), 2u);
    const float state_1 = std::sin(initial_state + scan_input[0]);
    const float state_2 = std::sin(state_1 + scan_input[1]);
    const float state_3 = std::sin(state_2 + scan_input[2]);
    ExpectFloatTensor(outputs[0], {}, {state_3});
    ExpectFloatTensor(outputs[1], {3}, {state_1, state_2, state_3});
    const std::string profile = EndAndReadProfile(session);
    ExpectProfileUsesProvider(profile, "CPUExecutionProvider");
    ExpectProfileUsesProvider(profile, kEpName);
}
