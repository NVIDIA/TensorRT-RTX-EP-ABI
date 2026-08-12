// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Integration tests for nv_enable_profiling / nv_profiling_output_file provider options.
// All tests run a real ORT session and inspect the JSON file written by TrtRtxProfiler.

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

extern std::unique_ptr<Ort::Env> ort_env;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void AppendTrtRtxEp(Ort::SessionOptions& so, const std::unordered_map<std::string, std::string>& options = {})
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv;
    for (auto& [k, v] : options)
        kv.Add(k.c_str(), v.c_str());
    so.AppendExecutionProvider_V2(*ort_env, devices, kv);
}

// Read entire file into a string.
static std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "Cannot open: " << path;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Count non-overlapping occurrences of needle in haystack.
static size_t CountOccurrences(const std::string& haystack, const std::string& needle)
{
    size_t count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos)
    {
        ++count;
        pos += needle.size();
    }
    return count;
}

// Build a tiny model and return its filename.
static std::string MakeTestModel(const std::string& name)
{
    model_builder::CreateBaseModel(name, "test", {1, 3, 2});
    return name;
}

// Run one forward pass through a freshly created session.
static void RunOnce(Ort::Session& session)
{
    auto binding = generate_io_binding(session);
    Ort::RunOptions ro;
    session.Run(ro, binding);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// 1. No profile file created when nv_enable_profiling is not set.
TEST(TensorRTRTXEpTest_Profiler, DisabledByDefault)
{
    const std::string model = MakeTestModel("profiler_disabled.onnx");
    const std::string out = "profiler_disabled_out.json";
    clearFileIfExists(out);

    // Remove any pre-existing auto-timestamped profile files so we can detect new ones.
    std::regex ts_pattern(R"(trt_rtx_profile_\d{8}_\d{6}_\d+\.json)");
    for (auto& entry : std::filesystem::directory_iterator("."))
    {
        if (std::regex_match(entry.path().filename().string(), ts_pattern))
            std::filesystem::remove(entry.path());
    }

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    EXPECT_FALSE(std::filesystem::exists(out))
        << "Named profile file should not be created when profiling is disabled.";

    // Also verify no auto-timestamped file was created.
    size_t ts_files = 0;
    for (auto& entry : std::filesystem::directory_iterator("."))
    {
        if (std::regex_match(entry.path().filename().string(), ts_pattern))
            ++ts_files;
    }
    EXPECT_EQ(ts_files, 0u) << "Auto-timestamped profile file was created even though profiling was disabled.";
}

// 2. Profile file is created after Run() when nv_enable_profiling=1.
TEST(TensorRTRTXEpTest_Profiler, FileCreatedAfterRun)
{
    const std::string model = MakeTestModel("profiler_created.onnx");
    const std::string out = "profiler_created_out.json";
    clearFileIfExists(out);

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    ASSERT_TRUE(std::filesystem::exists(out)) << "Profile file was not created.";
    EXPECT_GT(std::filesystem::file_size(out), 0u) << "Profile file is empty.";
}

// 3. Custom output filename is respected.
TEST(TensorRTRTXEpTest_Profiler, CustomFilename)
{
    const std::string model = MakeTestModel("profiler_custom.onnx");
    const std::string out = "my_custom_profile_output.json";
    clearFileIfExists(out);

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    ASSERT_TRUE(std::filesystem::exists(out)) << "File not found at custom path: " << out;
}

// 4. Auto-generated filename matches expected pattern when no path is set.
TEST(TensorRTRTXEpTest_Profiler, AutoTimestampFilename)
{
    const std::string model = MakeTestModel("profiler_auto.onnx");

    // Remove any pre-existing auto-generated files to avoid false positives.
    for (auto& entry : std::filesystem::directory_iterator("."))
    {
        if (entry.path().filename().string().rfind("trt_rtx_profile_", 0) == 0)
            std::filesystem::remove(entry.path());
    }

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    // Expect exactly one file matching trt_rtx_profile_YYYYMMDD_HHMMSS.json.
    std::regex pattern(R"(trt_rtx_profile_\d{8}_\d{6}_\d+\.json)");
    size_t found = 0;
    for (auto& entry : std::filesystem::directory_iterator("."))
    {
        if (std::regex_match(entry.path().filename().string(), pattern))
        {
            ++found;
            std::filesystem::remove(entry.path());
        }
    }
    EXPECT_EQ(found, 1u) << "Expected exactly one auto-timestamped profile file.";
}

// 5. Output JSON contains at least one layer entry with the expected cat field.
TEST(TensorRTRTXEpTest_Profiler, LayerEntriesPresent)
{
    const std::string model = MakeTestModel("profiler_layers.onnx");
    const std::string out = "profiler_layers_out.json";
    clearFileIfExists(out);

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    ASSERT_TRUE(std::filesystem::exists(out));
    const std::string content = ReadFile(out);

    // Every layer entry must carry the category tag.
    EXPECT_GT(CountOccurrences(content, "\"nv::trt::layer\""), 0u)
        << "No layer entries with cat=nv::trt::layer found in:\n"
        << content;

    // Must be a JSON array.
    EXPECT_NE(content.find('['), std::string::npos);
    EXPECT_NE(content.find(']'), std::string::npos);
}

// 6. JSON contains a process_name metadata entry per subgraph (section count >= 1).
TEST(TensorRTRTXEpTest_Profiler, SectionCountNonZero)
{
    const std::string model = MakeTestModel("profiler_sections.onnx");
    const std::string out = "profiler_sections_out.json";
    clearFileIfExists(out);

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    ASSERT_TRUE(std::filesystem::exists(out));
    const std::string content = ReadFile(out);
    const size_t section_count = CountOccurrences(content, "\"process_name\"");

    EXPECT_GE(section_count, 1u) << "Expected at least one section (process_name metadata event).";
}

// 7. pid values in the JSON are non-negative integers (stable subgraph IDs).
TEST(TensorRTRTXEpTest_Profiler, PidFieldPresent)
{
    const std::string model = MakeTestModel("profiler_pid.onnx");
    const std::string out = "profiler_pid_out.json";
    clearFileIfExists(out);

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }

    ASSERT_TRUE(std::filesystem::exists(out));
    const std::string content = ReadFile(out);
    EXPECT_GT(CountOccurrences(content, "\"pid\":"), 0u) << "No pid field found in profile output.";
}

// 8. Second Run() produces tid=1 for the same subgraph (tid increments across runs).
TEST(TensorRTRTXEpTest_Profiler, TidIncrementsAcrossRuns)
{
    const std::string model = MakeTestModel("profiler_tid.onnx");
    const std::string out = "profiler_tid_out.json";
    clearFileIfExists(out);

    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);

        // Run 1 — tid=0 accumulated in sections_.
        RunOnce(session);
        // Run 2 — tid=1 accumulated; file now contains both tid=0 and tid=1.
        RunOnce(session);
    }

    ASSERT_TRUE(std::filesystem::exists(out));
    const std::string content = ReadFile(out);

    // File must contain both tid=0 (run 1) and tid=1 (run 2) since sections accumulate.
    EXPECT_GT(CountOccurrences(content, "\"tid\": 0"), 0u) << "Expected tid=0 (run 1) in file. Contents:\n" << content;
    EXPECT_GT(CountOccurrences(content, "\"tid\": 1"), 0u) << "Expected tid=1 (run 2) in file. Contents:\n" << content;
}

// 9. Enabling profiling together with CUDA graphs does not crash or error;
//    the session runs successfully (CUDA graph is silently disabled).
TEST(TensorRTRTXEpTest_Profiler, CudaGraphConflictNoError)
{
    const std::string model = MakeTestModel("profiler_cuda_graph.onnx");
    const std::string out = "profiler_cuda_graph_out.json";
    clearFileIfExists(out);

    EXPECT_NO_THROW({
        Ort::SessionOptions so;
        AppendTrtRtxEp(so,
                       {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}, {"enable_cuda_graph", "1"}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
    }) << "Session should not throw when both profiling and CUDA graph are requested.";

    // The profile file must still be created — profiling must have stayed enabled.
    EXPECT_TRUE(std::filesystem::exists(out)) << "Profile file not created; profiling may have been silently disabled.";
}

// 10. Bad output file path: session should not crash; a stderr warning is emitted.
//     sections_ are cleared on open failure to prevent unbounded memory growth.
TEST(TensorRTRTXEpTest_Profiler, BadOutputFilePath)
{
    const std::string model = MakeTestModel("profiler_bad_path.onnx");
    const std::string bad_path = "Z:\\nonexistent\\dir\\profile.json";
    const std::string good_path = "profiler_bad_path_recovery.json";
    clearFileIfExists(good_path);

    EXPECT_NO_THROW({
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", bad_path}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        RunOnce(session);
        // No file should appear at the bad path (can't create it) — no crash expected.
        EXPECT_FALSE(std::filesystem::exists(bad_path));
    }) << "Session must not throw when the profiling output path is invalid.";
}

// 11. After N runs, the JSON contains N sections with tid values 0..N-1.
//     FlushToFile accumulates sections across runs so FTK can do N×M 1:1 mapping.
TEST(TensorRTRTXEpTest_Profiler, MultipleRunsProduceDistinctTids)
{
    const std::string model = MakeTestModel("profiler_multi_run.onnx");
    const std::string out = "profiler_multi_run_out.json";
    clearFileIfExists(out);

    constexpr int kRuns = 3;
    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_enable_profiling", "1"}, {"nv_profiling_output_file", out}});
        Ort::Session session(*ort_env, toOrtString(model).c_str(), so);
        for (int i = 0; i < kRuns; ++i)
        {
            RunOnce(session);
        }
    }

    ASSERT_TRUE(std::filesystem::exists(out));
    const std::string content = ReadFile(out);

    // Each run produces one section; file must contain all kRuns sections.
    // Verify tid=0, tid=1, tid=2 are all present.
    for (int i = 0; i < kRuns; ++i)
    {
        const std::string tid_str = "\"tid\": " + std::to_string(i);
        EXPECT_GT(CountOccurrences(content, tid_str), 0u)
            << "Expected tid=" << i << " in profile after " << kRuns << " runs.";
    }
}
