// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from onnxruntime/test/providers/nv_tensorrt_rtx/nv_options_test.cc
// Uses only public ORT SDK APIs.

#include "tensorrt_rtx_execution_provider_info.h"

#include <cuda.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

extern std::unique_ptr<Ort::Env> ort_env;

static constexpr const char* kRuntimeCacheReadFailureLogMarker = "configured cache could not be read";
static constexpr const char* kRuntimeCacheUnsafeNameLogMarker = "model-supplied node/partition name is unsafe";

//!
//! \brief Records whether ORT emitted the expected runtime-cache read-failure diagnostic.
//!
//! \param param Pointer to the std::atomic_bool updated when the diagnostic is observed.
//! \param message ORT log message to inspect.
//!
//! The remaining ORT callback fields are intentionally ignored.
//!
static void ORT_API_CALL CaptureRuntimeCacheReadFailureLog(void* param, OrtLoggingLevel /*severity*/,
                                                           const char* /*category*/, const char* /*logid*/,
                                                           const char* /*code_location*/, const char* message) noexcept
{
    if (param != nullptr && message != nullptr && std::strstr(message, kRuntimeCacheReadFailureLogMarker) != nullptr)
    {
        static_cast<std::atomic_bool*>(param)->store(true, std::memory_order_relaxed);
    }
}

//!
//! \brief Records whether ORT disabled runtime caching for an unsafe model-derived filename.
//!
//! \param param Pointer to the std::atomic_bool updated when the diagnostic is observed.
//! \param message ORT log message to inspect.
//!
//! The remaining ORT callback fields are intentionally ignored.
//!
static void ORT_API_CALL CaptureRuntimeCacheUnsafeNameLog(void* param, OrtLoggingLevel /*severity*/,
                                                          const char* /*category*/, const char* /*logid*/,
                                                          const char* /*code_location*/, const char* message) noexcept
{
    if (param != nullptr && message != nullptr && std::strstr(message, kRuntimeCacheUnsafeNameLogMarker) != nullptr)
    {
        static_cast<std::atomic_bool*>(param)->store(true, std::memory_order_relaxed);
    }
}

// Helper: append TRT RTX EP to session options.
static void AppendTrtRtxEp(Ort::SessionOptions& so, const std::unordered_map<std::string, std::string>& options = {})
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv_options;
    for (auto& [k, v] : options)
    {
        kv_options.Add(k.c_str(), v.c_str());
    }
    so.AppendExecutionProvider_V2(*ort_env, devices, kv_options);
}

static size_t countFilesInDirectory(const std::string& dir_path)
{
    return static_cast<size_t>(
        std::distance(std::filesystem::directory_iterator(dir_path), std::filesystem::directory_iterator{}));
}

//!
//! \brief Captures the CUDA context state surrounding session destruction on a fresh host thread.
//!
struct ThreadContextSnapshot
{
    CUresult before_status = CUDA_ERROR_UNKNOWN;
    CUcontext before_context = nullptr;
    CUresult after_status = CUDA_ERROR_UNKNOWN;
    CUcontext after_context = nullptr;
};

//!
//! \brief Destroys an ORT session on a fresh host thread and observes CUDA context restoration.
//!
//! \param session Session whose ownership is transferred to the worker thread for destruction.
//! \return CUDA context query results from immediately before and after synchronous session teardown.
//!
static ThreadContextSnapshot DestroySessionOnFreshThread(std::unique_ptr<Ort::Session> session)
{
    ThreadContextSnapshot snapshot{};
    std::thread destroy_session(
        [&snapshot, session = std::move(session)]() mutable
        {
            snapshot.before_status = cuCtxGetCurrent(&snapshot.before_context);
            session.reset();
            snapshot.after_status = cuCtxGetCurrent(&snapshot.after_context);
        });
    destroy_session.join();
    return snapshot;
}

//!
//! \brief Runs a cache-enabled session and verifies that fresh-thread teardown does not leave a CUDA context current.
//!
//! \param model_name Model or EPContext model to load.
//! \param runtime_cache_dir Directory containing the runtime cache.
//! \param read_failure_logged Optional flag set when an expected cache read failure is logged.
//!
static void RunCachedSessionAndDestroyOnFreshThread(const std::string& model_name, const std::string& runtime_cache_dir,
                                                    std::atomic_bool* read_failure_logged = nullptr)
{
    Ort::SessionOptions so;
    AppendTrtRtxEp(so, {{"nv_runtime_cache_path", runtime_cache_dir}});
    if (read_failure_logged != nullptr)
    {
        so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
        Ort::ThrowOnError(
            Ort::GetApi().SetUserLoggingFunction(so, CaptureRuntimeCacheReadFailureLog, read_failure_logged));
    }

    std::unique_ptr<Ort::Session> session;
    ASSERT_NO_THROW(session = std::make_unique<Ort::Session>(*ort_env, toOrtString(model_name).c_str(), so));
    ASSERT_NO_THROW(run_with_cpu_bindings(*session, 1));

    const auto context_snapshot = DestroySessionOnFreshThread(std::move(session));
    ASSERT_EQ(context_snapshot.before_status, CUDA_SUCCESS);
    ASSERT_EQ(context_snapshot.before_context, nullptr);
    ASSERT_EQ(context_snapshot.after_status, CUDA_SUCCESS);
    ASSERT_EQ(context_snapshot.after_context, nullptr);
}

//!
//! \brief Verifies cache creation, reload, corruption recovery, I/O failure handling, and context-safe teardown.
//!
TEST(TensorRTRTXEpTest_Options, RuntimeCaching)
{
    const std::string model_name = "nv_execution_provider_runtime_caching.onnx";
    const std::string model_name_ctx = "nv_execution_provider_runtime_caching_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    const std::string runtime_cache_dir = "./runtime_cache/";
    if (std::filesystem::exists(runtime_cache_dir))
    {
        std::filesystem::remove_all(runtime_cache_dir);
    }

    model_builder::CreateBaseModel(model_name, "test", {1, 3, 2});

    // AOT: compile with runtime cache enabled
    {
        Ort::SessionOptions so;
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx.c_str());
        AppendTrtRtxEp(so, {{"nv_runtime_cache_path", runtime_cache_dir}});
        auto session = std::make_unique<Ort::Session>(*ort_env, toOrtString(model_name).c_str(), so);

        ASSERT_NO_THROW(run_with_cpu_bindings(*session, 1));

        // A fresh host thread has no current CUDA context. This is the shutdown pattern that previously caused
        // IRuntimeCache::serialize() to return null in TensorRT RTX 1.6.
        const auto context_snapshot = DestroySessionOnFreshThread(std::move(session));
        ASSERT_EQ(context_snapshot.before_status, CUDA_SUCCESS);
        ASSERT_EQ(context_snapshot.before_context, nullptr);
        ASSERT_EQ(context_snapshot.after_status, CUDA_SUCCESS);
        ASSERT_EQ(context_snapshot.after_context, nullptr);
    }
    // Cache remains valid after the final teardown save.
    ASSERT_TRUE(std::filesystem::exists(runtime_cache_dir));
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir), 1u);
    ASSERT_GT(std::filesystem::file_size(std::filesystem::directory_iterator(runtime_cache_dir)->path()), 0u);

    // Run with the existing cache through the EPContext path, then destroy that context on a context-less host thread.
    ASSERT_NO_FATAL_FAILURE(RunCachedSessionAndDestroyOnFreshThread(model_name_ctx, runtime_cache_dir));
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir), 1u);
    ASSERT_GT(std::filesystem::file_size(std::filesystem::directory_iterator(runtime_cache_dir)->path()), 0u);

    const std::filesystem::path runtime_cache_file = std::filesystem::directory_iterator(runtime_cache_dir)->path();

#ifdef _WIN32
    // An existing cache can be temporarily unreadable on Windows (for example, an exclusive writer or scanner).
    // Runtime caching is optional, so this must disable caching for the context instead of throwing through the
    // noexcept Compile callback and terminating the host process.
    const auto cache_size_before_read_failure = std::filesystem::file_size(runtime_cache_file);
    {
        HANDLE raw_cache_lock = ::CreateFileW(runtime_cache_file.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
                                              FILE_ATTRIBUTE_NORMAL, nullptr);
        ASSERT_NE(raw_cache_lock, INVALID_HANDLE_VALUE) << "GetLastError=" << ::GetLastError();
        std::unique_ptr<void, decltype(&::CloseHandle)> cache_lock(raw_cache_lock, &::CloseHandle);

        std::atomic_bool read_failure_logged{false};
        ASSERT_NO_FATAL_FAILURE(
            RunCachedSessionAndDestroyOnFreshThread(model_name_ctx, runtime_cache_dir, &read_failure_logged));
        EXPECT_TRUE(read_failure_logged.load(std::memory_order_relaxed));
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(runtime_cache_file));
    ASSERT_EQ(std::filesystem::file_size(runtime_cache_file), cache_size_before_read_failure);
#endif

    // Exercise the rejected-deserialize path with non-empty corrupt data. TRT-RTX 1.5 does not guarantee that a
    // rejected deserialize leaves the cache unchanged, so the provider explicitly resets it before reuse.
    constexpr char invalid_cache_data[] = "not a TensorRT RTX runtime cache";
    {
        std::ofstream invalid_cache(runtime_cache_file, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(invalid_cache);
        invalid_cache.write(invalid_cache_data, sizeof(invalid_cache_data));
        ASSERT_TRUE(invalid_cache);
    }
    ASSERT_EQ(std::filesystem::file_size(runtime_cache_file), sizeof(invalid_cache_data));

    ASSERT_NO_FATAL_FAILURE(RunCachedSessionAndDestroyOnFreshThread(model_name_ctx, runtime_cache_dir));
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir), 1u);
    ASSERT_GT(std::filesystem::file_size(runtime_cache_file), sizeof(invalid_cache_data));

    // Confirm that the rewritten cache can be consumed again.
    ASSERT_NO_FATAL_FAILURE(RunCachedSessionAndDestroyOnFreshThread(model_name_ctx, runtime_cache_dir));
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir), 1u);
    ASSERT_GT(std::filesystem::file_size(runtime_cache_file), sizeof(invalid_cache_data));

    std::filesystem::remove_all(runtime_cache_dir);

    // Create new cache in different directory
    {
        const std::string new_cache_dir = "./runtime_cache_new/";
        if (std::filesystem::exists(new_cache_dir))
        {
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

//!
//! \brief Verifies that an unsafe EPContext partition name disables only the optional runtime cache.
//!
TEST(TensorRTRTXEpTest_Options, RuntimeCacheUnsafePartitionNameDisablesCaching)
{
    const std::string model_name = "nv_execution_provider_runtime_cache_unsafe_name.onnx";
    const std::string model_name_ctx = "nv_execution_provider_runtime_cache_unsafe_name_ctx.onnx";
    const std::string unsafe_model_name_ctx = "nv_execution_provider_runtime_cache_unsafe_name_modified_ctx.onnx";
    const std::filesystem::path runtime_cache_dir = "./runtime_cache_unsafe_name";
    clearFileIfExists(model_name_ctx);
    clearFileIfExists(unsafe_model_name_ctx);
    std::filesystem::remove_all(runtime_cache_dir);

    model_builder::CreateBaseModel(model_name, "runtime_cache_unsafe_name", {1, 3, 2});
    {
        Ort::SessionOptions so;
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx.c_str());
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);
        ASSERT_NO_THROW(run_with_cpu_bindings(session, 1));
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(model_name_ctx));

    onnx::ModelProto context_model;
    {
        std::ifstream input(model_name_ctx, std::ios::binary);
        ASSERT_TRUE(input);
        ASSERT_TRUE(context_model.ParseFromIstream(&input));
    }

    bool partition_name_updated = false;
    for (auto& node : *context_model.mutable_graph()->mutable_node())
    {
        if (node.op_type() != "EPContext")
        {
            continue;
        }
        for (auto& attribute : *node.mutable_attribute())
        {
            if (attribute.name() == "partition_name")
            {
                attribute.set_s("..");
                partition_name_updated = true;
            }
        }
    }
    ASSERT_TRUE(partition_name_updated);
    model_builder::SaveModel(context_model, unsafe_model_name_ctx);
    std::filesystem::create_directories(runtime_cache_dir);

    std::atomic_bool unsafe_name_logged{false};
    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so, {{"nv_runtime_cache_path", runtime_cache_dir.string()}});
        so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
        Ort::ThrowOnError(
            Ort::GetApi().SetUserLoggingFunction(so, CaptureRuntimeCacheUnsafeNameLog, &unsafe_name_logged));

        std::unique_ptr<Ort::Session> session;
        ASSERT_NO_THROW(session =
                            std::make_unique<Ort::Session>(*ort_env, toOrtString(unsafe_model_name_ctx).c_str(), so));
        ASSERT_NO_THROW(run_with_cpu_bindings(*session, 1));
    }

    EXPECT_TRUE(unsafe_name_logged.load(std::memory_order_relaxed));
    EXPECT_EQ(countFilesInDirectory(runtime_cache_dir.string()), 0u);

    std::filesystem::remove_all(runtime_cache_dir);
    clearFileIfExists(unsafe_model_name_ctx);
}

//!
//! \brief Verifies that the runtime configuration remains valid through execution-context destruction.
//!
TEST(TensorRTRTXEpTest_Options, RuntimeConfigOutlivesExecutionContext)
{
    const std::string model_name = "nv_execution_provider_runtime_config_lifetime.onnx";
    model_builder::CreateBaseModel(model_name, "test", {1, -1, -1});

    Ort::SessionOptions so;
    AppendTrtRtxEp(so);
    auto session = std::make_unique<Ort::Session>(*ort_env, toOrtString(model_name).c_str(), so);

    {
        auto io_binding = generate_io_binding(*session, {{"X", {1, 5, 5}}, {"Y", {1, 5, 1}}});
        Ort::RunOptions run_options;
        session->Run(run_options, io_binding);
    }

    // A second dynamic shape exercises post-creation specialization paths that consult RuntimeConfig.
    {
        auto io_binding = generate_io_binding(*session, {{"X", {1, 7, 3}}, {"Y", {1, 7, 1}}});
        Ort::RunOptions run_options;
        session->Run(run_options, io_binding);
    }

    // TensorRT also accesses RuntimeConfig during execution-context destruction. A separate thread makes that teardown
    // boundary explicit and mirrors the host's real shutdown path.
    const auto context_snapshot = DestroySessionOnFreshThread(std::move(session));
    ASSERT_EQ(context_snapshot.before_status, CUDA_SUCCESS);
    ASSERT_EQ(context_snapshot.before_context, nullptr);
    ASSERT_EQ(context_snapshot.after_status, CUDA_SUCCESS);
    ASSERT_EQ(context_snapshot.after_context, nullptr);
}

//!
//! \brief Verifies that a final runtime-cache write failure cannot escape session teardown.
//!
TEST(TensorRTRTXEpTest_Options, RuntimeCacheWriteFailureDoesNotEscapeTeardown)
{
    const std::string model_name = "nv_execution_provider_runtime_cache_write_failure.onnx";
    const std::filesystem::path runtime_cache_dir = "./runtime_cache_write_failure";
    std::filesystem::remove_all(runtime_cache_dir);

    model_builder::CreateBaseModel(model_name, "test", {1, 3, 2});

    Ort::SessionOptions so;
    AppendTrtRtxEp(so, {{"nv_runtime_cache_path", runtime_cache_dir.string()}});
    auto session = std::make_unique<Ort::Session>(*ort_env, toOrtString(model_name).c_str(), so);

    ASSERT_NO_THROW(run_with_cpu_bindings(*session, 1));
    ASSERT_TRUE(std::filesystem::is_directory(runtime_cache_dir));
    ASSERT_EQ(countFilesInDirectory(runtime_cache_dir.string()), 0u);

    // Replace the cache directory with a regular file. The final save must log and continue: a persistence failure is
    // not allowed to terminate the host from a unique_ptr destructor.
    std::filesystem::remove_all(runtime_cache_dir);
    {
        std::ofstream blocker(runtime_cache_dir, std::ios::binary);
        ASSERT_TRUE(blocker);
        blocker << "block final cache write";
    }

    testing::internal::CaptureStderr();
    const auto context_snapshot = DestroySessionOnFreshThread(std::move(session));
    const std::string teardown_log = testing::internal::GetCapturedStderr();

    EXPECT_EQ(context_snapshot.before_status, CUDA_SUCCESS);
    EXPECT_EQ(context_snapshot.before_context, nullptr);
    EXPECT_EQ(context_snapshot.after_status, CUDA_SUCCESS);
    EXPECT_EQ(context_snapshot.after_context, nullptr);
    EXPECT_TRUE(std::filesystem::is_regular_file(runtime_cache_dir));
    EXPECT_EQ(teardown_log.find("cache save was skipped"), std::string::npos) << teardown_log;
    EXPECT_NE(teardown_log.find("[NvTensorRTRTX EP] Failed to save runtime cache"), std::string::npos) << teardown_log;
    std::filesystem::remove(runtime_cache_dir);
}

// The synchronous GPU allocator option (nv_use_sync_gpu_allocator) forces TensorRT RTX to allocate
// through cudaMalloc/cudaFree (via the device BFC arena) instead of its default cudaMallocAsync
// path. Verify a session with the option enabled is accepted, builds, and runs end-to-end.
TEST(TensorRTRTXEpTest_Options, SyncGpuAllocator)
{
    const std::string model_name = "nv_execution_provider_sync_gpu_allocator.onnx";
    model_builder::CreateBaseModel(model_name, "test", {1, 3, 2});

    Ort::SessionOptions so;
    AppendTrtRtxEp(so, {{"nv_use_sync_gpu_allocator", "1"}});
    Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

    // The single-output base model builds and runs through the synchronous allocator.
    ASSERT_EQ(session.GetOutputCount(), 1u);

    auto io_binding = generate_io_binding(session);
    Ort::RunOptions run_options;
    session.Run(run_options, io_binding);
}

TEST(TensorRTRTXEpTest_Options, MultiRotaryCacheConcatOffsetAcceptsPrefixedOption)
{
    const auto info =
        TensorrtRtxExecutionProviderInfo::FromProviderOptions({{"nv_multi_rotary_cache_concat_offset", "4096"}});

    EXPECT_EQ(info.multi_rotary_cache_concat_offset, 4096);
}

TEST(TensorRTRTXEpTest_Options, MultiRotaryCacheConcatOffsetAcceptsLegacyOption)
{
    const auto info =
        TensorrtRtxExecutionProviderInfo::FromProviderOptions({{"multi_rotary_cache_concat_offset", "4096"}});

    EXPECT_EQ(info.multi_rotary_cache_concat_offset, 4096);
}

TEST(TensorRTRTXEpTest_Options, MultiRotaryCacheConcatOffsetRejectsConflictingAliases)
{
    EXPECT_THROW((void)TensorrtRtxExecutionProviderInfo::FromProviderOptions(
                     {{"nv_multi_rotary_cache_concat_offset", "4096"}, {"multi_rotary_cache_concat_offset", "8192"}}),
                 std::runtime_error);
}
