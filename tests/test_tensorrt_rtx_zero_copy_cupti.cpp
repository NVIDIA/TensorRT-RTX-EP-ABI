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

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#if TRT_RTX_ENABLE_CUPTI_MEMCPY_TEST
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cupti.h>
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{

struct TensorSpec
{
    std::string name;
    ONNXTensorElementDataType type{};
    std::vector<int64_t> shape;
};

size_t ElementCount(const std::vector<int64_t>& shape)
{
    size_t count = 1;
    for (int64_t dim : shape)
    {
        if (dim <= 0)
        {
            throw std::runtime_error("unresolved tensor shape");
        }

        const auto unsigned_dim = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / unsigned_dim)
        {
            throw std::runtime_error("tensor shape overflows size_t");
        }
        count *= unsigned_dim;
    }
    return count;
}

std::vector<TensorSpec> GetInputSpecs(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator)
{
    std::vector<TensorSpec> specs;
    const size_t count = session.GetInputCount();
    specs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        specs.push_back({name.get(), info.GetElementType(), info.GetShape()});
    }
    return specs;
}

std::vector<TensorSpec> GetOutputSpecs(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator)
{
    std::vector<TensorSpec> specs;
    const size_t count = session.GetOutputCount();
    specs.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
        auto type_info = session.GetOutputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        specs.push_back({name.get(), info.GetElementType(), info.GetShape()});
    }
    return specs;
}

std::vector<const char*> MakeNamePointers(const std::vector<std::string>& names)
{
    std::vector<const char*> ptrs;
    ptrs.reserve(names.size());
    for (const std::string& name : names)
    {
        ptrs.push_back(name.c_str());
    }
    return ptrs;
}

Ort::Value CreateTensor(OrtAllocator* allocator, const TensorSpec& spec)
{
    return Ort::Value::CreateTensor(allocator, spec.shape.data(), spec.shape.size(), spec.type);
}

void FillInputTensor(Ort::Value& tensor, const TensorSpec& spec)
{
    ASSERT_EQ(spec.type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        << "This CUPTI zero-copy test currently expects base-model float inputs.";

    float* data = tensor.GetTensorMutableData<float>();
    const size_t elements = ElementCount(spec.shape);
    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = static_cast<float>(static_cast<int>(i % 255) - 127) / 127.0f;
    }
}

std::vector<Ort::Value> CreateInputs(OrtAllocator* allocator, const std::vector<TensorSpec>& specs)
{
    std::vector<Ort::Value> values;
    values.reserve(specs.size());
    for (const TensorSpec& spec : specs)
    {
        values.push_back(CreateTensor(allocator, spec));
        FillInputTensor(values.back(), spec);
    }
    return values;
}

std::vector<Ort::Value> CreateOutputs(OrtAllocator* allocator, const std::vector<TensorSpec>& specs)
{
    std::vector<Ort::Value> values;
    values.reserve(specs.size());
    for (const TensorSpec& spec : specs)
    {
        values.push_back(CreateTensor(allocator, spec));
    }
    return values;
}

int GetEnvInt(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }

    try
    {
        return std::max(0, std::stoi(value));
    }
    catch (...)
    {
        return fallback;
    }
}

void RunOnce(Ort::Session& session, Ort::RunOptions& run_options, const std::vector<const char*>& input_names,
             std::vector<Ort::Value>& inputs, const std::vector<const char*>& output_names,
             std::vector<Ort::Value>& outputs)
{
    session.Run(run_options, input_names.data(), inputs.data(), inputs.size(), output_names.data(), outputs.data(),
                outputs.size());
    CUDA_CHECK(cudaDeviceSynchronize());
}

double RunTimedAverageMs(Ort::Session& session, Ort::RunOptions& run_options,
                         const std::vector<const char*>& input_names, std::vector<Ort::Value>& inputs,
                         const std::vector<const char*>& output_names, std::vector<Ort::Value>& outputs, int iterations)
{
    if (iterations <= 0)
    {
        return 0.0;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        RunOnce(session, run_options, input_names, inputs, output_names, outputs);
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / static_cast<double>(iterations);
}

void ExpectOutputsClose(const std::vector<Ort::Value>& expected, const std::vector<Ort::Value>& actual)
{
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t output_index = 0; output_index < expected.size(); ++output_index)
    {
        auto expected_info = expected[output_index].GetTensorTypeAndShapeInfo();
        auto actual_info = actual[output_index].GetTensorTypeAndShapeInfo();
        ASSERT_EQ(expected_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            << "This CUPTI zero-copy test currently expects base-model float outputs.";
        ASSERT_EQ(actual_info.GetElementType(), expected_info.GetElementType());
        ASSERT_EQ(actual_info.GetShape(), expected_info.GetShape());

        const size_t elements = expected_info.GetElementCount();
        const float* expected_data = expected[output_index].GetTensorData<float>();
        const float* actual_data = actual[output_index].GetTensorData<float>();
        for (size_t i = 0; i < elements; ++i)
        {
            ASSERT_NEAR(actual_data[i], expected_data[i], 1e-4f)
                << "Mismatch at output " << output_index << ", element " << i;
        }
    }
}

void AppendTrtRtxEp(Ort::SessionOptions& session_options)
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, devices, ep_options);
}

#if TRT_RTX_ENABLE_CUPTI_MEMCPY_TEST

struct MemcpyStats
{
    uint64_t count{};
    uint64_t bytes{};
    uint64_t h2d_bytes{};
    uint64_t d2h_bytes{};
};

class CuptiApi
{
public:
    using ActivityDisableFn = CUptiResult(CUPTIAPI*)(CUpti_ActivityKind);
    using ActivityEnableFn = CUptiResult(CUPTIAPI*)(CUpti_ActivityKind);
    using ActivityFlushAllFn = CUptiResult(CUPTIAPI*)(uint32_t);
    using ActivityGetNextRecordFn = CUptiResult(CUPTIAPI*)(uint8_t*, size_t, CUpti_Activity**);
    using ActivityGetNumDroppedRecordsFn = CUptiResult(CUPTIAPI*)(CUcontext, uint32_t, size_t*);
    using ActivityRegisterCallbacksFn = CUptiResult(CUPTIAPI*)(CUpti_BuffersCallbackRequestFunc,
                                                               CUpti_BuffersCallbackCompleteFunc);
    using GetResultStringFn = CUptiResult(CUPTIAPI*)(CUptiResult, const char**);

    CuptiApi(const CuptiApi&) = delete;
    CuptiApi& operator=(const CuptiApi&) = delete;

    static CuptiApi& Instance()
    {
        static CuptiApi api;
        return api;
    }

    bool Available() const
    {
        return available_;
    }

    const std::string& LoadError() const
    {
        return load_error_;
    }

    ActivityDisableFn activity_disable{};
    ActivityEnableFn activity_enable{};
    ActivityFlushAllFn activity_flush_all{};
    ActivityGetNextRecordFn activity_get_next_record{};
    ActivityGetNumDroppedRecordsFn activity_get_num_dropped_records{};
    ActivityRegisterCallbacksFn activity_register_callbacks{};
    GetResultStringFn get_result_string{};

private:
    CuptiApi()
    {
        LoadLibraryHandle();
        if (library_ == nullptr)
        {
            return;
        }

        activity_disable = LoadSymbol<ActivityDisableFn>("cuptiActivityDisable");
        activity_enable = LoadSymbol<ActivityEnableFn>("cuptiActivityEnable");
        activity_flush_all = LoadSymbol<ActivityFlushAllFn>("cuptiActivityFlushAll");
        activity_get_next_record = LoadSymbol<ActivityGetNextRecordFn>("cuptiActivityGetNextRecord");
        activity_get_num_dropped_records =
            LoadSymbol<ActivityGetNumDroppedRecordsFn>("cuptiActivityGetNumDroppedRecords");
        activity_register_callbacks = LoadSymbol<ActivityRegisterCallbacksFn>("cuptiActivityRegisterCallbacks");
        get_result_string = LoadSymbol<GetResultStringFn>("cuptiGetResultString");

        available_ = activity_disable != nullptr && activity_enable != nullptr && activity_flush_all != nullptr &&
                     activity_get_next_record != nullptr && activity_get_num_dropped_records != nullptr &&
                     activity_register_callbacks != nullptr && get_result_string != nullptr;
        if (!available_ && load_error_.empty())
        {
            load_error_ = "CUPTI library is missing one or more required activity symbols.";
        }
    }

    void LoadLibraryHandle()
    {
#ifdef _WIN32
        auto try_load_from_directory = [this](const std::filesystem::path& directory)
        {
            if (library_ != nullptr || directory.empty() || !std::filesystem::exists(directory))
            {
                return;
            }

            for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory))
            {
                if (!entry.is_regular_file())
                {
                    continue;
                }

                const std::string filename = entry.path().filename().string();
                if (filename.rfind("cupti", 0) != 0 || entry.path().extension() != ".dll")
                {
                    continue;
                }

                library_ = LoadLibraryA(entry.path().string().c_str());
                if (library_ != nullptr)
                {
                    return;
                }
            }
        };

        char module_path[MAX_PATH]{};
        const DWORD module_path_size =
            GetModuleFileNameA(nullptr, module_path, static_cast<DWORD>(sizeof(module_path)));
        if (module_path_size != 0 && module_path_size < sizeof(module_path))
        {
            try_load_from_directory(std::filesystem::path(module_path).parent_path());
        }
        try_load_from_directory(std::filesystem::current_path());
        if (library_ != nullptr)
        {
            return;
        }

        const char* const candidates[] = {
            "cupti64_2026.3.0.dll", "cupti64_2026.2.0.dll", "cupti64_2026.1.0.dll",
            "cupti64_2025.4.0.dll", "cupti64_2025.3.0.dll", "cupti64_2025.2.0.dll",
            "cupti64_2025.1.0.dll", "cupti64.dll",          "cupti.dll",
        };
        for (const char* candidate : candidates)
        {
            library_ = LoadLibraryA(candidate);
            if (library_ != nullptr)
            {
                return;
            }
        }

        load_error_ = "CUPTI DLL was not found in the test process DLL search path.";
#else
        const char* const candidates[] = {
            "libcupti.so",
            "libcupti.so.13",
            "libcupti.so.12",
        };
        for (const char* candidate : candidates)
        {
            library_ = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
            if (library_ != nullptr)
            {
                return;
            }
        }

        const char* error = dlerror();
        load_error_ = error != nullptr ? error : "CUPTI shared library was not found.";
#endif
    }

    template <typename Fn>
    Fn LoadSymbol(const char* name)
    {
#ifdef _WIN32
        FARPROC symbol = GetProcAddress(library_, name);
        if (symbol == nullptr && load_error_.empty())
        {
            load_error_ = std::string("CUPTI symbol was not found: ") + name;
        }
        return reinterpret_cast<Fn>(symbol);
#else
        void* symbol = dlsym(library_, name);
        if (symbol == nullptr && load_error_.empty())
        {
            load_error_ = std::string("CUPTI symbol was not found: ") + name;
        }
        return reinterpret_cast<Fn>(symbol);
#endif
    }

#ifdef _WIN32
    HMODULE library_{};
#else
    void* library_{};
#endif
    bool available_{};
    std::string load_error_;
};

const char* CuptiResultString(CUptiResult result)
{
    const char* error_string = nullptr;
    const CuptiApi& cupti = CuptiApi::Instance();
    if (!cupti.Available() || cupti.get_result_string(result, &error_string) != CUPTI_SUCCESS ||
        error_string == nullptr)
    {
        return "<unknown CUPTI error>";
    }
    return error_string;
}

void CheckCupti(CUptiResult result, const char* call)
{
    if (result != CUPTI_SUCCESS)
    {
        throw std::runtime_error(std::string(call) + " failed: " + CuptiResultString(result));
    }
}

class CuptiMemcpyTracker
{
public:
    CuptiMemcpyTracker() = default;
    CuptiMemcpyTracker(const CuptiMemcpyTracker&) = delete;
    CuptiMemcpyTracker& operator=(const CuptiMemcpyTracker&) = delete;

    void Start()
    {
        CuptiApi& cupti = CuptiApi::Instance();
        Reset();
        if (active_)
        {
            throw std::runtime_error("CUPTI memcpy tracker is already active");
        }

        CheckCupti(cupti.activity_register_callbacks(BufferRequested, BufferCompleted),
                   "cuptiActivityRegisterCallbacks");
        CheckCupti(cupti.activity_enable(CUPTI_ACTIVITY_KIND_MEMCPY),
                   "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY)");
#if defined(CUPTI_ACTIVITY_KIND_MEMCPY2)
        CheckCupti(cupti.activity_enable(CUPTI_ACTIVITY_KIND_MEMCPY2),
                   "cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2)");
#endif
        active_ = true;
    }

    MemcpyStats Stop()
    {
        CuptiApi& cupti = CuptiApi::Instance();
        if (!active_)
        {
            throw std::runtime_error("CUPTI memcpy tracker is not active");
        }

        CheckCupti(cupti.activity_flush_all(0), "cuptiActivityFlushAll");
        CheckCupti(cupti.activity_disable(CUPTI_ACTIVITY_KIND_MEMCPY),
                   "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY)");
#if defined(CUPTI_ACTIVITY_KIND_MEMCPY2)
        CheckCupti(cupti.activity_disable(CUPTI_ACTIVITY_KIND_MEMCPY2),
                   "cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2)");
#endif
        active_ = false;
        return Snapshot();
    }

private:
    static void Reset()
    {
        count_.store(0);
        bytes_.store(0);
        h2d_bytes_.store(0);
        d2h_bytes_.store(0);
    }

    static MemcpyStats Snapshot()
    {
        return {
            count_.load(),
            bytes_.load(),
            h2d_bytes_.load(),
            d2h_bytes_.load(),
        };
    }

    static void AddMemcpy(uint64_t bytes, uint8_t copy_kind)
    {
        count_.fetch_add(1);
        bytes_.fetch_add(bytes);
        if (copy_kind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD)
        {
            h2d_bytes_.fetch_add(bytes);
        }
        else if (copy_kind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH)
        {
            d2h_bytes_.fetch_add(bytes);
        }
    }

    static void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* max_num_records)
    {
        constexpr size_t kBufferSize = 64 * 1024;
        *buffer = static_cast<uint8_t*>(std::malloc(kBufferSize));
        if (*buffer == nullptr)
        {
            *size = 0;
            *max_num_records = 0;
            return;
        }

        *size = kBufferSize;
        *max_num_records = 0;
    }

    static void CUPTIAPI BufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size)
    {
        CuptiApi& cupti = CuptiApi::Instance();
        CUpti_Activity* record = nullptr;
        while (cupti.activity_get_next_record(buffer, valid_size, &record) == CUPTI_SUCCESS)
        {
            if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY)
            {
                const auto* memcpy = reinterpret_cast<const CUpti_ActivityMemcpy*>(record);
                AddMemcpy(memcpy->bytes, memcpy->copyKind);
            }
#if defined(CUPTI_ACTIVITY_KIND_MEMCPY2)
            else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY2)
            {
                const auto* memcpy = reinterpret_cast<const CUpti_ActivityMemcpy2*>(record);
                AddMemcpy(memcpy->bytes, memcpy->copyKind);
            }
#endif
        }

        size_t dropped = 0;
        (void)cupti.activity_get_num_dropped_records(nullptr, 0, &dropped);
        if (dropped != 0)
        {
            std::cerr << "CUPTI dropped " << dropped << " activity record(s)\n";
        }
        std::free(buffer);
    }

    bool active_{};
    static std::atomic<uint64_t> count_;
    static std::atomic<uint64_t> bytes_;
    static std::atomic<uint64_t> h2d_bytes_;
    static std::atomic<uint64_t> d2h_bytes_;
};

std::atomic<uint64_t> CuptiMemcpyTracker::count_{0};
std::atomic<uint64_t> CuptiMemcpyTracker::bytes_{0};
std::atomic<uint64_t> CuptiMemcpyTracker::h2d_bytes_{0};
std::atomic<uint64_t> CuptiMemcpyTracker::d2h_bytes_{0};

#endif

}  // namespace

TEST(TensorRTRTXEpTest, HostAccessibleRunAvoidsPageableMemcpyActivity)
{
#if !TRT_RTX_ENABLE_CUPTI_MEMCPY_TEST
    GTEST_SKIP() << "CUPTI headers are not available in this build.";
#else
    CuptiApi& cupti = CuptiApi::Instance();
    if (!cupti.Available())
    {
        GTEST_SKIP() << cupti.LoadError();
    }

    ASSERT_NE(ort_env, nullptr);

    std::string ort_version;
    if (!ort_runtime_at_least(1, 27, ort_version))
    {
        GTEST_SKIP() << "ORT runtime " << ort_version
                     << " does not support the HOST_ACCESSIBLE zero-copy path. Requires ORT 1.27 or newer.";
    }

    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";

    const OrtApi& api = Ort::GetApi();
    const OrtMemoryInfo* host_accessible_mem_info =
        api.EpDevice_MemoryInfo(devices[0], OrtDeviceMemoryType_HOST_ACCESSIBLE);
    ASSERT_NE(host_accessible_mem_info, nullptr) << "TRT RTX EP did not expose HOST_ACCESSIBLE memory.";

    OrtAllocator* host_accessible_allocator = nullptr;
    Ort::ThrowOnError(api.GetSharedAllocator(*ort_env, host_accessible_mem_info, &host_accessible_allocator));
    ASSERT_NE(host_accessible_allocator, nullptr);

    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options.DisableMemPattern();
    AppendTrtRtxEp(session_options);

    const int spatial_dim = std::max(1, GetEnvInt("TRT_RTX_ZERO_COPY_CUPTI_BASE_MODEL_SPATIAL", 1024));
    const std::string model_path = "zero_copy_base_model.onnx";
    clearFileIfExists(model_path);
    model_builder::CreateBaseModel(model_path, "zero_copy_base_model", {1, 3, spatial_dim, spatial_dim});
    ASSERT_TRUE(std::filesystem::exists(model_path));

    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    std::vector<TensorSpec> input_specs = GetInputSpecs(session, cpu_allocator);
    std::vector<TensorSpec> output_specs = GetOutputSpecs(session, cpu_allocator);
    ASSERT_FALSE(input_specs.empty());
    ASSERT_FALSE(output_specs.empty());

    std::vector<std::string> input_names_storage;
    input_names_storage.reserve(input_specs.size());
    for (const TensorSpec& spec : input_specs)
    {
        input_names_storage.push_back(spec.name);
    }
    std::vector<std::string> output_names_storage;
    output_names_storage.reserve(output_specs.size());
    for (const TensorSpec& spec : output_specs)
    {
        output_names_storage.push_back(spec.name);
    }
    const std::vector<const char*> input_names = MakeNamePointers(input_names_storage);
    const std::vector<const char*> output_names = MakeNamePointers(output_names_storage);

    std::vector<Ort::Value> pageable_inputs = CreateInputs(cpu_allocator, input_specs);
    std::vector<Ort::Value> pageable_outputs = CreateOutputs(cpu_allocator, output_specs);
    std::vector<Ort::Value> pinned_inputs = CreateInputs(host_accessible_allocator, input_specs);
    std::vector<Ort::Value> pinned_outputs = CreateOutputs(host_accessible_allocator, output_specs);

    Ort::RunOptions run_options;

    RunOnce(session, run_options, input_names, pageable_inputs, output_names, pageable_outputs);
    RunOnce(session, run_options, input_names, pinned_inputs, output_names, pinned_outputs);

    const int timing_iterations = GetEnvInt("TRT_RTX_ZERO_COPY_CUPTI_TIMING_ITERS", 10);
    const double pageable_avg_ms = RunTimedAverageMs(session, run_options, input_names, pageable_inputs, output_names,
                                                     pageable_outputs, timing_iterations);
    const double pinned_avg_ms = RunTimedAverageMs(session, run_options, input_names, pinned_inputs, output_names,
                                                   pinned_outputs, timing_iterations);

    std::cout << "E2E pageable avg ms over " << timing_iterations << " iteration(s): " << pageable_avg_ms << "\n";
    std::cout << "E2E HOST_ACCESSIBLE avg ms over " << timing_iterations << " iteration(s): " << pinned_avg_ms << "\n";
    if (pinned_avg_ms > 0.0)
    {
        std::cout << "E2E pageable / HOST_ACCESSIBLE ratio: " << pageable_avg_ms / pinned_avg_ms << "\n";
    }

    CuptiMemcpyTracker tracker;
    tracker.Start();
    RunOnce(session, run_options, input_names, pageable_inputs, output_names, pageable_outputs);
    const MemcpyStats pageable = tracker.Stop();

    tracker.Start();
    RunOnce(session, run_options, input_names, pinned_inputs, output_names, pinned_outputs);
    const MemcpyStats pinned = tracker.Stop();

    std::cout << "CUPTI pageable memcpys for 1 iteration: count=" << pageable.count << " bytes=" << pageable.bytes
              << " h2d=" << pageable.h2d_bytes << " d2h=" << pageable.d2h_bytes << "\n";
    std::cout << "CUPTI HOST_ACCESSIBLE memcpys for 1 iteration: count=" << pinned.count << " bytes=" << pinned.bytes
              << " h2d=" << pinned.h2d_bytes << " d2h=" << pinned.d2h_bytes << "\n";

    EXPECT_GT(pageable.h2d_bytes + pageable.d2h_bytes, 0u)
        << "Expected pageable Session::Run to require host/device memcpy activity.";
    EXPECT_LT(pinned.h2d_bytes + pinned.d2h_bytes, pageable.h2d_bytes + pageable.d2h_bytes)
        << "Expected HOST_ACCESSIBLE Session::Run to reduce host/device memcpy activity.";

    ExpectOutputsClose(pageable_outputs, pinned_outputs);
#endif
}
