// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils/nvtx_range.h"

#include <cuda_runtime_api.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../../src/tensorrt_rtx_provider_options.h"
#include "trt_ep_perf_test_command_line_parser.h"
#include "trt_ep_perf_test_memory_monitor.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>

namespace
{
auto ToOrtString(const std::filesystem::path& path)
{
#ifdef _WIN32
    return path.wstring();
#else
    return path.string();
#endif
}

auto ToOrtDirectoryString(const std::filesystem::path& path)
{
    auto result = ToOrtString(path);
#ifdef _WIN32
    if (result.empty() || (result.back() != L'/' && result.back() != L'\\'))
        result += L'/';
#else
    if (result.empty() || result.back() != '/')
        result += '/';
#endif
    return result;
}

std::string EscapeJsonString(std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value)
    {
        switch (character)
        {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (character < 0x20)
            {
                result += "\\u00";
                result += hex[character >> 4];
                result += hex[character & 0x0f];
            }
            else
                result += static_cast<char>(character);
        }
    }
    return result;
}

std::vector<Ort::ConstEpDevice> GetTrtDevices(Ort::Env& env, const std::string& ep_name)
{
    std::vector<Ort::ConstEpDevice> devices;
    for (auto& device : env.GetEpDevices())
    {
        if (std::string(device.EpName()) == ep_name)
            devices.push_back(device);
    }
    if (devices.empty())
        throw std::runtime_error("no TensorRT RTX EP devices were registered");
    return devices;
}

Ort::SessionOptions MakeOptions(Ort::Env& env, const Arguments& args)
{
    Ort::SessionOptions options;
    if (args.overwrite_shapes)
        for (const auto& [name, dimension] : args.shapes)
            options.AddFreeDimensionOverrideByName(name.c_str(), dimension);
    Ort::KeyValuePairs ep_options;
    for (const auto& [key, value] : args.ep_options)
        ep_options.Add(key.c_str(), value.c_str());
    if (!args.runtime_cache.empty())
    {
        const auto cache = args.runtime_cache.u8string();
        ep_options.Add("nv_runtime_cache_path", reinterpret_cast<const char*>(cache.c_str()));
    }
    options.AppendExecutionProvider_V2(env, GetTrtDevices(env, args.ep_name), ep_options);
    return options;
}

Ort::ModelCompilationOptions MakeCompileOptions(Ort::Env& env, const Arguments& args,
                                                Ort::SessionOptions const& options)
{
    if (!args.context.parent_path().empty())
    {
        std::filesystem::create_directories(args.context.parent_path());
    }
    if (!args.runtime_cache.empty())
        std::filesystem::create_directories(args.runtime_cache);
    Ort::ModelCompilationOptions compile_options(env, options);
    compile_options.SetEpContextEmbedMode(args.context_embed_mode);
    compile_options.SetInputModelPath(ToOrtString(args.model).c_str());
    compile_options.SetOutputModelPath(ToOrtString(args.context).c_str());
    // emits warning that the context file path is set twice
    // compile_options.SetEpContextBinaryInformation(ToOrtDirectoryString(args.context.parent_path()).c_str(),
    //                                               ToOrtString(args.context.stem()).c_str());
    auto data_path = ToOrtString(args.context.filename().string() + ".data");
    compile_options.SetOutputModelExternalInitializersFile(data_path.c_str(), /* initializer_size_threshold= */ 1024);
    return compile_options;
}

struct InputSet
{
    std::vector<std::string> names;
    std::vector<Ort::Value> values;
};

size_t ElementBytes(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return 1;
    default:
        throw std::runtime_error("unsupported input tensor element type");
    }
}

const OrtMemoryInfo* GetValueMemoryInfo(const Ort::ConstEpDevice& device, IOBindingAllocator allocator)
{
    const OrtDeviceMemoryType memory_type = allocator == IOBindingAllocator::HostAccessible
                                                ? OrtDeviceMemoryType_HOST_ACCESSIBLE
                                                : OrtDeviceMemoryType_DEFAULT;
    return static_cast<const OrtMemoryInfo*>(device.GetMemoryInfo(memory_type));
}

class DeterministicRandom
{
public:
    uint64_t Next()
    {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 2685821657736338717ULL;
    }

private:
    uint64_t state_{0x4d595df4d0f33173ULL};
};

template <typename T>
void FillRandomUnsigned(void* data, size_t elements, DeterministicRandom& random)
{
    auto* values = static_cast<T*>(data);
    for (size_t i = 0; i < elements; ++i)
        values[i] = static_cast<T>(random.Next());
}

template <typename T>
void FillRandomSigned(void* data, size_t elements, DeterministicRandom& random)
{
    auto* values = static_cast<T*>(data);
    for (size_t i = 0; i < elements; ++i)
        values[i] = std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(random.Next()));
}

// Fills supported input types deterministically:
//      - floating-point values are in [-1, 1)
//      - signed and unsigned integers span their respective full representable ranges
//      - booleans are false or true with equal probability
void FillRandomInput(void* data, size_t elements, ONNXTensorElementDataType type, DeterministicRandom& random)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    {
        auto* values = static_cast<float*>(data);
        for (size_t i = 0; i < elements; ++i)
            values[i] = 2.0f * static_cast<float>(random.Next() >> 40) / static_cast<float>(1U << 24) - 1.0f;
        return;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    {
        auto* values = static_cast<double*>(data);
        for (size_t i = 0; i < elements; ++i)
            values[i] = 2.0 * static_cast<double>(random.Next() >> 11) / static_cast<double>(1ULL << 53) - 1.0;
        return;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        // Finite IEEE 754 binary16 values in [-1, 1).
        for (size_t i = 0; i < elements; ++i)
            static_cast<uint16_t*>(data)[i] =
                static_cast<uint16_t>((random.Next() & 0x3bff) | (random.Next() & 0x8000));
        return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        // Finite bfloat16 values in [-1, 1).
        for (size_t i = 0; i < elements; ++i)
            static_cast<uint16_t*>(data)[i] =
                static_cast<uint16_t>((random.Next() & 0x3f7f) | (random.Next() & 0x8000));
        return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        for (size_t i = 0; i < elements; ++i)
            static_cast<uint8_t*>(data)[i] = static_cast<uint8_t>((random.Next() & 0x37) | (random.Next() & 0x80));
        return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
        for (size_t i = 0; i < elements; ++i)
            static_cast<uint8_t*>(data)[i] =
                static_cast<uint8_t>((1 + (random.Next() % 0x37)) | (random.Next() & 0x80));
        return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        for (size_t i = 0; i < elements; ++i)
            static_cast<uint8_t*>(data)[i] = static_cast<uint8_t>((random.Next() & 0x3b) | (random.Next() & 0x80));
        return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        for (size_t i = 0; i < elements; ++i)
            static_cast<uint8_t*>(data)[i] =
                static_cast<uint8_t>((1 + (random.Next() % 0x3b)) | (random.Next() & 0x80));
        return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return FillRandomSigned<int8_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return FillRandomSigned<int16_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return FillRandomSigned<int32_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return FillRandomSigned<int64_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return FillRandomUnsigned<uint8_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return FillRandomUnsigned<uint16_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return FillRandomUnsigned<uint32_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return FillRandomUnsigned<uint64_t>(data, elements, random);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    {
        auto* values = static_cast<bool*>(data);
        for (size_t i = 0; i < elements; ++i)
            values[i] = (random.Next() & 1) != 0;
        return;
    }
    default:
        throw std::runtime_error("unsupported input tensor element type");
    }
}

InputSet MakeRandomInputs(Ort::Env& env, Ort::Session& session, const Arguments& args)
{
    InputSet result;
    const size_t count = session.GetInputCount();
    const auto input_devices = session.GetEpDeviceForInputs();
    if (input_devices.size() != count)
        throw std::runtime_error("input EP-device assignment count does not match input count");

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    DeterministicRandom random;
    result.names.reserve(count);
    result.values.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        auto name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        auto input_type_info = session.GetInputTypeInfo(i);
        auto info = input_type_info.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        const auto symbolic_dimensions = info.GetSymbolicDimensions();
        const std::string input_name(name.get());
        for (size_t dimension_index = 0; dimension_index < shape.size(); ++dimension_index)
        {
            const char* symbolic_dimension = symbolic_dimensions[dimension_index];
            if (symbolic_dimension == nullptr)
                continue;
            if (!args.overwrite_shapes)
            {
                if (const auto override = args.shapes.find(symbolic_dimension); override != args.shapes.end())
                    shape[dimension_index] = override->second;
            }
        }
        uint64_t elements = 1;
        for (auto& dimension : shape)
        {
            if (dimension <= 0)
                throw std::runtime_error("cannot warm up dynamic input shape for " + input_name +
                                         "; pass --shape DIMENSION=SIZE");
            elements *= static_cast<uint64_t>(dimension);
        }
        const size_t bytes = static_cast<size_t>(elements) * ElementBytes(info.GetElementType());
        result.names.emplace_back(input_name);
        OrtAllocator* allocator = nullptr;
        if (args.value_allocator == IOBindingAllocator::Cpu)
        {
            allocator = cpu_allocator;
        }
        else
        {
            const OrtMemoryInfo* memory_info = GetValueMemoryInfo(input_devices[i], args.value_allocator);
            allocator = env.GetSharedAllocator(memory_info);
            if (allocator == nullptr)
                throw std::runtime_error("no shared allocator is available for an input EP device");
        }
        result.values.emplace_back(
            Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType()));
        void* data = result.values.back().GetTensorMutableData<void>();
        if (args.value_allocator == IOBindingAllocator::EpDefault)
        {
            std::vector<uint8_t> host_data(bytes);
            FillRandomInput(host_data.data(), static_cast<size_t>(elements), info.GetElementType(), random);
            if (cudaMemcpy(data, host_data.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed while initializing an input tensor");
        }
        else
            FillRandomInput(data, static_cast<size_t>(elements), info.GetElementType(), random);
    }
    return result;
}

struct InferenceRunTiming
{
    uint64_t enqueue_us{};
    uint64_t total_us{};
};

class InferenceRunner
{
public:
    InferenceRunner(Ort::Env& env, Ort::Session& session, const Arguments& args)
        : session_(session)
        , cpu_memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
        , inputs_(MakeRandomInputs(env, session, args))
        , async_(args.async)
        , reuse_io_binding_(args.disable_io)
    {
        const auto output_devices = session.GetEpDeviceForOutputs();
        if (output_devices.size() != session.GetOutputCount())
            throw std::runtime_error("output EP-device assignment count does not match output count");

        for (size_t i = 0; i < session_.GetOutputCount(); ++i)
        {
            auto name = session_.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            output_names_.emplace_back(name.get());
            const OrtMemoryInfo* memory_info = cpu_memory_;
            if (args.value_allocator != IOBindingAllocator::Cpu)
                memory_info = GetValueMemoryInfo(output_devices[i], args.value_allocator);
            output_memory_infos_.push_back(memory_info);
        }
        if (reuse_io_binding_)
            binding_ = CreateIoBinding();
    }

    InferenceRunTiming Run()
    {
        Ort::RunOptions run_options;
        if (async_)
            run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

        const auto started = std::chrono::steady_clock::now();
        if (reuse_io_binding_)
        {
            session_.Run(run_options, *binding_);
            const auto enqueued = std::chrono::steady_clock::now();
            binding_->SynchronizeOutputs();
            return {static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(enqueued - started).count()),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - started)
                                              .count())};
        }
        else
        {
            auto binding = CreateIoBinding();
            session_.Run(run_options, *binding);
            const auto enqueued = std::chrono::steady_clock::now();
            binding->SynchronizeOutputs();
            return {static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(enqueued - started).count()),
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - started)
                                              .count())};
        }
    }

private:
    std::unique_ptr<Ort::IoBinding> CreateIoBinding()
    {
        auto binding = std::make_unique<Ort::IoBinding>(session_);
        for (size_t i = 0; i < inputs_.names.size(); ++i)
            binding->BindInput(inputs_.names[i].c_str(), inputs_.values[i]);
        for (size_t i = 0; i < output_names_.size(); ++i)
            binding->BindOutput(output_names_[i].c_str(), output_memory_infos_[i]);
        return binding;
    }

    Ort::Session& session_;
    Ort::MemoryInfo cpu_memory_;
    InputSet inputs_;
    bool async_{};
    bool reuse_io_binding_{};
    std::vector<std::string> output_names_;
    std::vector<const OrtMemoryInfo*> output_memory_infos_;
    std::unique_ptr<Ort::IoBinding> binding_;
};

struct MemoryDeltas;

struct InferenceStats
{
    uint64_t first_warmup_us{};
    uint64_t first_warmup_enqueue_us{};
    uint64_t min_us{};
    uint64_t max_us{};
    double median_us{};
    uint64_t enqueue_min_us{};
    uint64_t enqueue_max_us{};
    double enqueue_median_us{};

    void Serialize(std::ostream& output, const MemoryDeltas& memory, const char* indent) const;
};

InferenceStats RunInferenceBenchmark(InferenceRunner& runner, const Arguments& args)
{
    InferenceStats stats;
    const auto warmup_started = std::chrono::steady_clock::now();
    const auto first_warmup = runner.Run();
    stats.first_warmup_us = first_warmup.total_us;
    stats.first_warmup_enqueue_us = first_warmup.enqueue_us;
    if (args.warmup_iterations)
    {
        for (uint32_t i = 1; i < *args.warmup_iterations; ++i)
            runner.Run();
    }
    else
    {
        const auto duration = std::chrono::duration<double>(*args.warmup_duration_seconds);
        while (std::chrono::steady_clock::now() - warmup_started < duration)
            runner.Run();
    }

    std::vector<uint64_t> total_samples;
    std::vector<uint64_t> enqueue_samples;
    if (args.run_iterations)
    {
        total_samples.reserve(*args.run_iterations);
        enqueue_samples.reserve(*args.run_iterations);
        for (uint32_t i = 0; i < *args.run_iterations; ++i)
        {
            const auto timing = runner.Run();
            total_samples.push_back(timing.total_us);
            enqueue_samples.push_back(timing.enqueue_us);
        }
    }
    else
    {
        const auto duration = std::chrono::duration<double>(*args.run_duration_seconds);
        const auto run_started = std::chrono::steady_clock::now();
        do
        {
            const auto timing = runner.Run();
            total_samples.push_back(timing.total_us);
            enqueue_samples.push_back(timing.enqueue_us);
        } while (std::chrono::steady_clock::now() - run_started < duration);
    }
    const auto summarize = [](std::vector<uint64_t>& samples, uint64_t& min, uint64_t& max, double& median)
    {
        std::sort(samples.begin(), samples.end());
        min = samples.front();
        max = samples.back();
        const size_t middle = samples.size() / 2;
        median = samples.size() % 2 == 0
                     ? (static_cast<double>(samples[middle - 1]) + static_cast<double>(samples[middle])) / 2.0
                     : static_cast<double>(samples[middle]);
    };
    summarize(total_samples, stats.min_us, stats.max_us, stats.median_us);
    summarize(enqueue_samples, stats.enqueue_min_us, stats.enqueue_max_us, stats.enqueue_median_us);
    return stats;
}

struct MemoryDeltas
{
    uint64_t cpu_private_bytes{};
    uint64_t cpu_resident_bytes{};
    uint64_t gpu_process_bytes{};

    void Serialize(std::ostream& output, const char* indent) const
    {
        output << "{\n"
               << indent << "\t\"cpu_private_delta_bytes\": " << cpu_private_bytes << ",\n"
               << indent << "\t\"cpu_resident_delta_bytes\": " << cpu_resident_bytes << ",\n"
               << indent << "\t\"gpu_process_delta_bytes\": " << gpu_process_bytes << "\n"
               << indent << "}";
    }
};

void InferenceStats::Serialize(std::ostream& output, const MemoryDeltas& memory, const char* indent) const
{
    const std::string child_indent = std::string(indent) + "\t";
    output << "{\n"
           << child_indent << "\"first_us\": " << first_warmup_us << ",\n"
           << child_indent << "\"min_us\": " << min_us << ",\n"
           << child_indent << "\"max_us\": " << max_us << ",\n"
           << child_indent << "\"median_us\": " << median_us << ",\n"
           << child_indent << "\"first_enqueue_us\": " << first_warmup_enqueue_us << ",\n"
           << child_indent << "\"enqueue_min_us\": " << enqueue_min_us << ",\n"
           << child_indent << "\"enqueue_max_us\": " << enqueue_max_us << ",\n"
           << child_indent << "\"enqueue_median_us\": " << enqueue_median_us << ",\n"
           << child_indent << "\"memory\": ";
    memory.Serialize(output, child_indent.c_str());
    output << "\n" << indent << "}";
}

struct StageSample
{
    uint64_t elapsed_us{};
    MemoryDeltas memory;
};

template <typename Action>
StageSample MeasureStage(const char* stage_name, uint32_t index, Action&& action)
{
    MemoryMonitor monitor;
    monitor.Start();
    const HostMemory before_host = GetHostMemory();
    const auto started = std::chrono::steady_clock::now();
    {
        const std::string nvtx_label = std::string(stage_name) + "_" + std::to_string(index);
        trt_rtx_ep::NvtxRange range(nvtx_label.c_str());
        action();
    }
    const auto finished = std::chrono::steady_clock::now();
    monitor.Stop();
    const auto delta = [](uint64_t after, uint64_t before)
    {
        return after > before ? after - before : 0;
    };
    return {static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count()),
            {delta(monitor.PeakHost().private_bytes, before_host.private_bytes),
             delta(monitor.PeakHost().resident_bytes, before_host.resident_bytes),
             delta(monitor.PeakGpu(), monitor.BaselineGpu())}};
}

struct StageStats
{
    uint64_t first_us{};
    uint64_t min_us{};
    uint64_t max_us{};
    double median_us{};
    MemoryDeltas memory;

    void Serialize(std::ostream& output, const char* indent) const
    {
        const std::string child_indent = std::string(indent) + "\t";
        output << "{\n"
               << child_indent << "\"first_us\": " << first_us << ",\n"
               << child_indent << "\"min_us\": " << min_us << ",\n"
               << child_indent << "\"max_us\": " << max_us << ",\n"
               << child_indent << "\"median_us\": " << median_us << ",\n"
               << child_indent << "\"memory\": ";
        memory.Serialize(output, child_indent.c_str());
        output << "\n" << indent << "}";
    }
};

StageStats SummarizeStage(const std::vector<StageSample>& samples)
{
    if (samples.empty())
        throw std::runtime_error("cannot summarize an empty benchmark stage");

    StageStats result;
    result.first_us = samples.front().elapsed_us;
    std::vector<uint64_t> elapsed;
    elapsed.reserve(samples.size() > 1 ? samples.size() - 1 : 1);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sample = samples[i];
        if (i != 0 || samples.size() == 1)
            elapsed.push_back(sample.elapsed_us);
        result.memory.cpu_private_bytes = std::max(result.memory.cpu_private_bytes, sample.memory.cpu_private_bytes);
        result.memory.cpu_resident_bytes = std::max(result.memory.cpu_resident_bytes, sample.memory.cpu_resident_bytes);
        result.memory.gpu_process_bytes = std::max(result.memory.gpu_process_bytes, sample.memory.gpu_process_bytes);
    }
    std::sort(elapsed.begin(), elapsed.end());
    result.min_us = elapsed.front();
    result.max_us = elapsed.back();
    const size_t middle = elapsed.size() / 2;
    result.median_us = elapsed.size() % 2 == 0
                           ? (static_cast<double>(elapsed[middle - 1]) + static_cast<double>(elapsed[middle])) / 2.0
                           : static_cast<double>(elapsed[middle]);
    return result;
}

void PrintBenchmarkResult(std::ostream& output, const std::optional<StageStats>& compile,
                          const std::optional<StageStats>& load, const StageSample* inference_stage,
                          const InferenceStats* inference)
{
    output << "{\n\t\"status\": \"ok\",\n\t\"compile\": ";
    if (compile)
        compile->Serialize(output, "\t");
    else
        output << "{}";

    output << ",\n\t\"load\": ";
    if (load)
        load->Serialize(output, "\t");
    else
        output << "{}";

    output << ",\n\t\"inference\": ";
    if (inference_stage == nullptr || inference == nullptr)
    {
        output << "{}";
    }
    else
    {
        inference->Serialize(output, inference_stage->memory, "\t");
    }
    output << "\n}" << std::endl;
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Arguments args = CommandLineParser::Parse(argc, argv);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "trt_rtx_ep_context_benchmark");
        if (args.ep.empty())
        {
            args.ep = DLL_NAME("onnxruntime_providers_nv_tensorrt_rtx");
        }
        env.RegisterExecutionProviderLibrary(args.ep_name.c_str(), ToOrtString(args.ep).c_str());
        std::ofstream output_file;
        std::ostream* output = &std::cout;
        if (!args.output.empty())
        {
            if (!args.output.parent_path().empty())
                std::filesystem::create_directories(args.output.parent_path());
            output_file.open(args.output);
            if (!output_file)
                throw std::runtime_error("failed to open output file " + args.output.string());
            output = &output_file;
        }

        auto options = MakeOptions(env, args);
        std::optional<StageStats> compile;
        if (!args.skip_compile)
        {
            auto compile_options = MakeCompileOptions(env, args, options);
            std::vector<StageSample> samples;
            samples.reserve(args.time_compile_iterations);
            for (uint32_t i = 0; i < args.time_compile_iterations; ++i)
                samples.push_back(MeasureStage("compile", i,
                                               [&]
                                               {
                                                   Ort::ThrowOnError(Ort::CompileModel(env, compile_options));
                                               }));
            compile = SummarizeStage(samples);
        }

        std::unique_ptr<Ort::Session> session;
        std::vector<StageSample> load_samples;
        load_samples.reserve(args.time_deserialize_iterations);
        for (uint32_t i = 0; i < args.time_deserialize_iterations; ++i)
        {
            // Do not include the previous session in this load's peak-memory sample.
            session.reset();
            load_samples.push_back(MeasureStage("load", i,
                                                [&]
                                                {
                                                    session = std::make_unique<Ort::Session>(
                                                        env, ToOrtString(args.context).c_str(), options);
                                                }));
        }
        const StageStats load = SummarizeStage(load_samples);
        if (args.load_only)
        {
            PrintBenchmarkResult(*output, compile, load, nullptr, nullptr);
            return 0;
        }
        InferenceRunner runner(env, *session, args);
        InferenceStats inference;
        const StageSample inference_stage = MeasureStage("inference", 0,
                                                         [&]
                                                         {
                                                             inference = RunInferenceBenchmark(runner, args);
                                                         });
        PrintBenchmarkResult(*output, compile, load, &inference_stage, &inference);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cout << "{\"status\":\"error\",\"message\":\"" << EscapeJsonString(error.what()) << "\"}" << std::endl;
        return 1;
    }
}
