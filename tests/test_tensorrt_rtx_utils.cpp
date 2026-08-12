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

#include "test_tensorrt_rtx_utils.h"

#include <cuda_profiler_api.h>

#include <chrono>
#include <iostream>

// =============================================================================
// NVTX — compile with -DNVTX to enable; no-op stub otherwise.
// =============================================================================
#ifdef NVTX
#include <nvtx3/nvtx3.hpp>
#else
namespace nvtx3
{
struct scoped_range
{
    explicit scoped_range(const char*) {}
    explicit scoped_range(const std::string&) {}
};
}  // namespace nvtx3
#endif

// =============================================================================
// Session inspection
// =============================================================================

void describe_session(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions cpu_alloc;

    const auto input_count = session.GetInputCount();
    std::cout << "Input count: " << input_count << "\n";
    for (size_t i = 0; i < input_count; ++i)
    {
        auto name = session.GetInputNameAllocated(i, cpu_alloc);
        auto type_info = session.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << "  [" << i << "] " << name.get() << "  dtype=" << info.GetElementType() << "  shape=[ ";
        for (auto s : info.GetShape())
            std::cout << s << " ";
        std::cout << "]\n";
    }

    const auto output_count = session.GetOutputCount();
    std::cout << "Output count: " << output_count << "\n";
    for (size_t i = 0; i < output_count; ++i)
    {
        auto name = session.GetOutputNameAllocated(i, cpu_alloc);
        auto type_info = session.GetOutputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << "  [" << i << "] " << name.get() << "  dtype=" << info.GetElementType() << "  shape=[ ";
        for (auto s : info.GetShape())
            std::cout << s << " ";
        std::cout << "]\n";
    }
}

// =============================================================================
// Type / size helpers
// =============================================================================

size_t ONNXDtypeToBytes(ONNXTensorElementDataType t)
{
    switch (t)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return sizeof(int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return sizeof(uint64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return sizeof(bool);
    default:
        std::cerr << "ONNXDtypeToBytes: unexpected dtype " << t << "\n";
        return 0;
    }
}

size_t ORTValueToBytes(const Ort::Value& value)
{
    auto info = value.GetTensorTypeAndShapeInfo();
    return info.GetElementCount() * ONNXDtypeToBytes(info.GetElementType());
}

// =============================================================================
// run_with_cpu_bindings
// =============================================================================

void run_with_cpu_bindings(Ort::Session& session, int iterations)
{
    Ort::AllocatorWithDefaultOptions cpu_alloc;
    Ort::IoBinding io_binding(session);

    // Pre-allocate output tensors on CPU
    const auto output_count = session.GetOutputCount();
    std::vector<std::string> output_names;
    std::vector<Ort::Value> output_values_flop, output_values_flip;
    output_names.reserve(output_count);
    output_values_flop.reserve(output_count);
    output_values_flip.reserve(output_count);

    for (size_t i = 0; i < output_count; ++i)
    {
        auto name = session.GetOutputNameAllocated(i, cpu_alloc);
        auto info = session.GetOutputTypeInfo(i);
        auto shape_info = info.GetTensorTypeAndShapeInfo();
        auto type = shape_info.GetElementType();
        output_values_flop.emplace_back(
            Ort::Value::CreateTensor(cpu_alloc, shape_info.GetShape().data(), shape_info.GetShape().size(), type));
        output_values_flip.emplace_back(Ort::Value::CreateTensor(
            cpu_alloc, shape_info.GetShape().data(), shape_info.GetShape().size(), shape_info.GetElementType()));
        io_binding.BindOutput(name.get(), output_values_flop.back());
        output_names.emplace_back(name.get());
    }

    CUDA_CHECK(cudaProfilerStart());
    {
        nvtx3::scoped_range r{"cpu bindings"};
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            // Re-bind inputs every iteration (zero-filled)
            const auto input_count = session.GetInputCount();
            for (size_t in = 0; in < input_count; ++in)
            {
                auto name = session.GetInputNameAllocated(in, cpu_alloc);
                auto info = session.GetInputTypeInfo(in);
                auto shape_info = info.GetTensorTypeAndShapeInfo();
                auto type = shape_info.GetElementType();
                auto val = Ort::Value::CreateTensor(cpu_alloc, shape_info.GetShape().data(),
                                                    shape_info.GetShape().size(), type);
                io_binding.BindInput(name.get(), val);
            }

            Ort::RunOptions run_options;
            session.Run(run_options, io_binding);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << "CPU bindings per iteration [ms]: "
                  << static_cast<float>(us) / static_cast<float>(iterations) / 1000.f << "\n";
    }
    CUDA_CHECK(cudaProfilerStop());
}

// =============================================================================
// run_with_gpu_bindings
// =============================================================================

void run_with_gpu_bindings(Ort::Session& session, int iterations, cudaStream_t stream)
{
    const int device_id = 0;

    Ort::MemoryInfo pinned_info("CudaPinned", OrtArenaAllocator, device_id, OrtMemTypeDefault);
    Ort::Allocator cpu_alloc(session, pinned_info);

    Ort::MemoryInfo cuda_info("Cuda", OrtArenaAllocator, device_id, OrtMemTypeDefault);
    Ort::Allocator gpu_alloc(session, cuda_info);
    OrtAllocator* gpu = gpu_alloc;

    cudaStream_t upload_stream, download_stream;
    CUDA_CHECK(cudaStreamCreate(&upload_stream));
    CUDA_CHECK(cudaStreamCreate(&download_stream));

    // Pre-allocate GPU output tensors (double-buffered: flop/flip)
    const auto output_count = session.GetOutputCount();
    std::vector<std::string> output_names;
    std::vector<Ort::Value> out_flop, out_flip;
    output_names.reserve(output_count);
    out_flop.reserve(output_count);
    out_flip.reserve(output_count);

    Ort::IoBinding io_binding(session);
    for (size_t i = 0; i < output_count; ++i)
    {
        auto name = session.GetOutputNameAllocated(i, cpu_alloc);
        auto info = session.GetOutputTypeInfo(i);
        auto shape_info = info.GetTensorTypeAndShapeInfo();
        out_flop.emplace_back(Ort::Value::CreateTensor(gpu, shape_info.GetShape().data(), shape_info.GetShape().size(),
                                                       shape_info.GetElementType()));
        out_flip.emplace_back(Ort::Value::CreateTensor(gpu, shape_info.GetShape().data(), shape_info.GetShape().size(),
                                                       shape_info.GetElementType()));
        io_binding.BindOutput(name.get(), out_flop.back());
        output_names.emplace_back(name.get());
    }

    // Pre-allocate GPU input tensors and CPU staging buffers
    const auto input_count = session.GetInputCount();
    std::vector<Ort::Value> in_gpu, in_cpu, out_cpu;
    in_gpu.reserve(input_count);
    in_cpu.reserve(input_count);

    for (size_t i = 0; i < input_count; ++i)
    {
        auto name = session.GetInputNameAllocated(i, cpu_alloc);
        auto type_info = session.GetInputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        in_gpu.emplace_back(
            Ort::Value::CreateTensor(gpu, info.GetShape().data(), info.GetShape().size(), info.GetElementType()));
        in_cpu.emplace_back(
            Ort::Value::CreateTensor(cpu_alloc, info.GetShape().data(), info.GetShape().size(), info.GetElementType()));
        io_binding.BindInput(name.get(), in_gpu.back());
    }
    for (size_t i = 0; i < output_count; ++i)
    {
        auto name = session.GetOutputNameAllocated(i, cpu_alloc);
        auto type_info = session.GetOutputTypeInfo(i);
        auto info = type_info.GetTensorTypeAndShapeInfo();
        out_cpu.emplace_back(
            Ort::Value::CreateTensor(cpu_alloc, info.GetShape().data(), info.GetShape().size(), info.GetElementType()));
    }

    cudaEvent_t ev_infer, ev_infer_cpu, ev_upload;
    CUDA_CHECK(cudaEventCreate(&ev_infer));
    CUDA_CHECK(cudaEventCreate(&ev_infer_cpu));
    CUDA_CHECK(cudaEventCreate(&ev_upload));
    // Mark as already done so the first iteration doesn't block
    CUDA_CHECK(cudaEventRecord(ev_infer, stream));

    CUDA_CHECK(cudaProfilerStart());
    {
        nvtx3::scoped_range r{"gpu bindings"};
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            auto& out_current = (i % 2) ? out_flip : out_flop;
            const std::string label = (i % 2) ? "flip" : "flop";

            // Upload inputs
            {
                nvtx3::scoped_range ru{"upload"};
                for (size_t in = 0; in < in_gpu.size(); ++in)
                {
                    CUDA_CHECK(cudaMemcpyAsync(in_gpu[in].GetTensorMutableRawData(), in_cpu[in].GetTensorRawData(),
                                               ORTValueToBytes(in_gpu[in]), cudaMemcpyHostToDevice, upload_stream));
                }
                CUDA_CHECK(cudaEventRecord(ev_upload, upload_stream));
            }

            for (size_t oi = 0; oi < output_names.size(); ++oi)
            {
                io_binding.BindOutput(output_names[oi].c_str(), out_current[oi]);
            }

            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_upload));
            {
                nvtx3::scoped_range ri{"inference"};
                Ort::RunOptions run_options;
                run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
                session.Run(run_options, io_binding);
                CUDA_CHECK(cudaEventRecord(ev_infer, stream));
            }

            // Throttle: wait for the previous inference before scheduling more work
            CUDA_CHECK(cudaEventSynchronize(ev_infer_cpu));
            CUDA_CHECK(cudaEventRecord(ev_infer_cpu, stream));

            // Download outputs
            {
                nvtx3::scoped_range rd{"download " + label};
                CUDA_CHECK(cudaStreamWaitEvent(download_stream, ev_infer));
                for (size_t oi = 0; oi < out_current.size(); ++oi)
                {
                    CUDA_CHECK(cudaMemcpyAsync(out_cpu[oi].GetTensorMutableRawData(),
                                               out_current[oi].GetTensorRawData(), ORTValueToBytes(out_flop[oi]),
                                               cudaMemcpyDeviceToHost, download_stream));
                }
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto stop = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << "GPU bindings per iteration [ms]: "
                  << static_cast<float>(us) / static_cast<float>(iterations) / 1000.f << "\n";
    }
    CUDA_CHECK(cudaProfilerStop());

    CUDA_CHECK(cudaEventDestroy(ev_infer));
    CUDA_CHECK(cudaEventDestroy(ev_infer_cpu));
    CUDA_CHECK(cudaEventDestroy(ev_upload));
    CUDA_CHECK(cudaStreamDestroy(upload_stream));
    CUDA_CHECK(cudaStreamDestroy(download_stream));
}

// =============================================================================
// IoBinding helper
// =============================================================================

Ort::IoBinding generate_io_binding(Ort::Session& session, std::map<std::string, std::vector<int64_t>> shape_overwrites,
                                   OrtAllocator* allocator)
{
    Ort::IoBinding binding(session);
    Ort::AllocatorWithDefaultOptions default_allocator;
    if (allocator == nullptr)
    {
        allocator = default_allocator;
    }

    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        auto name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        // session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo(); has lifetime issues
        // https://github.com/microsoft/onnxruntime/issues/24300
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        auto type = tensor_info.GetElementType();

        auto it = shape_overwrites.find(name.get());
        if (it != shape_overwrites.end())
        {
            shape = it->second;
        }
        else
        {
            for (auto& v : shape)
            {
                if (v == -1)
                    v = 1;
            }
        }

        auto tensor = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type);
        binding.BindInput(name.get(), tensor);
    }

    for (size_t i = 0; i < session.GetOutputCount(); ++i)
    {
        auto name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        binding.BindOutput(name.get(), default_allocator.GetInfo());
    }
    return binding;
}
