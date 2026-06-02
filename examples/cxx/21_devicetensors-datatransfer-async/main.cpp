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

// Async data transfers using EP-provided ORT allocators for pinned host memory.
// Model: https://github.com/yakhyo/fast-neural-style-transfer (MIT license)

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

constexpr int kAsyncSubmissions = 10;

static Ort::UnownedAllocator get_ep_allocator(Ort::Env& env, Ort::ConstEpDevice ep_device,
                                              OrtDeviceMemoryType memory_type)
{
    Ort::ConstMemoryInfo memory_info = ep_device.GetMemoryInfo(memory_type);
    if (!memory_info)
    {
        THROW_ERROR("TensorRT RTX EP did not expose requested memory type {}", static_cast<int>(memory_type));
    }

    Ort::UnownedAllocator allocator = env.GetSharedAllocator(memory_info);
    if (!allocator)
    {
        THROW_ERROR("TensorRT RTX EP did not expose a shared allocator for requested memory type {}",
                    static_cast<int>(memory_type));
    }

    return allocator;
}

int main()
{
    try
    {
        Ort::Env ort_environment(ORT_LOGGING_LEVEL_WARNING, "DeviceTensorsAsyncTransferExample");
        Ort::SessionOptions session_options;
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.DisableMemPattern();
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.AddFreeDimensionOverrideByName("batch_size", 1);

        register_execution_providers(ort_environment);
        session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

        Ort::ConstEpDevice trt_ep_device = find_trt_rtx_device(ort_environment);
        if (!trt_ep_device)
        {
            LOG("Failed to find TensorRT RTX EP device");
            return EXIT_FAILURE;
        }

        Ort::SyncStream compute_stream = trt_ep_device.CreateSyncStream();
        Ort::SyncStream upload_stream = trt_ep_device.CreateSyncStream();

        size_t compute_stream_addr = reinterpret_cast<size_t>(compute_stream.GetHandle());
        std::string compute_stream_address = std::to_string(compute_stream_addr);
        Ort::KeyValuePairs ep_options;
        ep_options.Add("user_compute_stream", compute_stream_address.c_str());
        ep_options.Add("has_user_compute_stream", "1");
        std::vector<Ort::ConstEpDevice> devices = {trt_ep_device};
        session_options.AppendExecutionProvider_V2(ort_environment, devices, ep_options);

        Ort::Session session(ort_environment, toOrtFileString(get_executable_parent_path() / "candy.onnx").c_str(),
                             session_options);

        Ort::AllocatorWithDefaultOptions cpu_allocator;
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, cpu_allocator);
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, cpu_allocator);

        Ort::UnownedAllocator pinned_allocator =
            get_ep_allocator(ort_environment, trt_ep_device, OrtDeviceMemoryType_HOST_ACCESSIBLE);
        Ort::UnownedAllocator gpu_allocator =
            get_ep_allocator(ort_environment, trt_ep_device, OrtDeviceMemoryType_DEFAULT);

        std::vector<int64_t> shape{1, 3, image_dim, image_dim};
        std::vector<Ort::Value> pinned_input;
        std::vector<Ort::Value> pinned_output;
        std::vector<Ort::Value> gpu_input;
        std::vector<Ort::Value> gpu_output;
        pinned_input.emplace_back(Ort::Value::CreateTensor<float>(pinned_allocator, shape.data(), shape.size()));
        pinned_output.emplace_back(Ort::Value::CreateTensor<float>(pinned_allocator, shape.data(), shape.size()));
        gpu_input.emplace_back(Ort::Value::CreateTensor<float>(gpu_allocator, shape.data(), shape.size()));
        gpu_output.emplace_back(Ort::Value::CreateTensor<float>(gpu_allocator, shape.data(), shape.size()));

        loadInputImage(pinned_input[0].GetTensorMutableData<float>(),
                       (get_executable_parent_path() / "Input.png").string().c_str());

        OrtSyncNotificationImpl* upload_done = create_sync_notification(upload_stream);
        NotificationUniquePtr upload_done_ptr(upload_done, release_sync_notification);
        OrtSyncNotificationImpl* output_done = create_sync_notification(compute_stream);
        NotificationUniquePtr output_done_ptr(output_done, release_sync_notification);

        Ort::ThrowOnError(ort_environment.CopyTensors(pinned_input, gpu_input, upload_stream));
        CHECK_ORT(upload_done->Activate(upload_done));
        CHECK_ORT(upload_done->WaitOnDevice(upload_done, compute_stream));

        Ort::IoBinding io_binding(session);
        io_binding.BindInput(input_name.get(), gpu_input[0]);
        io_binding.BindOutput(output_name.get(), gpu_output[0]);

        Ort::RunOptions async_run_options;
        async_run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

        for (int i = 0; i < kAsyncSubmissions; ++i)
        {
            session.Run(async_run_options, io_binding);
        }

        Ort::ThrowOnError(ort_environment.CopyTensors(gpu_output, pinned_output, compute_stream));
        CHECK_ORT(output_done->Activate(output_done));
        CHECK_ORT(output_done->WaitOnHost(output_done));

        saveOutputImage(pinned_output[0].GetTensorMutableData<float>(),
                        (get_executable_parent_path() / "output.png").string().c_str());

        LOG("Submitted {} runs with {}=1 and synchronized only before reading output", kAsyncSubmissions,
            kOrtRunOptionsConfigDisableSynchronizeExecutionProviders);
    }
    catch (std::exception& ex)
    {
        LOG("Error: {}", ex.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
