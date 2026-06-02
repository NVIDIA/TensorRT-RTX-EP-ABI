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

// SyncStreams sample: async upload + inference on separate ORT sync streams.
// Model: https://github.com/yakhyo/fast-neural-style-transfer (MIT license)

#include <cstdlib>
#include <exception>

#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <stdio.h>

constexpr int LOADED_IMAGE_DIM = 224;
constexpr int INFERENCE_IMAGE_DIM = 224;
static_assert(LOADED_IMAGE_DIM == INFERENCE_IMAGE_DIM,
              "CopyTensors copies whole tensors. Preprocess on the CPU if these dimensions diverge.");

std::vector<float> cpuOutputFloat(3 * INFERENCE_IMAGE_DIM * INFERENCE_IMAGE_DIM);

int main()
{
    try
    {
        Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "SyncStreamsExample");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.DisableMemPattern();
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.AddFreeDimensionOverrideByName("batch_size", 1);

        register_execution_providers(ortEnvironment);

        sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

        Ort::ConstEpDevice trt_ep_device = find_trt_rtx_device(ortEnvironment);
        if (!trt_ep_device)
        {
            LOG("Error: could not select TensorRT RTX execution provider!");
            return EXIT_FAILURE;
        }

        Ort::SyncStream stream = trt_ep_device.CreateSyncStream();
        Ort::SyncStream upload_stream = trt_ep_device.CreateSyncStream();

        size_t stream_addr_val = reinterpret_cast<size_t>(stream.GetHandle());
        auto streamAddress = std::to_string(stream_addr_val);
        Ort::KeyValuePairs ep_options;
        ep_options.Add("user_compute_stream", streamAddress.c_str());
        ep_options.Add("has_user_compute_stream", "1");
        std::vector<Ort::ConstEpDevice> devices = {trt_ep_device};
        sessionOptions.AppendExecutionProvider_V2(ortEnvironment, devices, ep_options);

        Ort::Session session(ortEnvironment, toOrtFileString(get_executable_parent_path() / "candy.onnx").c_str(),
                             sessionOptions);

        std::vector<Ort::ConstEpDevice> session_ep_devices = session.GetEpDeviceForInputs();

        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;

        Ort::AllocatorWithDefaultOptions cpu_allocator;
        Ort::AllocatedStringPtr InputTensorName = session.GetInputNameAllocated(0, cpu_allocator);
        Ort::AllocatedStringPtr OutputTensorName = session.GetOutputNameAllocated(0, cpu_allocator);

        std::vector<int64_t> full_shape{1, 3, LOADED_IMAGE_DIM, LOADED_IMAGE_DIM};
        std::vector<int64_t> inference_shape{1, 3, INFERENCE_IMAGE_DIM, INFERENCE_IMAGE_DIM};

        Ort::Value full_cpu_tensor{nullptr};
        if (Ort::ConstMemoryInfo pinned_memory_info = trt_ep_device.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE))
        {
            try
            {
                Ort::UnownedAllocator pinned_allocator = ortEnvironment.GetSharedAllocator(pinned_memory_info);
                if (pinned_allocator)
                {
                    full_cpu_tensor =
                        Ort::Value::CreateTensor<float>(pinned_allocator, full_shape.data(), full_shape.size());
                }
            }
            catch (const Ort::Exception&)
            {
                full_cpu_tensor = Ort::Value{nullptr};
            }
        }
        if (!full_cpu_tensor)
        {
            full_cpu_tensor = Ort::Value::CreateTensor<float>(cpu_allocator, full_shape.data(), full_shape.size());
        }
        loadInputImage(full_cpu_tensor.GetTensorMutableData<float>(),
                       (get_executable_parent_path() / "Input.png").string().c_str());

        Ort::Value inference_cpu_output =
            Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), cpuOutputFloat.data(), cpuOutputFloat.size(),
                                            inference_shape.data(), inference_shape.size());

        if (session_ep_devices.empty() || !session_ep_devices[0])
        {
            THROW_ERROR("Session did not expose an EP device for its inputs");
        }
        Ort::ConstMemoryInfo mem_info = session_ep_devices[0].GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
        if (!mem_info)
        {
            THROW_ERROR("Session input EP device did not expose default device memory");
        }
        Ort::ConstHardwareDevice input_device = session_ep_devices[0].Device();
        Ort::ConstHardwareDevice selected_device = trt_ep_device.Device();
        if (!input_device || !selected_device)
        {
            THROW_ERROR("Failed to resolve the selected EP hardware device");
        }
        const int mem_device_id = mem_info.GetDeviceId();
        const uint32_t input_device_id = input_device.DeviceId();
        const uint32_t selected_device_id = selected_device.DeviceId();
        if (mem_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
            mem_info.GetDeviceMemoryType() != OrtDeviceMemoryType_DEFAULT ||
            input_device.Type() != OrtHardwareDeviceType_GPU || input_device.Type() != selected_device.Type() ||
            mem_device_id < 0 || static_cast<uint32_t>(mem_device_id) != input_device_id ||
            input_device_id != selected_device_id || mem_info.GetVendorId() != input_device.VendorId() ||
            input_device.VendorId() != selected_device.VendorId())
        {
            THROW_ERROR("Session input memory is not the selected GPU EP device memory: allocator={}, device_id={}, "
                        "input_ep_device_id={}, selected_ep_device_id={}",
                        mem_info.GetAllocatorName(), mem_device_id, input_device_id, selected_device_id);
        }
        Ort::UnownedAllocator gpu_allocator = ortEnvironment.GetSharedAllocator(mem_info);
        if (!gpu_allocator)
        {
            THROW_ERROR("EP did not expose a shared allocator for default device memory");
        }
        Ort::Value full_gpu_tensor =
            Ort::Value::CreateTensor<float>(gpu_allocator, full_shape.data(), full_shape.size());
        Ort::Value inference_gpu_input_tensor =
            Ort::Value::CreateTensor<float>(gpu_allocator, inference_shape.data(), inference_shape.size());
        Ort::Value inference_gpu_output_tensor =
            Ort::Value::CreateTensor<float>(gpu_allocator, inference_shape.data(), inference_shape.size());

        OrtSyncNotificationImpl* uploadNotification = create_sync_notification(upload_stream);
        NotificationUniquePtr upload_notification_ptr(uploadNotification, release_sync_notification);
        OrtSyncNotificationImpl* outputNotification = create_sync_notification(stream);
        NotificationUniquePtr output_notification_ptr(outputNotification, release_sync_notification);

        std::vector<Ort::Value> cpu_src_tensors;
        cpu_src_tensors.push_back(std::move(full_cpu_tensor));
        std::vector<Ort::Value> gpu_dst_tensors;
        gpu_dst_tensors.push_back(std::move(full_gpu_tensor));
        Ort::ThrowOnError(ortEnvironment.CopyTensors(cpu_src_tensors, gpu_dst_tensors, upload_stream));

        CHECK_ORT(uploadNotification->Activate(uploadNotification));
        CHECK_ORT(uploadNotification->WaitOnDevice(uploadNotification, stream));

        input_tensors.push_back(std::move(inference_gpu_input_tensor));
        Ort::ThrowOnError(ortEnvironment.CopyTensors(gpu_dst_tensors, input_tensors, stream));
        output_tensors.push_back(std::move(inference_gpu_output_tensor));

        Ort::IoBinding iobinding(session);
        iobinding.BindInput(InputTensorName.get(), input_tensors[0]);
        iobinding.BindOutput(OutputTensorName.get(), output_tensors[0]);

        session.Run(Ort::RunOptions{}, iobinding);

        std::vector<Ort::Value> output_dst_tensors;
        output_dst_tensors.push_back(std::move(inference_cpu_output));
        Ort::ThrowOnError(ortEnvironment.CopyTensors(output_tensors, output_dst_tensors, stream));
        CHECK_ORT(outputNotification->Activate(outputNotification));
        CHECK_ORT(outputNotification->WaitOnHost(outputNotification));

        saveOutputImage(cpuOutputFloat.data(), (get_executable_parent_path() / "output.png").string().c_str());
    }
    catch (const Ort::Exception& e)
    {
        printf("ONNX Runtime exception caught: %s\n", e.what());
        return -1;
    }
    catch (const std::exception& e)
    {
        printf("Runtime exception caught: %s\n", e.what());
        return -1;
    }

    return 0;
}
