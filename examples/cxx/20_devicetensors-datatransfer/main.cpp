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

// EP-agnostic device tensors and data transfer using CopyTensors API.
// Model: https://github.com/yakhyo/fast-neural-style-transfer (MIT license)

#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>

#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

float cpuInputFloat[3 * image_dim * image_dim];
float cpuOutputFloat[3 * image_dim * image_dim];

int main()
{
    try
    {
        Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DeviceTensorsExample");
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.DisableMemPattern();
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.AddFreeDimensionOverrideByName("batch_size", 1);

        register_execution_providers(ortEnvironment);

        sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

        Ort::ConstEpDevice trt_ep_device = find_trt_rtx_device(ortEnvironment);
        if (!trt_ep_device)
        {
            LOG("Failed to find EP device with support for sync streams");
            return EXIT_FAILURE;
        }

        Ort::SyncStream stream = trt_ep_device.CreateSyncStream();
        size_t stream_addr = reinterpret_cast<size_t>(stream.GetHandle());

        std::string streamAddress = std::to_string(stream_addr);
        Ort::KeyValuePairs ep_options;
        ep_options.Add("user_compute_stream", streamAddress.c_str());
        ep_options.Add("has_user_compute_stream", "1");
        std::vector<Ort::ConstEpDevice> devices = {trt_ep_device};
        sessionOptions.AppendExecutionProvider_V2(ortEnvironment, devices, ep_options);

        Ort::Session session(ortEnvironment, toOrtFileString(get_executable_parent_path() / "candy.onnx").c_str(),
                             sessionOptions);
        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        if (num_inputs != 1 || num_outputs != 1)
        {
            LOG("This sample expects exactly one input and one output, got {} inputs and {} outputs.", num_inputs,
                num_outputs);
            return EXIT_FAILURE;
        }
        std::vector<Ort::ConstEpDevice> session_ep_devices = session.GetEpDeviceForInputs();

        std::vector<Ort::Value> cpu_input_tensors;
        std::vector<Ort::Value> cpu_output_tensors;
        std::vector<const OrtValue*> src_tensor_ptrs;
        std::vector<OrtValue*> dst_tensor_ptrs;
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;

        Ort::AllocatorWithDefaultOptions cpu_allocator;
        Ort::AllocatedStringPtr InputTensorName = session.GetInputNameAllocated(0, cpu_allocator);
        Ort::AllocatedStringPtr OutputTensorName = session.GetOutputNameAllocated(0, cpu_allocator);

        std::vector<int64_t> input_shape{1, 3, image_dim, image_dim};
        std::vector<float> input_data(3 * image_dim * image_dim, 0.0f);

        loadInputImage(cpuInputFloat, (get_executable_parent_path() / "Input.png").string().c_str());
        for (int i = 0; i < 3 * image_dim * image_dim; i++)
        {
            input_data[i] = cpuInputFloat[i];
        }

        Ort::Value input_value = Ort::Value::CreateTensor<float>(
            cpu_allocator.GetInfo(), input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
        cpu_input_tensors.push_back(std::move(input_value));

        Ort::Value output_value = Ort::Value::CreateTensor<float>(
            cpu_allocator.GetInfo(), cpuOutputFloat, 3 * image_dim * image_dim, input_shape.data(), input_shape.size());
        cpu_output_tensors.push_back(std::move(output_value));

        for (size_t idx = 0; idx < num_inputs; ++idx)
        {
            Ort::ConstEpDevice input_ep_device =
                idx < session_ep_devices.size() ? session_ep_devices[idx] : Ort::ConstEpDevice{};
            Ort::ConstMemoryInfo mem_info =
                input_ep_device ? input_ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT) : Ort::ConstMemoryInfo{};

            if (mem_info && mem_info.GetDeviceType() == OrtMemoryInfoDeviceType_GPU &&
                mem_info.GetDeviceMemoryType() == OrtDeviceMemoryType_DEFAULT)
            {
                Ort::UnownedAllocator allocator = ortEnvironment.GetSharedAllocator(mem_info);

                auto src_shape = cpu_input_tensors[idx].GetTensorTypeAndShapeInfo().GetShape();
                Ort::Value device_input_value =
                    Ort::Value::CreateTensor<float>(allocator, src_shape.data(), src_shape.size());

                auto dst_shape = cpu_output_tensors[idx].GetTensorTypeAndShapeInfo().GetShape();
                Ort::Value device_output_value =
                    Ort::Value::CreateTensor<float>(allocator, dst_shape.data(), dst_shape.size());

                src_tensor_ptrs.push_back(cpu_input_tensors[idx]);
                input_tensors.push_back(std::move(device_input_value));
                dst_tensor_ptrs.push_back(input_tensors.back());
                output_tensors.push_back(std::move(device_output_value));
            }
            else
            {
                input_tensors.push_back(std::move(cpu_input_tensors[idx]));
                output_tensors.push_back(std::move(cpu_output_tensors[idx]));
            }
        }

        std::vector<const char*> input_names = {"input"};
        std::vector<const char*> output_names = {"output"};
        Ort::Value output = Ort::Value(nullptr);

        Ort::IoBinding iobinding(session);
        if (!src_tensor_ptrs.empty())
        {
            Ort::ThrowOnError(ortEnvironment.CopyTensors(cpu_input_tensors, input_tensors, stream));
            iobinding.BindInput(InputTensorName.get(), input_tensors[0]);
            iobinding.BindOutput(OutputTensorName.get(), output_tensors[0]);
        }

        Ort::RunOptions run_options;
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

        if (!src_tensor_ptrs.empty())
        {
            for (int i = 0; i < 10; i++)
            {
                session.Run(Ort::RunOptions{}, iobinding);
                for (int j = 0; j < 10; j++)
                    session.Run(run_options, iobinding);
            }
        }
        else
        {
            for (int i = 0; i < 10; i++)
            {
                session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(), input_tensors.size(),
                            output_names.data(), output_tensors.data(), output_tensors.size());
                for (int j = 0; j < 10; j++)
                    session.Run(run_options, input_names.data(), input_tensors.data(), input_tensors.size(),
                                output_names.data(), output_tensors.data(), output_tensors.size());
            }
        }

        if (!src_tensor_ptrs.empty())
        {
            Ort::ThrowOnError(ortEnvironment.CopyTensors(output_tensors, cpu_output_tensors, stream));
        }

        float* result_data = src_tensor_ptrs.empty() ? output_tensors[0].GetTensorMutableData<float>() : cpuOutputFloat;
        saveOutputImage(result_data, (get_executable_parent_path() / "output.png").string().c_str());
    }
    catch (std::exception& ex)
    {
        LOG("Error: {}", ex.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
