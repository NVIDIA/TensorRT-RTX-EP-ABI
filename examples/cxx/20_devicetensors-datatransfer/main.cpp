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
#include <string>
constexpr int image_dim = 240;

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

namespace onnxruntime {
constexpr const char* kNvTensorRTRTXExecutionProvider = "nv_tensorrt_rtx";
constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
}
#include <stdio.h>

#include <format>

#include "utils.h"

using StreamUniquePtr = std::unique_ptr<OrtSyncStream, std::function<void(OrtSyncStream*)>>;
using OrtFileString = std::basic_string<ORTCHAR_T>;

static OrtFileString toOrtFileString(const std::filesystem::path& path) {
#ifdef _WIN32
  return path.wstring();
#else
  return path.string();
#endif
}

float cpuInputFloat[3 * image_dim * image_dim];
float cpuOutputFloat[3 * image_dim * image_dim];

#define PROVIDER_LIB_PAIR(NAME) \
  std::pair { NAME, DLL_NAME("onnxruntime_providers_" NAME) }

static void register_execution_providers(Ort::Env& env) {
  // clang-format off
  std::array provider_libraries{
      PROVIDER_LIB_PAIR("nv_tensorrt_rtx")
  };
  // clang-format on

  for (auto& [registration_name, dll] : provider_libraries) {
    auto providers_library = get_executable_path().parent_path() / dll;
    if (!std::filesystem::is_regular_file(providers_library)) {
      LOG("{} does not exist! Skipping execution provider", providers_library.string());
      continue;
    }
    try {
      env.RegisterExecutionProviderLibrary(registration_name, toOrtFileString(providers_library));
    } catch (std::exception& ex) {
      LOG("Failed to register {}! Skipping execution provider", providers_library.string());
    }
  }
}

int main() {
  try {
    OrtApi const& ortApi = Ort::GetApi();
    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "DeviceTensorsExample");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    CHECK_ORT(ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1));

    register_execution_providers(ortEnvironment);

    sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_ep_devices;
    CHECK_ORT(ortApi.GetEpDevices(ortEnvironment, &ep_devices, &num_ep_devices));

    const OrtEpDevice* trt_ep_device = nullptr;
    for (uint32_t i = 0; i < num_ep_devices; i++) {
      if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]),
                 onnxruntime::kNvTensorRTRTXExecutionProvider) == 0) {
        trt_ep_device = ep_devices[i];
        break;
      }
    }
    if (trt_ep_device == nullptr) {
      LOG("Failed to find EP device with support for sync streams");
      return EXIT_FAILURE;
    }

    OrtSyncStream* stream = nullptr;
    StreamUniquePtr stream_ptr;
    OrtSyncStream* upload_stream = nullptr;
    StreamUniquePtr upload_stream_ptr;
    CHECK_ORT(ortApi.CreateSyncStreamForEpDevice(trt_ep_device, nullptr, &stream));
    CHECK_ORT(ortApi.CreateSyncStreamForEpDevice(trt_ep_device, nullptr, &upload_stream));
    stream_ptr = StreamUniquePtr(stream, [ortApi](OrtSyncStream* stream) { ortApi.ReleaseSyncStream(stream); });
    upload_stream_ptr = StreamUniquePtr(
        upload_stream, [ortApi](OrtSyncStream* upload_stream) { ortApi.ReleaseSyncStream(upload_stream); });
    size_t stream_addr = reinterpret_cast<size_t>(ortApi.SyncStream_GetHandle(stream));

    std::string streamAddress = std::to_string(stream_addr);
    const char* option_keys[] = {"user_compute_stream", "has_user_compute_stream"};
    const char* option_values[] = {streamAddress.c_str(), "1"};
    for (size_t i = 0; i < num_ep_devices; i++) {
      if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]), onnxruntime::kCpuExecutionProvider) != 0)
        CHECK_ORT(ortApi.SessionOptionsAppendExecutionProvider_V2(sessionOptions, ortEnvironment, &ep_devices[i], 1,
                                                                  option_keys, option_values, 2));
    }

    Ort::Session session(ortEnvironment, toOrtFileString(get_executable_parent_path() / "candy.onnx").c_str(),
                         sessionOptions);
    size_t num_inputs = session.GetInputCount();

    const OrtEpDevice* session_epDevices = {nullptr};
    CHECK_ORT(ortApi.SessionGetEpDeviceForInputs(session, &session_epDevices, num_inputs));

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
    for (int i = 0; i < 3 * image_dim * image_dim; i++) {
      input_data[i] = cpuInputFloat[i];
    }

    Ort::Value input_value = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), input_data.data(),
                                                             input_data.size(), input_shape.data(), input_shape.size());
    cpu_input_tensors.push_back(std::move(input_value));

    Ort::Value output_value = Ort::Value::CreateTensor<float>(
        cpu_allocator.GetInfo(), cpuOutputFloat, 3 * image_dim * image_dim, input_shape.data(), input_shape.size());
    cpu_output_tensors.push_back(std::move(output_value));

    OrtMemoryInfo* input_memory_info_agnostic = nullptr;
    for (size_t idx = 0; idx < num_inputs; ++idx) {
      const OrtHardwareDevice* hw_device = ortApi.EpDevice_Device(session_epDevices);
      auto vID = ortApi.HardwareDevice_VendorId(hw_device);
      CHECK_ORT(ortApi.CreateMemoryInfo_V2("Input_Agnostic", OrtMemoryInfoDeviceType_GPU, /*vendor_id*/ vID,
                                           /*device_id*/ 0, OrtDeviceMemoryType_DEFAULT, /*default alignment*/ 0,
                                           OrtArenaAllocator, &input_memory_info_agnostic));

      const OrtMemoryInfo* mem_info = input_memory_info_agnostic;
      OrtDeviceMemoryType mem_type = ortApi.MemoryInfoGetDeviceMemType(mem_info);
      OrtMemoryInfoDeviceType device_type;
      ortApi.MemoryInfoGetDeviceType(mem_info, &device_type);
      const char* name;
      CHECK_ORT(ortApi.MemoryInfoGetName(mem_info, &name));

      if (device_type == OrtMemoryInfoDeviceType_GPU && mem_type == OrtDeviceMemoryType_DEFAULT) {
        OrtAllocator* allocator = nullptr;
        CHECK_ORT(ortApi.GetSharedAllocator(ortEnvironment, mem_info, &allocator));

        auto src_shape = cpu_input_tensors[idx].GetTensorTypeAndShapeInfo().GetShape();
        Ort::Value device_input_value = Ort::Value::CreateTensor<float>(allocator, src_shape.data(), src_shape.size());

        auto dst_shape = cpu_output_tensors[idx].GetTensorTypeAndShapeInfo().GetShape();
        Ort::Value device_output_value = Ort::Value::CreateTensor<float>(allocator, dst_shape.data(), dst_shape.size());

        src_tensor_ptrs.push_back(cpu_input_tensors[idx]);
        input_tensors.push_back(std::move(device_input_value));
        dst_tensor_ptrs.push_back(input_tensors.back());
        output_tensors.push_back(std::move(device_output_value));
      } else {
        input_tensors.push_back(std::move(cpu_input_tensors[idx]));
        output_tensors.push_back(std::move(cpu_output_tensors[idx]));
      }
    }

    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};
    Ort::Value output = Ort::Value(nullptr);

    Ort::IoBinding iobinding(session);
    if (!src_tensor_ptrs.empty()) {
      CHECK_ORT(ortApi.CopyTensors(ortEnvironment, src_tensor_ptrs.data(), dst_tensor_ptrs.data(), stream,
                                   src_tensor_ptrs.size()));
      iobinding.BindInput(InputTensorName.get(), input_tensors[0]);
      iobinding.BindOutput(OutputTensorName.get(), output_tensors[0]);
    }

    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");

    if (!src_tensor_ptrs.empty()) {
      for (int i = 0; i < 10; i++) {
        session.Run(Ort::RunOptions{}, iobinding);
        for (int j = 0; j < 10; j++) session.Run(run_options, iobinding);
      }
    } else {
      for (int i = 0; i < 10; i++) {
        session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(), input_tensors.size(),
                    output_names.data(), output_tensors.data(), output_tensors.size());
        for (int j = 0; j < 10; j++)
          session.Run(run_options, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(),
                      output_tensors.data(), output_tensors.size());
      }
    }

    if (!src_tensor_ptrs.empty()) {
      std::vector<const OrtValue*> output_src_tensor_ptrs = {output_tensors[0]};
      std::vector<OrtValue*> output_dst_tensor_ptrs = {cpu_output_tensors[0]};
      CHECK_ORT(
          ortApi.CopyTensors(ortEnvironment, output_src_tensor_ptrs.data(), output_dst_tensor_ptrs.data(), stream, 1));
    }

    float* result_data = src_tensor_ptrs.empty()
        ? output_tensors[0].GetTensorMutableData<float>()
        : cpuOutputFloat;
    saveOutputImage(result_data, (get_executable_parent_path() / "output.png").string().c_str());

    ortApi.ReleaseMemoryInfo(input_memory_info_agnostic);
  } catch (std::exception& ex) {
    LOG("Error: {}", ex.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
