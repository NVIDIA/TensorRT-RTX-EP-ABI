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

#pragma once

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace onnxruntime {
constexpr const char* kNvTensorRTRTXExecutionProvider = "nv_tensorrt_rtx";
constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
}

#if ORT_API_VERSION < 23
#error "ONNX Runtime >= 1.23.0 required"
#endif

using OrtFileString = std::basic_string<ORTCHAR_T>;

OrtFileString toOrtFileString(const std::filesystem::path& path);
std::filesystem::path get_executable_path();

void register_execution_providers(Ort::Env& env);
const OrtEpDevice* find_trt_rtx_device(Ort::Env& env);
void append_ep_v2(Ort::SessionOptions& so, Ort::Env& env,
                  const OrtEpDevice* device,
                  const std::vector<std::pair<std::string, std::string>>& options = {});

#define DLL_NAME(name) (DLL_PREFIX name DLL_SUFFIX)
#if _WIN32
#define DLL_PREFIX ""
#define DLL_SUFFIX ".dll"
#else
#define DLL_PREFIX "lib"
#define DLL_SUFFIX ".so"
#endif

#define CHECK_ORT(call)                                    \
  {                                                        \
    auto _ort_status = (call);                             \
    if (_ort_status != nullptr) {                          \
      auto _msg = Ort::GetApi().GetErrorMessage(_ort_status); \
      Ort::GetApi().ReleaseStatus(_ort_status);            \
      throw std::runtime_error(std::string(_msg));         \
    }                                                      \
  }
