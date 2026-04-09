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

#include "utils.h"

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <limits.h>
#include <unistd.h>
#elif __APPLE__
#include <limits.h>
#include <mach-o/dyld.h>
#endif

OrtFileString toOrtFileString(const std::filesystem::path& path) {
#ifdef _WIN32
  return path.wstring();
#else
  return path.string();
#endif
}

std::filesystem::path get_executable_path() {
#ifdef _WIN32
  std::vector<wchar_t> pathBuf(MAX_PATH);
  DWORD length = GetModuleFileNameW(NULL, pathBuf.data(), static_cast<DWORD>(pathBuf.size()));
  while (length == pathBuf.size()) {
    pathBuf.resize(pathBuf.size() * 2);
    length = GetModuleFileNameW(NULL, pathBuf.data(), static_cast<DWORD>(pathBuf.size()));
  }
  if (length == 0) {
    std::cerr << "Error: GetModuleFileNameW failed with error " << GetLastError() << std::endl;
    return {};
  }
  return std::filesystem::path(pathBuf.data());
#elif __APPLE__
  std::vector<char> pathBuf(PATH_MAX);
  uint32_t length = pathBuf.size();
  if (_NSGetExecutablePath(pathBuf.data(), &length) != 0) {
    pathBuf.resize(length + 1);
    if (_NSGetExecutablePath(pathBuf.data(), &length) != 0) {
      std::cerr << "Error: _NSGetExecutablePath failed" << std::endl;
      return {};
    }
  }
  return std::filesystem::canonical(pathBuf.data());
#elif __linux__
  return std::filesystem::canonical(std::filesystem::read_symlink("/proc/self/exe"));
#endif
}

void register_execution_providers(Ort::Env& env) {
  try {
    env.RegisterExecutionProviderLibrary(
        onnxruntime::kNvTensorRTRTXExecutionProvider,
        toOrtFileString(DLL_NAME("onnxruntime_providers_nv_tensorrt_rtx")));
  } catch (std::exception&) {
  }
}

const OrtEpDevice* find_trt_rtx_device(Ort::Env& env) {
  OrtApi const& ortApi = Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_ep_devices;
  CHECK_ORT(ortApi.GetEpDevices(env, &ep_devices, &num_ep_devices));

  for (size_t i = 0; i < num_ep_devices; i++) {
    if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]),
              onnxruntime::kNvTensorRTRTXExecutionProvider) == 0) {
      return ep_devices[i];
    }
  }
  return nullptr;
}

void append_ep_v2(Ort::SessionOptions& so, Ort::Env& env,
                  const OrtEpDevice* device,
                  const std::vector<std::pair<std::string, std::string>>& options) {
  OrtApi const& ortApi = Ort::GetApi();
  std::vector<const char*> keys, values;
  for (auto& [k, v] : options) {
    keys.push_back(k.c_str());
    values.push_back(v.c_str());
  }
  CHECK_ORT(ortApi.SessionOptionsAppendExecutionProvider_V2(
      so, env, &device, 1,
      keys.empty() ? nullptr : keys.data(),
      keys.empty() ? nullptr : values.data(),
      keys.size()));
}
