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
#include <filesystem>
#include <format>
#include <iostream>

void loadInputImage(float* pData, const char* imageFileName);
void saveOutputImage(float* pData, const char* imageFileName);

#define DLL_NAME(name) (DLL_PREFIX name DLL_SUFFIX)
#if _WIN32
#define DLL_PREFIX ""
#define DLL_SUFFIX ".dll"
#else
#define DLL_PREFIX "lib"
#define DLL_SUFFIX ".so"
#endif
#define LOG(...) std::cout << std::format(__VA_ARGS__) << "\n"
#define THROW_ERROR(...)                                                       \
  do {                                                                         \
    LOG(__VA_ARGS__);                                                          \
    throw std::runtime_error(std::format(__VA_ARGS__));                        \
  } while (0)
#define CHECK_ORT(call)                                                        \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != nullptr) {                                                   \
      auto _msg = Ort::GetApi().GetErrorMessage(status);                       \
      Ort::GetApi().ReleaseStatus(status);                                     \
      THROW_ERROR("{}", _msg);                                                 \
    }                                                                          \
  }

std::filesystem::path get_executable_path();
std::filesystem::path get_executable_parent_path();
