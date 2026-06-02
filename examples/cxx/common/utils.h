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

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifndef __cpp_lib_format
#    error "__cpp_lib_format is not defined! These samples require a C++20 compiler"
#endif

constexpr int image_dim = 224;
using OrtFileString = std::basic_string<ORTCHAR_T>;
using NotificationUniquePtr = std::unique_ptr<OrtSyncNotificationImpl, std::function<void(OrtSyncNotificationImpl*)>>;

OrtFileString toOrtFileString(const std::filesystem::path& path);
std::filesystem::path get_executable_path();
std::filesystem::path get_executable_parent_path();

void loadInputImage(float* pData, const char* imageFileName);
void saveOutputImage(float* pData, const char* imageFileName);

static const char* kNvTensorRTRTXExecutionProvider = "nv_tensorrt_rtx";

void register_execution_providers(Ort::Env& env);
Ort::ConstEpDevice find_trt_rtx_device(Ort::Env& env);
OrtSyncNotificationImpl* create_sync_notification(Ort::SyncStream& stream);
void release_sync_notification(OrtSyncNotificationImpl* notification);

#define DLL_NAME(name) (DLL_PREFIX name DLL_SUFFIX)
#ifdef _WIN32
#    define DLL_PREFIX ""
#    define DLL_SUFFIX ".dll"
#else
#    define DLL_PREFIX "lib"
#    define DLL_SUFFIX ".so"
#endif

#define LOG(...) std::cout << std::format(__VA_ARGS__) << "\n"
#define THROW_ERROR(...)                                    \
    do                                                      \
    {                                                       \
        LOG(__VA_ARGS__);                                   \
        throw std::runtime_error(std::format(__VA_ARGS__)); \
    } while (0)

#define DEFER(resource, x)                              \
    std::shared_ptr<void> resource##_finalizer(nullptr, \
                                               [&](...) \
                                               {        \
                                                   x;   \
                                               })

#define CHECK_ORT(call)                                                    \
    do                                                                     \
    {                                                                      \
        auto _ort_status = (call);                                         \
        if (_ort_status != nullptr)                                        \
        {                                                                  \
            std::string _msg = Ort::GetApi().GetErrorMessage(_ort_status); \
            Ort::GetApi().ReleaseStatus(_ort_status);                      \
            THROW_ERROR("{}", _msg);                                       \
        }                                                                  \
    } while (0)

#define STRINGFY(s) _STRINGFY(s)
#define _STRINGFY(s) #s
#define CHECK_CUDA(call)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        auto _cuda_status = (call);                                                                          \
        if (_cuda_status != cudaSuccess)                                                                     \
        {                                                                                                    \
            THROW_ERROR("Failed to execute CUDA call {}: error code {}", STRINGFY(call), int(_cuda_status)); \
        }                                                                                                    \
    } while (0)

inline static std::string to_lowercase(const std::string& s)
{
    std::string rtn;
    rtn.resize(s.size());
    std::transform(s.begin(), s.end(), rtn.begin(),
                   [](unsigned char c)
                   {
                       return std::tolower(c);
                   });
    return rtn;
}
