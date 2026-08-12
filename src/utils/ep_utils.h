// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "onnxruntime_cxx_api.h"

#include <cuda_runtime_api.h>

#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "make_string.h"

//!
//! \brief Container for API pointers used throughout the EP.
//!
struct ApiPtrs
{
    const OrtApi& ort_api;
    const OrtEpApi& ep_api;
    const OrtModelEditorApi& model_editor_api;
};

#define ENFORCE(condition, ...)                                \
    do                                                         \
    {                                                          \
        if (!(condition))                                      \
        {                                                      \
            throw std::runtime_error(MakeString(__VA_ARGS__)); \
        }                                                      \
    } while (false)

#define THROW(...) throw std::runtime_error(MakeString(__VA_ARGS__));

#define RETURN_IF_ORTSTATUS_ERROR(fn) RETURN_IF_ERROR(fn)

#define RETURN_IF_ERROR(fn)        \
    do                             \
    {                              \
        OrtStatus* _status = (fn); \
        if (_status != nullptr)    \
        {                          \
            return _status;        \
        }                          \
    } while (0)

#define RETURN_IF_ORT_STATUS_ERROR(fn) \
    do                                 \
    {                                  \
        auto _status = (fn);           \
        if (!_status.IsOK())           \
        {                              \
            return _status;            \
        }                              \
    } while (0)

#define RETURN_IF(cond, ...)                                                                 \
    do                                                                                       \
    {                                                                                        \
        if ((cond))                                                                          \
        {                                                                                    \
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL, MakeString(__VA_ARGS__).c_str()); \
        }                                                                                    \
    } while (0)

#define RETURN_IF_NOT(condition, ...) RETURN_IF(!(condition), __VA_ARGS__)

#define RETURN_ERROR(error_code, message)                                  \
    do                                                                     \
    {                                                                      \
        std::ostringstream _oss;                                           \
        _oss << message;                                                   \
        return Ort::GetApi().CreateStatus(error_code, _oss.str().c_str()); \
    } while (false)

#define MAKE_STATUS(error_code, msg) Ort::GetApi().CreateStatus(error_code, (msg));

#define THROW_IF_ERROR(expr)                               \
    do                                                     \
    {                                                      \
        auto _status = (expr);                             \
        if (_status != nullptr)                            \
        {                                                  \
            std::ostringstream oss;                        \
            oss << Ort::GetApi().GetErrorMessage(_status); \
            Ort::GetApi().ReleaseStatus(_status);          \
            throw std::runtime_error(oss.str());           \
        }                                                  \
    } while (0)

#define RETURN_FALSE_AND_PRINT_IF_ERROR(fn)                                  \
    do                                                                       \
    {                                                                        \
        OrtStatus* status = (fn);                                            \
        if (status != nullptr)                                               \
        {                                                                    \
            std::cerr << Ort::GetApi().GetErrorMessage(status) << std::endl; \
            return false;                                                    \
        }                                                                    \
    } while (0)

#if defined(_WIN32)
#define EP_WSTR(x) L##x
#define EP_FILE_INTERNAL(x) EP_WSTR(x)
#define EP_FILE EP_FILE_INTERNAL(__FILE__)
#else
#define EP_FILE __FILE__
#endif

#define LOG(level, message)                                                                                       \
    do                                                                                                            \
    {                                                                                                             \
        std::ostringstream ss;                                                                                    \
        ss << message;                                                                                            \
        Ort::GetApi().Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_##level, ss.str().c_str(), EP_FILE, __LINE__, \
                                        __FUNCTION__);                                                            \
    } while (false)

#define ENFORCE_EP(condition, ...)                             \
    do                                                         \
    {                                                          \
        if (!(condition))                                      \
        {                                                      \
            std::ostringstream oss;                            \
            oss << "EP_ENFORCE failed: " << #condition << " "; \
            oss << __VA_ARGS__;                                \
            throw std::runtime_error(oss.str());               \
        }                                                      \
    } while (false)

//!
//! \brief Helper to release Ort one or more objects obtained from the public C API at the end of their scope.
//!
template <typename T>
struct DeferOrtRelease
{
    DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
        : objects_(object_ptr)
        , count_(1)
        , release_func_(release_func)
    {
    }

    DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
        : objects_(objects)
        , count_(count)
        , release_func_(release_func)
    {
    }

    ~DeferOrtRelease()
    {
        if (objects_ != nullptr && count_ > 0)
        {
            for (size_t i = 0; i < count_; ++i)
            {
                if (objects_[i] != nullptr)
                {
                    release_func_(objects_[i]);
                    objects_[i] = nullptr;
                }
            }
        }
    }

    T** objects_ = nullptr;
    size_t count_ = 0;
    std::function<void(T*)> release_func_ = nullptr;
};

template <typename T>
using AllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;
