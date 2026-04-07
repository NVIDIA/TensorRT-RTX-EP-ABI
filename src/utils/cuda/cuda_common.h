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

//!
//! \file cuda_common.h
//! \brief Minimal error handling for CUDA graph.
//!

#pragma once

#include "onnxruntime_cxx_api.h"

#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trt_rtx_ep
{

// Minimal error handling macros for cuda_graph.cc

#if !defined(CUDA_CALL_THROW)
#define CUDA_CALL_THROW(expr)                                                      \
    do                                                                             \
    {                                                                              \
        cudaError_t err = (expr);                                                  \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            std::ostringstream oss;                                                \
            oss << "CUDA error " << cudaGetErrorName(err) << ": "                  \
                << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str());                                   \
        }                                                                          \
    } while (0)
#endif

#if !defined(CUDA_RETURN_IF_ERROR)
#define CUDA_RETURN_IF_ERROR(expr)                                                 \
    do                                                                             \
    {                                                                              \
        cudaError_t err = (expr);                                                  \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            return Ort::GetApi().CreateStatus(ORT_EP_FAIL,                         \
                                              (std::string("CUDA error: ") +       \
                                               cudaGetErrorName(err) + ": " +      \
                                               cudaGetErrorString(err)).c_str());  \
        }                                                                          \
    } while (0)
#endif

#if !defined(ORT_ENFORCE)
#define ORT_ENFORCE(condition, ...)                                                \
    do                                                                             \
    {                                                                              \
        if (!(condition))                                                          \
        {                                                                          \
            std::ostringstream oss;                                                \
            oss << "Enforcement failed: " << #condition;                           \
            oss << " " << MakeString(__VA_ARGS__);                                 \
            throw std::runtime_error(oss.str());                                   \
        }                                                                          \
    } while (0)
#endif

#if !defined(ORT_THROW)
#define ORT_THROW(...)                                                             \
    do                                                                             \
    {                                                                              \
        std::ostringstream oss;                                                    \
        oss << MakeString(__VA_ARGS__);                                            \
        throw std::runtime_error(oss.str());                                       \
    } while (0)
#endif

// Simple logging - can be replaced with actual logging if needed
#if !defined(LOGS_DEFAULT)
#define LOGS_DEFAULT(level) std::cout
#endif

//!
//! \brief Helper to concatenate variadic arguments.
//!
template <typename... Args>
std::string MakeString(Args&&... args)
{
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

}  // namespace trt_rtx_ep
