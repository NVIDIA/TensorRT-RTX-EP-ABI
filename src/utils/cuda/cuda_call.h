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

#include "utils/ep_utils.h"  // For OrtStatus, MAKE_STATUS, THROW, ORT_EP_FAIL

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE)
{
    return "Unknown CUDA error type - no error string available";
}

// Specialization for CUDA error strings
template <>
inline const char* CudaErrString<cudaError_t>(cudaError_t x)
{
    return cudaGetErrorString(x);
}

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, OrtStatus*> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg, const char* file, const int line)
{
    if (retCode != successCode)
    {
        try
        {
            int currentCudaDevice = -1;
            cudaGetDevice(&currentCudaDevice);
            cudaGetLastError();  // clear last CUDA error
            static char str[1024];
            snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=? ; file=%s ; line=%d ; expr=%s; %s",
                     libName, (int)retCode, CudaErrString(retCode), currentCudaDevice,
                     file, line, exprString, msg);
            if constexpr (THRW)
            {
                // throw an exception with the error info
                THROW(str);
            }
            else
            {
                return MAKE_STATUS(ORT_EP_FAIL, str);
            }
        }
        catch (const std::exception& e)
        {
            // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
            if constexpr (THRW)
            {
                THROW(e.what());
            }
            else
            {
                return MAKE_STATUS(ORT_EP_FAIL, e.what());
            }
        }
    }
    if constexpr (!THRW)
    {
        return nullptr;
    }
}

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
