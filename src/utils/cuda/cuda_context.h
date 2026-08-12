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

#include "utils/ep_utils.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>

namespace trt_rtx_ep
{

inline const char* CudaDriverErrorString(CUresult result)
{
    const char* error_string = nullptr;
    cuGetErrorString(result, &error_string);
    return error_string ? error_string : "unknown error";
}

inline OrtStatus* CudaDriverStatus(const OrtApi& api, CUresult result, const char* what,
                                   OrtErrorCode error_code = ORT_EP_FAIL)
{
    if (result == CUDA_SUCCESS)
    {
        return nullptr;
    }

    std::string message = what;
    message += ": ";
    message += CudaDriverErrorString(result);
    return api.CreateStatus(error_code, message.c_str());
}

inline void CudaDriverThrowIfError(CUresult result, const char* what)
{
    if (result == CUDA_SUCCESS)
    {
        return;
    }

    THROW("CUDA Driver Error: ", what, ": ", CudaDriverErrorString(result), " (error code: ", static_cast<int>(result),
          ")");
}

inline void CudaRuntimeThrowIfError(cudaError_t result, const char* what)
{
    if (result == cudaSuccess)
    {
        return;
    }

    THROW("CUDA Error: ", what, ": ", cudaGetErrorString(result), " (error code: ", static_cast<int>(result), ")");
}

inline OrtStatus* GetCudaStreamContext(const OrtApi& api, cudaStream_t stream, CUcontext* context)
{
    if (context == nullptr)
    {
        return api.CreateStatus(ORT_INVALID_ARGUMENT, "GetCudaStreamContext: context output is null");
    }

    return CudaDriverStatus(api, cuStreamGetCtx(reinterpret_cast<CUstream>(stream), context), "cuStreamGetCtx");
}

inline CUcontext GetCudaStreamContextOrThrow(cudaStream_t stream)
{
    CUcontext context = nullptr;
    CudaDriverThrowIfError(cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &context), "cuStreamGetCtx");
    return context;
}

class ScopedCudaContext
{
public:
    explicit ScopedCudaContext(CUcontext context)
        : target_(context)
    {
        if (target_ == nullptr)
        {
            return;
        }

        CudaDriverThrowIfError(cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
        if (previous_ != target_)
        {
            CudaDriverThrowIfError(cuCtxSetCurrent(target_), "cuCtxSetCurrent");
            changed_ = true;
        }
    }

    ScopedCudaContext(const OrtApi& api, CUcontext context, OrtStatus** status) noexcept
        : target_(context)
    {
        if (status == nullptr || target_ == nullptr)
        {
            return;
        }

        *status = CudaDriverStatus(api, cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
        if (*status != nullptr)
        {
            return;
        }

        if (previous_ != target_)
        {
            *status = CudaDriverStatus(api, cuCtxSetCurrent(target_), "cuCtxSetCurrent");
            changed_ = (*status == nullptr);
        }
    }

    ScopedCudaContext(const ScopedCudaContext&) = delete;
    ScopedCudaContext& operator=(const ScopedCudaContext&) = delete;

    ~ScopedCudaContext()
    {
        if (changed_)
        {
            (void)cuCtxSetCurrent(previous_);
        }
    }

private:
    CUcontext target_ = nullptr;
    CUcontext previous_ = nullptr;
    bool changed_ = false;
};

class ScopedCudaContextNoThrow
{
public:
    explicit ScopedCudaContextNoThrow(CUcontext context) noexcept
        : target_(context)
    {
        if (target_ == nullptr)
        {
            return;
        }

        if (cuCtxGetCurrent(&previous_) == CUDA_SUCCESS && previous_ != target_ &&
            cuCtxSetCurrent(target_) == CUDA_SUCCESS)
        {
            changed_ = true;
        }
    }

    ScopedCudaContextNoThrow(const ScopedCudaContextNoThrow&) = delete;
    ScopedCudaContextNoThrow& operator=(const ScopedCudaContextNoThrow&) = delete;

    ~ScopedCudaContextNoThrow()
    {
        if (changed_)
        {
            (void)cuCtxSetCurrent(previous_);
        }
    }

private:
    CUcontext target_ = nullptr;
    CUcontext previous_ = nullptr;
    bool changed_ = false;
};

class ScopedCudaContextPush
{
public:
    explicit ScopedCudaContextPush(int device_id)
        : pushed_(true)
    {
        CUcontext context = nullptr;
        CudaDriverThrowIfError(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
        if (context == nullptr)
        {
            CudaRuntimeThrowIfError(cudaSetDevice(device_id), "cudaSetDevice");
            CudaDriverThrowIfError(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
        }

        CudaDriverThrowIfError(cuCtxPushCurrent(context), "cuCtxPushCurrent");
    }

    explicit ScopedCudaContextPush(CUcontext context)
        : pushed_(context != nullptr)
    {
        if (context != nullptr)
        {
            CudaDriverThrowIfError(cuCtxPushCurrent(context), "cuCtxPushCurrent");
        }
    }

    ScopedCudaContextPush(const ScopedCudaContextPush&) = delete;
    ScopedCudaContextPush& operator=(const ScopedCudaContextPush&) = delete;

    ~ScopedCudaContextPush()
    {
        if (pushed_)
        {
            (void)cuCtxPopCurrent(nullptr);
        }
    }

private:
    bool pushed_ = false;
};

}  // namespace trt_rtx_ep
