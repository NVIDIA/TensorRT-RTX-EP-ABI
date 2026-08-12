// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(_WIN32)
using CudaDriverLibraryHandle = HMODULE;
#else
using CudaDriverLibraryHandle = void*;
#endif

inline CudaDriverLibraryHandle OpenCudaDriverLibrary()
{
#if defined(_WIN32)
    return LoadLibraryA("nvcuda.dll");
#else
    return dlopen("libcuda.so.1", RTLD_LAZY);
#endif
}

inline void CloseCudaDriverLibrary(CudaDriverLibraryHandle handle)
{
#if defined(_WIN32)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

template <typename T>
inline T LoadCudaDriverSymbol(CudaDriverLibraryHandle handle, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<T>(GetProcAddress(handle, name));
#else
    return reinterpret_cast<T>(dlsym(handle, name));
#endif
}

class CudaDriverLoader
{
public:
    using cuCtxCreate_v4_t = CUresult (*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice);
    using cuCtxDestroy_t = CUresult (*)(CUcontext);
    using cuCtxSetCurrent_t = CUresult (*)(CUcontext);
    using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);

    CudaDriverLoader()
    {
        cuda_driver_dll_ = OpenCudaDriverLibrary();
        if (cuda_driver_dll_ != nullptr)
        {
            cuCtxCreate_v4_fn = LoadCudaDriverSymbol<cuCtxCreate_v4_t>(cuda_driver_dll_, "cuCtxCreate_v4");
            cuCtxDestroy_fn = LoadCudaDriverSymbol<cuCtxDestroy_t>(cuda_driver_dll_, "cuCtxDestroy");
            cuCtxSetCurrent_fn = LoadCudaDriverSymbol<cuCtxSetCurrent_t>(cuda_driver_dll_, "cuCtxSetCurrent");
            cuCtxGetCurrent_fn = LoadCudaDriverSymbol<cuCtxGetCurrent_t>(cuda_driver_dll_, "cuCtxGetCurrent");
        }
    }

    ~CudaDriverLoader()
    {
        if (cuda_driver_dll_ != nullptr)
        {
            CloseCudaDriverLibrary(cuda_driver_dll_);
        }
    }

    CudaDriverLoader(const CudaDriverLoader&) = delete;
    CudaDriverLoader& operator=(const CudaDriverLoader&) = delete;

    bool IsLoaded() const
    {
        return cuda_driver_dll_ != nullptr && cuCtxCreate_v4_fn != nullptr && cuCtxDestroy_fn != nullptr &&
               cuCtxSetCurrent_fn != nullptr && cuCtxGetCurrent_fn != nullptr;
    }

    cuCtxCreate_v4_t cuCtxCreate_v4_fn = nullptr;
    cuCtxDestroy_t cuCtxDestroy_fn = nullptr;
    cuCtxSetCurrent_t cuCtxSetCurrent_fn = nullptr;
    cuCtxGetCurrent_t cuCtxGetCurrent_fn = nullptr;

private:
    CudaDriverLibraryHandle cuda_driver_dll_ = nullptr;
};
