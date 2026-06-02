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
//! \file tensorrt_rtx_allocator.cc
//! \brief Implementation of memory allocators for TensorRT RTX Execution Provider
//!
//! \details Provides two allocator implementations:
//!          - TensorrtRtxAllocator: Device (GPU) memory allocation via CUDA
//!          - TensorrtRtxPinnedAllocator: Pinned host memory for efficient transfers
//!
//!          Both implement the OrtAllocator interface for ONNX Runtime integration.
//!
//! \note Requires CUDA runtime API. Allocators are not thread-safe for construction
//!       but memory operations (Alloc/Free) are thread-safe via CUDA thread-local context.
//!

#include "tensorrt_rtx_allocator.h"
#include "utils/ort_api_init.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace trt_rtx_ep
{

// Maximum allowed single allocation size (256GB) to prevent potential integer overflow
// or denial-of-service attacks through excessive memory requests
constexpr size_t kMaxAllowedAllocationSize = 256ULL * 1024 * 1024 * 1024;

//!
//! \brief Error handling utility for CUDA operations
//!
//! \param res CUDA error code to check
//! \throws Implementation-defined error handling (see tensorrt_rtx_execution_provider.cc)
//!
//! \note Forward declaration - actual implementation in tensorrt_rtx_execution_provider.cc
//!
void CUDA_RETURN_IF_ERROR(cudaError_t res);

// ========================================
// TensorrtRtxAllocator (Device Memory)
// ========================================

//!
//! \brief Constructs device memory allocator for specified CUDA device
//!
//! \details Initializes OrtAllocator base structure with function pointers
//!          that dispatch to this class's member functions via lambdas.
//!          Sets API version and leaves optional functions (Reserve, GetStats,
//!          AllocOnStream) as nullptr.
//!
//! \param mem_info OrtMemoryInfo describing the memory type and device
//! \param device_id CUDA device ID (GPU index) for memory allocation
//!
//! \note Does not validate mem_info for null; caller must ensure validity
//! \note Does not set CUDA device; device will be set during Alloc() calls
//!
TensorrtRtxAllocator::TensorrtRtxAllocator(const OrtMemoryInfo* mem_info, DeviceId device_id)
    : mem_info_(mem_info), device_id_(device_id)
{
    // Initialize OrtAllocator interface with current API version
    OrtAllocator::version = NegotiatedOrtApiVersion();

    // Set up function pointers to dispatch to member functions
    // Security checks are included in lambdas for defense in depth
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) -> void*
    {
        if (this_ == nullptr) return nullptr;
        return static_cast<TensorrtRtxAllocator*>(this_)->Alloc(size);
    };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p)
    {
        if (this_ == nullptr) return;
        static_cast<TensorrtRtxAllocator*>(this_)->Free(p);
    };
    OrtAllocator::Info = [](const OrtAllocator* this_) -> const OrtMemoryInfo*
    {
        if (this_ == nullptr) return nullptr;
        return static_cast<const TensorrtRtxAllocator*>(this_)->Info();
    };

    // Optional functions not implemented
    OrtAllocator::Reserve = nullptr;
    OrtAllocator::GetStats = nullptr;
    OrtAllocator::AllocOnStream = nullptr;
}

//!
//! \brief Validates current CUDA device matches allocator's device (debug only)
//!
//! \details In debug builds, queries current CUDA device and asserts it matches
//!          this allocator's device_id. In release builds, this is a no-op for
//!          performance. Use this after SetDevice() to verify device was set correctly.
//!
//! \param throw_when_fail If true, throws exception on CUDA errors; if false, ignores errors
//! \throws Implementation-defined exception if throw_when_fail is true and CUDA call fails
//!
//! \note Only active in debug builds (NDEBUG not defined)
//! \note Does not change device; use SetDevice() to change device
//!
void TensorrtRtxAllocator::CheckDevice(bool throw_when_fail) const
{
#if !defined(NDEBUG)
    // Check device to match at debug build for correctness validation
    // If device is expected to change, call cudaSetDevice instead of this check
    int current_device;
    auto cuda_err = cudaGetDevice(&current_device);
    if (cuda_err == cudaSuccess)
    {
        assert(current_device == device_id_);
    }
    else if (throw_when_fail)
    {
        CUDA_RETURN_IF_ERROR(cuda_err);
    }
#else
    (void)throw_when_fail;  // Suppress unused parameter warning in release builds
#endif
}

//!
//! \brief Sets current CUDA device to match allocator's device
//!
//! \details Queries current CUDA device and switches if necessary to match
//!          this allocator's device_id. If already on correct device, does nothing.
//!          Uses thread-local CUDA context, so safe to call from multiple threads.
//!
//! \param throw_when_fail If true, throws exception on CUDA errors; if false, silently ignores
//! \throws Implementation-defined exception if throw_when_fail is true and CUDA call fails
//!
//! \note Call this before device operations to ensure correct device context
//! \note Thread-safe: uses CUDA's thread-local device context
//! \pre CUDA runtime must be initialized
//! \post If successful, current thread's CUDA device is set to device_id_
//!
void TensorrtRtxAllocator::SetDevice(bool throw_when_fail) const
{
    int current_device;
    auto cuda_err = cudaGetDevice(&current_device);
    if (cuda_err == cudaSuccess)
    {
        int allocator_device_id = device_id_;
        if (current_device != allocator_device_id)
        {
            // Switch to the correct device for this allocator
            cuda_err = cudaSetDevice(allocator_device_id);
        }
    }

    if (cuda_err != cudaSuccess && throw_when_fail)
    {
        CUDA_RETURN_IF_ERROR(cuda_err);
    }
}

//!
//! \brief Allocates device memory on the GPU
//!
//! \details Sets CUDA device to this allocator's device_id, then allocates
//!          memory using cudaMalloc. Returns nullptr for zero-size requests
//!          without calling CUDA API.
//!
//! \param size Number of bytes to allocate
//! \return Pointer to allocated device memory, or nullptr if size is 0
//! \throws Implementation-defined exception if cudaMalloc fails
//!
//! \note BFCArena (ONNX Runtime's memory arena) handles exceptions and adjusts request sizes
//! \note Always sets device before allocation to ensure correct device context
//! \pre size should not exceed available device memory
//! \post On success, returns pointer to uninitialized device memory
//!
void* TensorrtRtxAllocator::Alloc(size_t size)
{
    // Security check: validate allocation size is within reasonable limits
    if (size > kMaxAllowedAllocationSize)
    {
        // Return nullptr for excessive allocation requests instead of crashing
        return nullptr;
    }

    SetDevice(true);  // Ensure we're on the correct CUDA device
    void* p = nullptr;
    if (size > 0)
    {
        // BFCArena was updated recently to handle the exception and adjust the request size
        CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&p, size));
    }
    return p;
}

//!
//! \brief Frees previously allocated device memory
//!
//! \details Safely frees device memory using cudaFree. Sets device context,
//!          validates device in debug builds, then frees memory. Silently handles
//!          errors during shutdown. No-op if pointer is nullptr.
//!
//! \param p Pointer to device memory to free (can be nullptr)
//!
//! \note noexcept: Never throws exceptions (safe for destructors)
//! \note Silently ignores CUDA errors - acceptable during process shutdown
//! \note Acceptable to call with nullptr (standard allocator behavior)
//! \post Device memory at p is released (if p was valid)
//!
void TensorrtRtxAllocator::Free(void* p) noexcept
{
    if (p == nullptr)
    {
        return;  // Standard behavior: free(nullptr) is no-op
    }
    SetDevice(false);    // Try to set device, but don't throw on failure
    CheckDevice(false);  // Validate device in debug builds, ignore errors
    cudaFree(p);         // Free memory - don't throw since cudaFree can fail during shutdown
}

//!
//! \brief Returns memory information descriptor for this allocator
//!
//! \details Provides OrtMemoryInfo that describes the memory type, device,
//!          and allocator ID. This allows ONNX Runtime to identify and
//!          manage different memory types.
//!
//! \return Pointer to OrtMemoryInfo structure (never nullptr)
//!
//! \note noexcept: Always returns valid pointer, never throws
//! \note Returned pointer is valid for lifetime of this allocator
//! \note Pointer is owned by this allocator, do not free
//!
const OrtMemoryInfo* TensorrtRtxAllocator::Info() const noexcept
{
    return mem_info_;
}

// ========================================
// TensorrtRtxPinnedAllocator (Pinned Host Memory)
// ========================================

//!
//! \brief Constructs pinned host memory allocator
//!
//! \details Initializes OrtAllocator base structure for pinned (page-locked)
//!          host memory. Pinned memory enables faster DMA transfers between
//!          host and device compared to pageable memory. Sets up function
//!          pointers to dispatch to member functions.
//!
//! \param mem_info OrtMemoryInfo describing the pinned memory type
//!
//! \note Does not validate mem_info for null; caller must ensure validity
//! \note Pinned memory is a limited resource; use sparingly
//! \note Faster transfers but higher memory pressure than regular malloc
//!
TensorrtRtxPinnedAllocator::TensorrtRtxPinnedAllocator(const OrtMemoryInfo* mem_info)
    : mem_info_(mem_info)
{
    // Initialize OrtAllocator interface with current API version
    OrtAllocator::version = NegotiatedOrtApiVersion();

    // Set up function pointers to dispatch to member functions
    // Security checks are included in lambdas for defense in depth
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) -> void*
    {
        if (this_ == nullptr) return nullptr;
        return static_cast<TensorrtRtxPinnedAllocator*>(this_)->Alloc(size);
    };
    OrtAllocator::Free = [](OrtAllocator* this_, void* p)
    {
        if (this_ == nullptr) return;
        static_cast<TensorrtRtxPinnedAllocator*>(this_)->Free(p);
    };
    OrtAllocator::Info = [](const OrtAllocator* this_) -> const OrtMemoryInfo*
    {
        if (this_ == nullptr) return nullptr;
        return static_cast<const TensorrtRtxPinnedAllocator*>(this_)->Info();
    };

    // Optional functions not implemented
    OrtAllocator::Reserve = nullptr;
    OrtAllocator::GetStats = nullptr;
    OrtAllocator::AllocOnStream = nullptr;
}

//!
//! \brief Allocates pinned (page-locked) host memory
//!
//! \details Allocates page-locked host memory using cudaMallocHost, which
//!          enables faster DMA transfers to/from GPU compared to regular malloc.
//!          Memory is accessible from both CPU and GPU. Returns nullptr for
//!          zero-size requests.
//!
//! \param size Number of bytes to allocate
//! \return Pointer to pinned host memory, or nullptr if size is 0
//! \throws Implementation-defined exception if cudaMallocHost fails
//!
//! \note Pinned memory is limited resource; failed allocations may indicate exhaustion
//! \note Pinned memory cannot be swapped to disk, increasing memory pressure
//! \note Use for frequently transferred data between host and device
//! \post On success, returns pointer to pinned host memory accessible from CPU
//!
void* TensorrtRtxPinnedAllocator::Alloc(size_t size)
{
    // Security check: validate allocation size is within reasonable limits
    if (size > kMaxAllowedAllocationSize)
    {
        // Return nullptr for excessive allocation requests instead of crashing
        return nullptr;
    }

    void* p = nullptr;
    if (size > 0)
    {
        // Allocate page-locked memory for faster host-device transfers
        CUDA_RETURN_IF_ERROR(cudaMallocHost((void**)&p, size));
    }
    return p;
}

//!
//! \brief Frees previously allocated pinned host memory
//!
//! \details Frees page-locked memory using cudaFreeHost. Silently handles
//!          errors that may occur during process shutdown. No-op if pointer
//!          is nullptr (standard allocator behavior).
//!
//! \param p Pointer to pinned host memory to free (can be nullptr)
//!
//! \note noexcept: Never throws exceptions (safe for destructors)
//! \note Silently ignores CUDA errors - acceptable during process shutdown
//! \note Acceptable to call with nullptr (standard allocator behavior)
//! \post Pinned memory at p is released (if p was valid)
//!
void TensorrtRtxPinnedAllocator::Free(void* p) noexcept
{
    if (p == nullptr)
    {
        return;  // Standard behavior: free(nullptr) is no-op
    }
    // Do not throw error since it's OK for cudaFreeHost to fail during shutdown
    cudaFreeHost(p);
}

//!
//! \brief Returns memory information descriptor for this allocator
//!
//! \details Provides OrtMemoryInfo that describes the pinned memory type.
//!          This allows ONNX Runtime to identify and distinguish between
//!          different memory types (device vs pinned host).
//!
//! \return Pointer to OrtMemoryInfo structure (never nullptr)
//!
//! \note noexcept: Always returns valid pointer, never throws
//! \note Returned pointer is valid for lifetime of this allocator
//! \note Pointer is owned by this allocator, do not free
//!
const OrtMemoryInfo* TensorrtRtxPinnedAllocator::Info() const noexcept
{
    return mem_info_;
}

}  // namespace trt_rtx_ep
