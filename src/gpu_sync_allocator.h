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

#include "nv_includes.h"

#include "onnxruntime_c_api.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace trt_rtx_ep
{

//!
//! \class GpuSyncAllocator
//! \brief A synchronous nvinfer1::IGpuAsyncAllocator for TensorRT RTX that forwards to an
//!        OrtAllocator (the device's BFC arena, backed by cudaMalloc/cudaFree).
//!
//! \details Implements nvinfer1::IGpuAsyncAllocator (the preferred, non-deprecated allocator base)
//!          but services *every* request — including the stream-ordered allocateAsync()/
//!          deallocateAsync() entry points TensorRT prefers — synchronously by forwarding to the
//!          wrapped arena. The base class routes the deprecated allocate()/deallocate() methods
//!          through the async ones, so once installed via setGpuAllocator() no allocation path can
//!          reach cudaMallocAsync.
//!
//!          Registering an instance on an IRuntime or IBuilder via setGpuAllocator() forces
//!          TensorRT RTX off its default allocator, whose allocateAsync() uses cudaMallocAsync when
//!          CUDA memory pools are supported. On RTX products cudaMallocAsync has proven unreliable —
//!          e.g. under CiG it can exhaust the process virtual address space while VRAM remains
//!          available due to a known CUDA Windows driver bug. Forwarding to the device BFC arena (cudaMalloc)
//!          keeps allocation deterministic while reusing arena chunks.
//!
//! \note Header-only; included by tensorrt_rtx_execution_provider.h so the type is complete at
//!       the sync_gpu_allocator_ member declaration (std::unique_ptr member functions require a
//!       complete type wherever they are instantiated -- GCC rejects the forward-declared form
//!       even though MSVC accepts it). The wrapped arena
//!       validates sizes/alignment and treats a null free as a no-op, so this adapter adds no
//!       redundant checks of its own; it only forwards and, being noexcept, converts any exception
//!       from the arena into the nullptr/false the IGpuAllocator contract expects.
//!
//! \warning The lifetime of the allocator (and the arena it wraps) must exceed that of every
//!          runtime, builder, engine and execution context that uses it (a TensorRT requirement).
//!          The arena is owned by the factory (device_allocators), which outlives the EP.
//!
class GpuSyncAllocator final : public nvinfer1::IGpuAsyncAllocator
{
public:
    //!
    //! \brief Construct a synchronous allocator that forwards to \p arena.
    //!
    //! \param arena Non-owning pointer to the device BFC arena (cudaMalloc-backed) that services
    //!        allocations. Must be non-null and remain valid for the lifetime of this allocator.
    //!
    explicit GpuSyncAllocator(OrtAllocator* arena) noexcept
        : arena_(arena)
    {
    }

    ~GpuSyncAllocator() override = default;

    void* allocateAsync(uint64_t size, uint64_t /*alignment*/, nvinfer1::AllocatorFlags /*flags*/,
                        cudaStream_t /*stream*/) noexcept override
    {
        try
        {
            return arena_->Reserve(arena_, static_cast<size_t>(size));
        }
        catch (...)
        {
            return nullptr;
        }
    }

    bool deallocateAsync(void* memory, cudaStream_t /*stream*/) noexcept override
    {
        try
        {
            arena_->Free(arena_, memory);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

private:
    // Non-copyable, non-movable: the address is handed to TensorRT via setGpuAllocator().
    GpuSyncAllocator(const GpuSyncAllocator&) = delete;
    GpuSyncAllocator& operator=(const GpuSyncAllocator&) = delete;
    GpuSyncAllocator(GpuSyncAllocator&&) = delete;
    GpuSyncAllocator& operator=(GpuSyncAllocator&&) = delete;

    OrtAllocator* arena_;  //!< Non-owning; owned by the factory (device_allocators).
};

}  // namespace trt_rtx_ep
