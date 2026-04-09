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

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "onnxruntime_c_api.h"
#include "tensorrt_rtx_allocator.h"

namespace trt_rtx_ep
{

//!
//! \brief Stream-aware CUDA allocator built on a private `cudaMemPool_t`.
//!
//! Assists with memory allocations in environments where a single process hosts
//! more than one CUDA session. This contrasts with BFCArena which only frees
//! memory on explicit Shrink() at end-of-run.
//!
//! ### Behavior
//! - Creates a **process-local** CUDA mempool for a specific device.
//! - All allocations use `cudaMallocFromPoolAsync()` on either the legacy
//!   default stream (0) or a caller-provided stream. The allocation stream is
//!   recorded for ordered free.
//! - `Free()` enqueues `cudaFreeAsync()` on the recorded stream.
//! - `Shrink()` trims the pool with `cudaMemPoolTrimTo(0)`.
//!
//! ### Thread-safety
//! - All updates to internal maps and statistics are guarded by `std::mutex`.
//!
struct CudaMempoolAllocator : OrtAllocator
{
    //!
    //! \brief Factory method. Creates the private CUDA mempool and sets pool attributes.
    //!
    //! \param memory_info ORT memory info descriptor for this allocator.
    //! \param device_id CUDA device on which to create the mempool.
    //! \param api ORT C API table.
    //! \param logger ORT logger instance.
    //! \param out Receives the newly created allocator on success.
    //! \return nullptr on success, or an OrtStatus describing the error.
    //!
    static OrtStatus* Create(
        const OrtMemoryInfo* memory_info,
        DeviceId device_id,
        const OrtApi& api,
        const OrtLogger& logger,
        std::unique_ptr<CudaMempoolAllocator>& out);

    ~CudaMempoolAllocator();

    //!
    //! \brief Trim the pool to zero. Allocated memory is not affected.
    //! Also rehashes internal maps under lock.
    //!
    //! \return nullptr on success, or an OrtStatus describing the error.
    //!
    OrtStatus* Shrink();

    void* AllocOnCudaStream(size_t size, cudaStream_t stream);
    void FreeOnCudaStream(void* p);

    CudaMempoolAllocator(const CudaMempoolAllocator&) = delete;
    CudaMempoolAllocator& operator=(const CudaMempoolAllocator&) = delete;
    CudaMempoolAllocator(CudaMempoolAllocator&&) = delete;
    CudaMempoolAllocator& operator=(CudaMempoolAllocator&&) = delete;

private:
    CudaMempoolAllocator(
        const OrtMemoryInfo* memory_info,
        DeviceId device_id,
        const OrtApi& api,
        const OrtLogger& logger);

    // ---- OrtAllocator callbacks (static) ----
    static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size);
    static void* ORT_API_CALL AllocOnStreamImpl(OrtAllocator* this_, size_t size, OrtSyncStream* stream);
    static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size);
    static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p);
    static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_);
    static OrtStatus* ORT_API_CALL GetStatsImpl(const OrtAllocator* this_, OrtKeyValuePairs** out) noexcept;

    // ---- Instance methods ----
    void* DoAlloc(size_t size, cudaStream_t cuda_stream);
    void DoFree(void* p);
    cudaStream_t ResolveCudaStream(OrtSyncStream* stream) const;
    void MaybeRehashLocked();
    void SyncAllKnownStreams_NoThrow();

    void LogMessage(OrtLoggingLevel level, const char* msg) const;

    struct AllocationRecord
    {
        size_t bytes;
        cudaStream_t stream;
    };

    const OrtMemoryInfo* memory_info_;
    DeviceId device_id_;
    const OrtApi& api_;
    const OrtLogger& logger_;
    cudaMemPool_t pool_{nullptr};

    std::mutex mutex_;
    std::unordered_map<void*, AllocationRecord> alloc_map_;
    std::unordered_map<cudaStream_t, std::unordered_set<void*>> stream_map_;

    size_t total_allocated_ = 0;
    size_t in_use_bytes_ = 0;
    size_t max_bytes_in_use_ = 0;
    size_t num_allocs_ = 0;
    size_t num_arena_shrinkages_ = 0;
    size_t max_alloc_size_ = 0;
};

}  // namespace trt_rtx_ep
