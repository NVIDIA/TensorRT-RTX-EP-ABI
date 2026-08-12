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

#include "cuda_mempool_arena.h"

#include "utils/cuda/cuda_call.h"
#include "utils/ort_api_init.h"

#include <cuda.h>  // driver API: CUcontext, cuCtxGetCurrent/cuCtxSetCurrent (probe context save/restore)

#include <algorithm>
#include <sstream>
#include <string>

namespace trt_rtx_ep
{

// ======================================================================
// Construction / Destruction
// ======================================================================

CudaMempoolAllocator::CudaMempoolAllocator(const OrtMemoryInfo* memory_info, DeviceId device_id, const OrtApi& api,
                                           const OrtLogger& logger)
    : memory_info_(memory_info)
    , device_id_(device_id)
    , api_(api)
    , logger_(logger)
{
    OrtAllocator::version = NegotiatedOrtApiVersion();
    OrtAllocator::Alloc = AllocImpl;
    OrtAllocator::Free = FreeImpl;
    OrtAllocator::Info = InfoImpl;
    OrtAllocator::Reserve = ReserveImpl;
    OrtAllocator::GetStats = GetStatsImpl;
    OrtAllocator::AllocOnStream = AllocOnStreamImpl;
}

// static
OrtStatus* CudaMempoolAllocator::Create(const OrtMemoryInfo* memory_info, DeviceId device_id, const OrtApi& api,
                                        const OrtLogger& logger, std::unique_ptr<CudaMempoolAllocator>& out)
{
    out.reset(new CudaMempoolAllocator(memory_info, device_id, api, logger));

    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = static_cast<int>(device_id);

    // OOM here means the pool is unusable on this system: an expected fallback,
    // not an error. Other errors propagate.
    cudaError_t pool_create_err = cudaMemPoolCreate(&out->pool_, &props);
    if (pool_create_err == cudaErrorMemoryAllocation)
    {
        (void)cudaGetLastError();  // clear the sticky error
        std::ostringstream ss;
        ss << "CudaMempoolAllocator: cudaMemPoolCreate out-of-memory on device " << device_id
           << " (likely exhausted/fragmented GPU virtual-address space). Falling back to the "
              "synchronous cudaMalloc allocator (higher VRAM use; CUDA graph capture unavailable).";
        out->LogMessage(ORT_LOGGING_LEVEL_WARNING, ss.str().c_str());
        out.reset();     // signal "unsupported" to the caller
        return nullptr;  // not an error: caller falls back to the BFC arena
    }
    RETURN_IF_ERROR(CUDA_CALL(pool_create_err));

    {
        uint64_t max_threshold = UINT64_MAX;
        RETURN_IF_ERROR(
            CUDA_CALL(cudaMemPoolSetAttribute(out->pool_, cudaMemPoolAttrReleaseThreshold, &max_threshold)));
    }

    // Pool usability is probed lazily on the first allocation (EnsureProbedOnce), NOT here. A
    // create-time probe needs a current CUDA context, but ORT calls CreateAllocator at EP
    // registration when none exists yet — so cudaSetDevice would initialize the device PRIMARY
    // context: a persistent ~200MB second context a CIG app does not want, and that the runtime
    // will not release (so it cannot be made transient). Deferring the same
    // probe to the first allocation runs it under the caller's already-current (CIG/user) context
    // instead, so no cudaSetDevice and no spurious primary context. On probe failure DoAlloc
    // returns null and the compute path latches the device to the synchronous arena via
    // NoteAsyncMempoolFailure() — the identical graceful BFC fallback the eager probe produced.

    {
        std::ostringstream ss;
        ss << "CudaMempoolAllocator created on device " << device_id;
        out->LogMessage(ORT_LOGGING_LEVEL_INFO, ss.str().c_str());
    }

    return nullptr;
}

CudaMempoolAllocator::~CudaMempoolAllocator()
{
    // Best-effort: enqueue frees for remaining allocations on their recorded streams.
    for (auto& kv : alloc_map_)
    {
        (void)cudaFreeAsync(kv.first, kv.second.stream);
    }

    SyncAllKnownStreams_NoThrow();

    alloc_map_.clear();
    stream_map_.clear();

    (void)cudaDeviceSynchronize();

    if (pool_)
    {
        (void)cudaMemPoolTrimTo(pool_, 0);
        (void)cudaMemPoolDestroy(pool_);
        pool_ = nullptr;
    }
}

// ======================================================================
// OrtAllocator callbacks (static, forwarding to instance methods)
// ======================================================================

// static
void* ORT_API_CALL CudaMempoolAllocator::AllocImpl(OrtAllocator* this_, size_t size)
{
    if (this_ == nullptr)
        return nullptr;
    auto& self = *static_cast<CudaMempoolAllocator*>(this_);
    constexpr cudaStream_t kDefaultStream = static_cast<cudaStream_t>(0);
    return self.DoAlloc(size, kDefaultStream);
}

// static
void* ORT_API_CALL CudaMempoolAllocator::AllocOnStreamImpl(OrtAllocator* this_, size_t size, OrtSyncStream* stream)
{
    if (this_ == nullptr)
        return nullptr;
    auto& self = *static_cast<CudaMempoolAllocator*>(this_);
    cudaStream_t cuda_stream = self.ResolveCudaStream(stream);
    return self.DoAlloc(size, cuda_stream);
}

// static
void* ORT_API_CALL CudaMempoolAllocator::ReserveImpl(OrtAllocator* this_, size_t size)
{
    // Reserve is implemented as Alloc on the default stream so that
    // initializer memory is reclaimed when the session is torn down.
    return AllocImpl(this_, size);
}

// static
void ORT_API_CALL CudaMempoolAllocator::FreeImpl(OrtAllocator* this_, void* p)
{
    if (this_ == nullptr)
        return;
    static_cast<CudaMempoolAllocator*>(this_)->DoFree(p);
}

// static
const OrtMemoryInfo* ORT_API_CALL CudaMempoolAllocator::InfoImpl(const OrtAllocator* this_)
{
    if (this_ == nullptr)
        return nullptr;
    return static_cast<const CudaMempoolAllocator*>(this_)->memory_info_;
}

// static
OrtStatus* ORT_API_CALL CudaMempoolAllocator::GetStatsImpl(const OrtAllocator* this_, OrtKeyValuePairs** out) noexcept
{
    if (this_ == nullptr || out == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "CudaMempoolAllocator::GetStatsImpl: null argument");
    }

    const auto& self = *static_cast<const CudaMempoolAllocator*>(this_);
    self.api_.CreateKeyValuePairs(out);

    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(self.mutex_));
    self.api_.AddKeyValuePair(*out, "InUse", std::to_string(self.in_use_bytes_).c_str());
    self.api_.AddKeyValuePair(*out, "TotalAllocated", std::to_string(self.total_allocated_).c_str());
    self.api_.AddKeyValuePair(*out, "MaxInUse", std::to_string(self.max_bytes_in_use_).c_str());
    self.api_.AddKeyValuePair(*out, "NumAllocs", std::to_string(self.num_allocs_).c_str());
    self.api_.AddKeyValuePair(*out, "NumArenaShrinkages", std::to_string(self.num_arena_shrinkages_).c_str());
    self.api_.AddKeyValuePair(*out, "MaxAllocSize", std::to_string(self.max_alloc_size_).c_str());

    return nullptr;
}

// ======================================================================
// Instance methods
// ======================================================================

void CudaMempoolAllocator::EnsureProbedOnce(cudaStream_t stream)
{
    // Deferred pool-usability probe, moved out of Create() so EP registration does not initialize
    // the device primary context. Runs exactly once, under whatever context the caller already
    // made current (the CIG/user context during compute) — so NO cudaSetDevice and no spurious
    // second context. Mirrors the original create-time probe: a 1-byte allocation surfaces an
    // unusable pool before the first real allocation, and freeing it warms the pool (the pool's
    // release threshold is UINT64_MAX, so the backing is retained for real allocations).
    std::call_once(probe_once_,
                   [this, stream]()
                   {
                       void* probe = nullptr;
                       cudaError_t probe_err = cudaMallocFromPoolAsync(&probe, 1, pool_, stream);
                       if (probe_err == cudaSuccess)
                       {
                           (void)cudaStreamSynchronize(stream);
                           (void)cudaFreeAsync(probe, stream);
                           (void)cudaStreamSynchronize(stream);
                           return;  // probe_usable_ stays true
                       }

                       // Any failure means the pool is unusable: DoAlloc returns null and the
                       // compute path latches the device to the synchronous arena. Matches DoAlloc's
                       // own policy of never throwing on a pool allocation failure.
                       (void)cudaGetLastError();  // clear the sticky error
                       probe_usable_ = false;
                       std::ostringstream ss;
                       ss << "CudaMempoolAllocator: deferred usability probe failed on device " << device_id_ << " ("
                          << cudaGetErrorString(probe_err)
                          << "). Falling back to the synchronous "
                             "cudaMalloc arena (higher VRAM use; CUDA graph capture unavailable).";
                       LogMessage(ORT_LOGGING_LEVEL_WARNING, ss.str().c_str());
                   });
}

void* CudaMempoolAllocator::DoAlloc(size_t size, cudaStream_t cuda_stream)
{
    if (size == 0)
        return nullptr;

    // Probe the pool's usability the first time it is used (deferred from Create() to avoid
    // initializing the device primary context at EP registration). If the deferred probe found
    // the pool unusable, return null so the caller falls back to the synchronous arena.
    EnsureProbedOnce(cuda_stream);
    if (!probe_usable_)
    {
        return nullptr;
    }

    void* p = nullptr;
    cudaError_t err = cudaMallocFromPoolAsync(&p, size, pool_, cuda_stream);
    if (err != cudaSuccess)
    {
        (void)cudaGetLastError();  // clear sticky error before retry
        // Retry once: cudaMallocFromPoolAsync has an intermittent failure mode.
        cudaError_t retry_err = cudaMallocFromPoolAsync(&p, size, pool_, cuda_stream);
        if (retry_err != cudaSuccess)
        {
            // Return nullptr (don't throw) so the caller falls back to the sync
            // arena -- the pool can fragment at run time, after the create probe.
            (void)cudaGetLastError();
            std::ostringstream ss;
            ss << "CudaMempoolAllocator::DoAlloc: cudaMallocFromPoolAsync failed after retry: "
               << cudaGetErrorString(retry_err) << " (" << static_cast<int>(retry_err) << "), size=" << size
               << ", stream=" << reinterpret_cast<uintptr_t>(cuda_stream)
               << ". Returning nullptr; caller should fall back to a synchronous allocator.";
            LogMessage(ORT_LOGGING_LEVEL_ERROR, ss.str().c_str());
            return nullptr;
        }
    }

    // Synchronize default stream allocations so the pointer is immediately usable.
    if (cuda_stream == static_cast<cudaStream_t>(0))
    {
        (void)cudaStreamSynchronize(cuda_stream);
    }

    size_t snap_in_use = 0, snap_total = 0, snap_num = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        alloc_map_.emplace(p, AllocationRecord{size, cuda_stream});
        stream_map_[cuda_stream].insert(p);

        total_allocated_ += size;
        in_use_bytes_ += size;
        max_bytes_in_use_ = std::max(max_bytes_in_use_, in_use_bytes_);
        max_alloc_size_ = std::max(max_alloc_size_, size);
        ++num_allocs_;

        snap_in_use = in_use_bytes_;
        snap_total = total_allocated_;
        snap_num = num_allocs_;
    }

    {
        size_t pool_reserved = 0;
        size_t pool_used = 0;
        (void)cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);
        (void)cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent, &pool_used);

        std::ostringstream ss;
        ss << "CudaMempoolAllocator::DoAlloc: ptr=" << p << " size=" << size
           << " stream=" << reinterpret_cast<uintptr_t>(cuda_stream) << " | in_use=" << snap_in_use
           << " total_allocated=" << snap_total << " num_allocs=" << snap_num << " pool_reserved=" << pool_reserved
           << " pool_used=" << pool_used;
        LogMessage(ORT_LOGGING_LEVEL_INFO, ss.str().c_str());
    }

    return p;
}

void CudaMempoolAllocator::DoFree(void* p)
{
    if (!p)
        return;

    cudaStream_t s = static_cast<cudaStream_t>(0);
    size_t sz = 0;

    size_t snap_in_use = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = alloc_map_.find(p);
        if (it == alloc_map_.end())
        {
            std::ostringstream ss;
            ss << "CudaMempoolAllocator::DoFree: pointer " << p << " not found in allocation map; ignoring.";
            LogMessage(ORT_LOGGING_LEVEL_WARNING, ss.str().c_str());
            return;
        }

        s = it->second.stream;
        sz = it->second.bytes;
        alloc_map_.erase(it);

        auto sit = stream_map_.find(s);
        if (sit != stream_map_.end())
        {
            sit->second.erase(p);
            if (sit->second.empty())
            {
                stream_map_.erase(sit);
            }
        }

        in_use_bytes_ = (sz <= in_use_bytes_) ? (in_use_bytes_ - sz) : 0;
        snap_in_use = in_use_bytes_;
    }

    {
        std::ostringstream ss;
        ss << "CudaMempoolAllocator::DoFree: ptr=" << p << " size=" << sz
           << " stream=" << reinterpret_cast<uintptr_t>(s) << " | in_use_after=" << snap_in_use;
        LogMessage(ORT_LOGGING_LEVEL_INFO, ss.str().c_str());
    }

    cudaError_t free_err = cudaFreeAsync(p, s);
    if (free_err != cudaSuccess)
    {
        (void)cudaGetLastError();

        cudaPointerAttributes attr{};
        cudaError_t attr_err = cudaPointerGetAttributes(&attr, p);

        std::ostringstream ss;
        ss << "CudaMempoolAllocator::DoFree: cudaFreeAsync FAILED for ptr=" << p
           << " stream=" << reinterpret_cast<uintptr_t>(s) << " error=" << cudaGetErrorString(free_err) << " ("
           << static_cast<int>(free_err) << ")";

        if (attr_err == cudaSuccess)
        {
            ss << " | pointer_attr: type=" << attr.type << " device=" << attr.device
               << " devicePointer=" << attr.devicePointer;
        }
        else
        {
            ss << " | cudaPointerGetAttributes also failed: " << cudaGetErrorString(attr_err) << " ("
               << static_cast<int>(attr_err) << ")";
        }

        size_t pool_reserved = 0, pool_used = 0;
        (void)cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &pool_reserved);
        (void)cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent, &pool_used);
        ss << " | pool_reserved=" << pool_reserved << " pool_used=" << pool_used;

        // The original stream was likely destroyed during session teardown.
        // Retry on the default stream (always valid) so the memory is returned to the pool.
        if (free_err == cudaErrorInvalidResourceHandle)
        {
            constexpr cudaStream_t kDefaultStream = static_cast<cudaStream_t>(0);
            cudaError_t retry_err = cudaFreeAsync(p, kDefaultStream);
            if (retry_err == cudaSuccess)
            {
                ss << " | RECOVERED: retried cudaFreeAsync on default stream (0) succeeded";
                LogMessage(ORT_LOGGING_LEVEL_WARNING, ss.str().c_str());
                return;
            }
            (void)cudaGetLastError();
            ss << " | retry on default stream also failed: " << cudaGetErrorString(retry_err) << " ("
               << static_cast<int>(retry_err) << ")";
        }

        LogMessage(ORT_LOGGING_LEVEL_ERROR, ss.str().c_str());
    }
}

OrtStatus* CudaMempoolAllocator::Shrink()
{
    RETURN_IF_ERROR(CUDA_CALL(cudaMemPoolTrimTo(pool_, 0)));

    size_t current_in_use = 0;
    (void)CUDA_CALL(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent, &current_in_use));

    size_t reserved_size = 0;
    if (CUDA_CALL(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &reserved_size)) == nullptr)
    {
        std::ostringstream ss;
        ss << "CudaMempoolAllocator::Shrink: pool current_in_use=" << current_in_use
           << " reserved_after_trim=" << reserved_size;
        LogMessage(ORT_LOGGING_LEVEL_INFO, ss.str().c_str());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    MaybeRehashLocked();
    ++num_arena_shrinkages_;
    return nullptr;
}

// ======================================================================
// Helpers
// ======================================================================

cudaStream_t CudaMempoolAllocator::ResolveCudaStream(OrtSyncStream* stream) const
{
    if (!stream)
        return static_cast<cudaStream_t>(0);
    return static_cast<cudaStream_t>(api_.SyncStream_GetHandle(stream));
}

void CudaMempoolAllocator::MaybeRehashLocked()
{
    const size_t alloc_sz = alloc_map_.size();
    const size_t stream_sz = stream_map_.size();
    if (alloc_sz > 0)
        alloc_map_.reserve(alloc_sz);
    if (stream_sz > 0)
        stream_map_.reserve(stream_sz);
}

void CudaMempoolAllocator::SyncAllKnownStreams_NoThrow()
{
    for (const auto& kv : stream_map_)
    {
        (void)cudaStreamSynchronize(kv.first);
    }
}

void CudaMempoolAllocator::LogMessage(OrtLoggingLevel level, const char* msg) const
{
    (void)api_.Logger_LogMessage(&logger_, level, msg, ORT_FILE, __LINE__, __FUNCTION__);
}

// ======================================================================
// Stream-aware allocation (raw cudaStream_t)
// ======================================================================

void* CudaMempoolAllocator::AllocOnCudaStream(size_t size, cudaStream_t stream)
{
    return DoAlloc(size, stream);
}

void CudaMempoolAllocator::FreeOnCudaStream(void* p)
{
    DoFree(p);
}

}  // namespace trt_rtx_ep
