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

#include "ep_arena.h"
#include "tensorrt_rtx_allocator.h"

#include "onnxruntime_c_api.h"

#include <cuda.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cuda_mempool_arena.h"
#include "ep_utils.h"

// Forward declarations for ORT types
struct OrtEpFactory;
struct OrtHardwareDevice;
struct OrtEpDevice;
struct OrtKeyValuePairs;
struct OrtSessionOptions;
struct OrtLogger;
struct OrtEp;
struct OrtAllocator;
struct OrtMemoryInfo;
struct OrtDataTransferImpl;
struct OrtStatus;
struct OrtCustomOpDomain;

using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

namespace trt_rtx_ep
{

// Forward declarations
struct TensorrtRtxDataTransfer;

//!
//! \brief Plugin TensorRT RTX EP factory that can create an OrtEp and return information about the supported hardware
//! devices.
//!
//! \details This factory manages device enumeration, allocators, and EP instance creation.
//!
struct TensorrtRtxExecutionProviderFactory
    : public OrtEpFactory
    , public ApiPtrs
{
public:
    TensorrtRtxExecutionProviderFactory(const char* ep_name, const OrtLogger& default_logger, ApiPtrs apis);
    virtual ~TensorrtRtxExecutionProviderFactory();

    OrtStatus* CreateMemoryInfoForDevices(int num_devices);

    // Device memory and pinned/host-accessible memory are required for allocator and data transfer.
    // These are the OrtMemoryInfo instances required for that.
    // Current implementation uses one default OrtMemoryInfo and one host accessible OrtMemoryInfo per EP device.
    std::unordered_map<uint32_t, MemoryInfoUniquePtr> device_memory_infos;  //!< Device ID -> memory info
    std::unordered_map<uint32_t, MemoryInfoUniquePtr> pinned_memory_infos;  //!< Device ID -> pinned memory info

    // Keeps allocators per EP device in factory so they can be shared across sessions.
    std::unordered_map<uint32_t, std::unique_ptr<ArenaAllocator>>
        device_allocators;  //!< Device ID -> BFC arena allocator
    // CUDA mempool allocators (used only for activation memory allocation.)
    std::unordered_map<uint32_t, std::unique_ptr<CudaMempoolAllocator>>
        device_mempool_allocators;  //!< Device ID -> mempool allocator
    std::unordered_map<uint32_t, std::unique_ptr<ArenaAllocator>> pinned_allocators;  //!< Device ID -> pinned allocator

    std::vector<const OrtMemoryDevice*> device_mem_devices;
    std::vector<const OrtMemoryDevice*> pinned_mem_devices;
    std::unique_ptr<TensorrtRtxDataTransfer> data_transfer_impl;  //!< Data transfer implementation for this factory

    //! \brief Shrink CUDA mempool allocator for the given device ID.
    //!
    //! Calls Shrink() on the CUDA mempool allocator (if present) for the device.
    //! \param[in]  device_id  GPU device ID to shrink CUDA mempool allocator for.
    //! \return nullptr on success, or an OrtStatus describing the error.
    OrtStatus* ShrinkCudaMempoolAllocators(uint32_t device_id);

    //! \brief Whether the async CUDA mempool is currently usable for the device:
    //! created+probed and not latched off by a runtime failure. Used by the compute
    //! path and CUDA graph capture to pick async vs synchronous behavior.
    //! Thread-safe: device_mempool_allocators is immutable after session setup; the
    //! disabled set is guarded by mempool_state_mutex_.
    //! \param[in]  device_id  GPU device ID.
    bool IsAsyncMempoolEnabledForDevice(uint32_t device_id) const;

    //! \brief The active async mempool allocator for the device, or nullptr if the
    //! pool is absent or latched off (caller then uses the BFC arena).
    //! \param[in]  device_id  GPU device ID.
    CudaMempoolAllocator* GetActiveMempoolForDevice(uint32_t device_id);

    //! \brief Get (creating on first use) the per-device BFC arena in device_allocators. Lets the EP
    //! reuse the same cudaMalloc/cudaFree arena ORT uses for the device (e.g. to wrap it in a
    //! GpuSyncAllocator and install it via setGpuAllocator) even when the EP constructor runs before
    //! ORT has requested the allocator. The arena is owned by the factory, so it outlives every
    //! EP/runtime that uses it.
    //! \param[in]  device_id  GPU device ID.
    //! \return The arena as an OrtAllocator*, or nullptr if it could not be created (e.g. no memory
    //!         info for the device or arena construction failed); a warning is logged in that case.
    OrtAllocator* GetOrCreateDeviceArena(uint32_t device_id);

    //! \brief Latch the device's async mempool off after a runtime allocation
    //! failure, so subsequent runs use the synchronous cudaMalloc arena. Idempotent;
    //! logs one WARNING per device. The pool object is kept alive but unused.
    //! \param[in]  device_id  GPU device ID.
    void NoteAsyncMempoolFailure(uint32_t device_id);

#if ORT_API_VERSION >= 25
    //! \brief Get the CIG CUDA context for a device (for use in stream creation).
    //! \param[in]  device_id  GPU device ID.
    //! \return CIG CUcontext if one was created via InitGraphicsInterop, nullptr otherwise.
    //! \note CIG (added in ORT API v25). On older hosts no CIG context is ever
    //!       set (host won't call InitGraphicsInterop), so this returns nullptr
    //!       and the non-CIG stream path is used.
    CUcontext GetCigContext(int32_t device_id) const;
#endif

    // Called by child OrtEp instances to retrieve the cached kernel registry for that EP.
    OrtStatus* GetKernelRegistryForEp(/*out*/ const OrtKernelRegistry** kernel_registry);

private:
    // OrtEpFactory interface implementations
    static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

    static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

    static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

    static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                           const OrtHardwareDevice* const* devices, size_t num_devices,
                                                           OrtEpDevice** ep_devices, size_t max_ep_devices,
                                                           size_t* p_num_ep_devices) noexcept;

    static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
                                                const OrtKeyValuePairs* const* ep_metadata, size_t num_devices,
                                                const OrtSessionOptions* session_options, const OrtLogger* logger,
                                                OrtEp** ep) noexcept;

    static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

    static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                                                       const OrtKeyValuePairs* allocator_options,
                                                       OrtAllocator** allocator) noexcept;

    static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

    static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                          OrtDataTransferImpl** data_transfer) noexcept;

    static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;
    static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                 const OrtMemoryDevice* memory_device,
                                                                 const OrtKeyValuePairs* stream_options,
                                                                 OrtSyncStreamImpl** ort_stream) noexcept;
    static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
        OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices, size_t num_devices,
        const char* compatibility_info, OrtCompiledModelCompatibility* model_compatibility) noexcept;

    static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
        OrtEpFactory* this_ptr, const OrtHardwareDevice* hw, OrtDeviceEpIncompatibilityDetails* details) noexcept;

#if ORT_API_VERSION >= 25
    // CIG graphics-interop callbacks (added in ORT API v25). Populated
    // on the factory v-table when built against 1.25+ headers; hosts
    // older than 1.25 ignore them because ort_version_supported gates
    // host-side invocation. OrtGraphicsInteropConfig is itself 1.25+.
    static OrtStatus* ORT_API_CALL InitGraphicsInteropImpl(OrtEpFactory* this_ptr, const OrtEpDevice* ep_device,
                                                           const OrtGraphicsInteropConfig* config) noexcept;

    static OrtStatus* ORT_API_CALL DeinitGraphicsInteropImpl(OrtEpFactory* this_ptr,
                                                             const OrtEpDevice* ep_device) noexcept;
#endif

#if ORT_API_VERSION >= 26
    // External resource import callback (added in ORT API v26). Creates an
    // OrtExternalResourceImporterImpl bound to the EP device, enabling import of
    // D3D12 shared resources as CUDA memory and D3D12 fences as CUDA external
    // semaphores. Hosts older than 1.26 ignore it (gated by ort_version_supported).
    static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
        OrtEpFactory* this_ptr, const OrtEpDevice* ep_device, OrtExternalResourceImporterImpl** out_importer) noexcept;
#endif

    static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr, size_t* num_domains) noexcept;

    static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(OrtEpFactory* this_ptr, OrtCustomOpDomain** domains,
                                                          size_t num_domains) noexcept;

    OrtStatus* InitializeCustomOpDomains();

    const std::string ep_name_;                         //!< Execution provider name
    const std::string vendor_{"NVIDIA"};                //!< Execution provider vendor name (customize as needed)
    const std::string ep_version_{TRT_RTX_EP_VERSION};  //!< Execution provider version (from CMake)
    const uint32_t vendor_id_{0x10DE};                  //!< NVIDIA PCI vendor ID
    const OrtLogger& default_logger_;                   //!< Default logger instance

    //! Devices whose async mempool was latched off at run time. Mutated from the
    //! (concurrent) compute path under mempool_state_mutex_; entries only added.
    mutable std::mutex mempool_state_mutex_;
    std::unordered_set<uint32_t> mempool_runtime_disabled_;

    //! Guards lazy creation of device_allocators entries in GetOrCreateDeviceArena().
    std::mutex device_arena_mutex_;

    //! \brief Create a BFC arena backed by a TensorrtRtxAllocator (cudaMalloc/cudaFree) for the
    //! given device. Shared by CreateAllocatorImpl and GetOrCreateDeviceArena so the creation code
    //! lives in one place.
    //! \param[in]  memory_info  Device memory info the underlying allocator binds to.
    //! \param[in]  device_id    GPU device ID.
    //! \param[in]  options      Arena configuration (may be null for defaults).
    //! \param[out] out          Receives the created arena on success.
    //! \return nullptr on success, or an OrtStatus describing the failure.
    OrtStatus* CreateBfcArenaForDevice(const OrtMemoryInfo* memory_info, uint32_t device_id,
                                       const OrtKeyValuePairs* options, std::unique_ptr<ArenaAllocator>& out);

    // Cached kernel registry used by all OrtEp instances created by this factory.
    OrtKernelRegistry* kernel_registry_ = nullptr;
    std::vector<OrtCustomOpDomain*> custom_op_domains_;  //!< Custom op domains for TensorRT operations

#if ORT_API_VERSION >= 25
    // CIG contexts per device (keyed by device_id). The map stays empty on
    // hosts older than ORT 1.25 (those hosts never call InitGraphicsInterop).
    // Threading: InitGraphicsInteropImpl, DeinitGraphicsInteropImpl, and CreateSyncStreamForDeviceImpl
    // (via GetCigContext) may be called from different threads. All access is guarded by cig_contexts_mutex_.
    mutable std::mutex cig_contexts_mutex_;
    std::unordered_map<int32_t, CUcontext> cig_contexts_;
#endif
};

}  // namespace trt_rtx_ep
