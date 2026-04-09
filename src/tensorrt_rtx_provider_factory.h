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

#include "ep_utils.h"
#include "ep_arena.h"
#include "cuda_mempool_arena.h"
#include "tensorrt_rtx_allocator.h"

#include "onnxruntime_c_api.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
//! \brief Plugin TensorRT RTX EP factory that can create an OrtEp and return information about the supported hardware devices.
//!
//! \details This factory manages device enumeration, allocators, and EP instance creation.
//!
struct TensorrtRtxExecutionProviderFactory : public OrtEpFactory, public ApiPtrs
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
    std::unordered_map<uint32_t, std::unique_ptr<ArenaAllocator>> device_allocators;  //!< Device ID -> BFC arena allocator
    // CUDA mempool allocators (used only for activation memory allocation.) 
    std::unordered_map<uint32_t, std::unique_ptr<CudaMempoolAllocator>> device_mempool_allocators;  //!< Device ID -> mempool allocator
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
        OrtEpFactory* this_ptr,
        const OrtHardwareDevice* const* devices,
        size_t num_devices,
        const char* compatibility_info,
        OrtCompiledModelCompatibility* model_compatibility) noexcept;

    static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
        OrtEpFactory* this_ptr,
        const OrtHardwareDevice* hw,
        OrtDeviceEpIncompatibilityDetails* details) noexcept;

    
    static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                              size_t* num_domains) noexcept;
                                                              
    static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                          OrtCustomOpDomain** domains,
                                                          size_t num_domains) noexcept;
    
    OrtStatus* InitializeCustomOpDomains();

    const std::string ep_name_;               //!< Execution provider name
    const std::string vendor_{"NVIDIA"};      //!< Execution provider vendor name (customize as needed)
    const std::string ep_version_{"0.1.0"};   //!< Execution provider version
    const uint32_t vendor_id_{0x10DE};        //!< NVIDIA PCI vendor ID
    const OrtLogger& default_logger_;         //!< Default logger instance

    // Cached kernel registry used by all OrtEp instances created by this factory.
    OrtKernelRegistry* kernel_registry_ = nullptr;
    std::vector<OrtCustomOpDomain*> custom_op_domains_;  //!< Custom op domains for TensorRT operations

};

}  // namespace trt_rtx_ep
