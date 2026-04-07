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
#include "nv_includes.h"

#include "onnxruntime_c_api.h"

#include <vector>

// Forward declarations for ORT types
struct OrtDataTransferImpl;
struct OrtMemoryDevice;
struct OrtValue;
struct OrtSyncStream;
struct OrtStatus;

namespace trt_rtx_ep
{

//!
//! \brief Data transfer implementation for copying tensors between memory locations.
//!
//! This handles data movement between different memory types (e.g., CPU <-> GPU, GPU <-> GPU).
//! Required for execution providers that manage device memory.
//!
struct TensorrtRtxDataTransfer : OrtDataTransferImpl, ApiPtrs
{
    // Constructor
    TensorrtRtxDataTransfer(const ApiPtrs& api_ptrs,
                            std::vector<const OrtMemoryDevice*>& device_mem_devices,
                            std::vector<const OrtMemoryDevice*>& shared_mem_devices,
                            uint32_t nvidia_vendor_id);

    // ========================================
    // Required OrtDataTransferImpl Interface
    // ========================================

    //! Check if copying between source and destination memory devices is supported
    static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                         const OrtMemoryDevice* src_memory_device,
                                         const OrtMemoryDevice* dst_memory_device) noexcept;

    //! Copy one or more tensors from source to destination
    //! Implementation can optionally use async copy if a stream is available
    static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                   const OrtValue** src_tensors_ptr,
                                                   OrtValue** dst_tensors_ptr,
                                                   OrtSyncStream** streams_ptr,
                                                   size_t num_tensors) noexcept;

    //! Release/cleanup the data transfer implementation
    static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

private:
    std::vector<const OrtMemoryDevice*>& device_mem_devices_;  //!< GPU/device memory devices
    std::vector<const OrtMemoryDevice*>& shared_mem_devices_;  //!< Pinned/shared memory devices
    const uint32_t nvidia_vendor_id_;                          //!< NVIDIA PCI vendor ID
};

}  // namespace trt_rtx_ep
