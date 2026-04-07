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

#include "tensorrt_rtx_provider_factory.h"
#include "utils/ep_utils.h"
#include "nv_includes.h"

#include "onnxruntime_c_api.h"

#include <cuda_runtime_api.h>

#include <memory>

// Forward declarations for ORT types
struct OrtSyncStreamImpl;
struct OrtSyncNotificationImpl;
struct OrtSyncStream;
struct OrtEp;
struct OrtKeyValuePairs;
struct OrtStatus;

namespace trt_rtx_ep
{

// Forward declaration
struct TensorrtRtxSyncStreamImpl;

//!
//! \brief Notification implementation for synchronization.
//!
//! This allows synchronization between streams and between device/host.
//! Only needed if your EP supports stream-aware execution.
//!
struct TensorrtRtxSyncNotificationImpl : OrtSyncNotificationImpl, ApiPtrs
{
    // Factory method to create notification
    static OrtStatus* Create(cudaStream_t stream, const ApiPtrs& api_ptrs,
                             std::unique_ptr<TensorrtRtxSyncNotificationImpl>& notification);

    ~TensorrtRtxSyncNotificationImpl();

private:
    // Constructor
    TensorrtRtxSyncNotificationImpl(cudaStream_t stream, const ApiPtrs& api_ptrs);

    // ========================================
    // Required OrtSyncNotificationImpl Interface
    // ========================================

    static OrtStatus* ORT_API_CALL ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                    _In_ OrtSyncStream* stream) noexcept;

    static OrtStatus* ORT_API_CALL WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

    static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept;

    cudaStream_t stream_;
    cudaEvent_t event_;
};

//!
//! \brief Stream implementation for synchronous execution.
//!
//! This is used for stream-based execution and synchronization between operations.
//! Only needed if your EP supports stream-aware execution.
//!
struct TensorrtRtxSyncStreamImpl : public OrtSyncStreamImpl, public ApiPtrs
{
    // Factory method - use this to create instances (handles CUDA errors gracefully)
    static OrtStatus* Create(TensorrtRtxExecutionProviderFactory& factory,
                             const OrtEp* ep,
                             uint32_t device_id,
                             const OrtKeyValuePairs* stream_options,
                             std::unique_ptr<TensorrtRtxSyncStreamImpl>& stream_impl);

    ~TensorrtRtxSyncStreamImpl();

private:
    // Private constructor - use Create() factory method instead
    TensorrtRtxSyncStreamImpl(TensorrtRtxExecutionProviderFactory& factory,
                              const OrtEp* ep,
                              uint32_t device_id,
                              const OrtKeyValuePairs* stream_options);

    // ========================================
    // Required OrtSyncStreamImpl Interface
    // ========================================

    static OrtStatus* ORT_API_CALL CreateNotificationImpl(_In_ OrtSyncStreamImpl* this_ptr,
                                                          _Outptr_ OrtSyncNotificationImpl** sync_notification) noexcept;

    static void* ORT_API_CALL GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

    static void ORT_API_CALL ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept;

    TensorrtRtxExecutionProviderFactory* factory_{nullptr};
    const OrtEp* ep_;
    cudaStream_t stream_{nullptr};
    bool own_stream_{true};
    uint32_t device_id_;
};

}  // namespace trt_rtx_ep
