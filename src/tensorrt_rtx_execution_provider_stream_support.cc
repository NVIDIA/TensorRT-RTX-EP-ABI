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

#include "tensorrt_rtx_execution_provider_stream_support.h"

#include "tensorrt_rtx_execution_provider.h"
#include "tensorrt_rtx_provider_factory.h"

#include "utils/cuda/cuda_call.h"
#include "utils/cuda/cuda_common.h"
#include "utils/cuda/cuda_context.h"
#include "utils/ort_api_init.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>

namespace trt_rtx_ep
{

// Buffer size for error messages
constexpr size_t kErrorMsgBufferSize = 128;

//
// TensorrtRtxSyncNotificationImpl implementation
//

/*static*/
OrtStatus* TensorrtRtxSyncNotificationImpl::Create(cudaStream_t stream, const ApiPtrs& api_ptrs,
                                                   std::unique_ptr<TensorrtRtxSyncNotificationImpl>& notification)
{
    notification.reset(
        new TensorrtRtxSyncNotificationImpl(stream, api_ptrs));  // can't use make_unique with private ctor
    RETURN_IF_ERROR(GetCudaStreamContext(api_ptrs.ort_api, stream, &notification->stream_context_));
    OrtStatus* context_status = nullptr;
    ScopedCudaContext stream_context(api_ptrs.ort_api, notification->stream_context_, &context_status);
    RETURN_IF_ERROR(context_status);
    RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&notification->event_, cudaEventDisableTiming)));

    return nullptr;
}

TensorrtRtxSyncNotificationImpl::TensorrtRtxSyncNotificationImpl(cudaStream_t stream, const ApiPtrs& api_ptrs)
    : OrtSyncNotificationImpl{}
    , ApiPtrs{api_ptrs}
    , stream_{stream}
    , event_{nullptr}
{
    ort_version_supported = NegotiatedOrtApiVersion();
    Activate = ActivateImpl;
    WaitOnDevice = WaitOnDeviceImpl;
    WaitOnHost = WaitOnHostImpl;
    Release = ReleaseImpl;
}

TensorrtRtxSyncNotificationImpl::~TensorrtRtxSyncNotificationImpl()
{
    if (event_ != nullptr)
    {
        ScopedCudaContextNoThrow stream_context(stream_context_);
        cudaEventDestroy(event_);
    }
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxSyncNotificationImpl::ActivateImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] SyncNotification ActivateImpl: this_ptr is null");
    }

    auto& impl = *static_cast<TensorrtRtxSyncNotificationImpl*>(this_ptr);
    OrtStatus* context_status = nullptr;
    ScopedCudaContext stream_context(impl.ort_api, impl.stream_context_, &context_status);
    RETURN_IF_ERROR(context_status);
    RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(impl.event_, impl.stream_)));

    return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxSyncNotificationImpl::WaitOnDeviceImpl(_In_ OrtSyncNotificationImpl* this_ptr,
                                                                          _In_ OrtSyncStream* consumer_stream) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] SyncNotification WaitOnDeviceImpl: this_ptr is null");
    }

    auto& impl = *static_cast<TensorrtRtxSyncNotificationImpl*>(this_ptr);

    // Security check: validate consumer_stream is not null
    if (consumer_stream == nullptr)
    {
        return impl.ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] SyncNotification WaitOnDeviceImpl: consumer_stream is null");
    }

    // setup the consumer stream to wait on our event.
    void* consumer_handle = impl.ort_api.SyncStream_GetHandle(consumer_stream);
    CUcontext consumer_context = nullptr;
    RETURN_IF_ERROR(GetCudaStreamContext(impl.ort_api, static_cast<cudaStream_t>(consumer_handle), &consumer_context));
    OrtStatus* context_status = nullptr;
    ScopedCudaContext stream_context(impl.ort_api, consumer_context, &context_status);
    RETURN_IF_ERROR(context_status);
    RETURN_IF_ERROR(CUDA_CALL(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer_handle), impl.event_)));

    return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxSyncNotificationImpl::WaitOnHostImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] SyncNotification WaitOnHostImpl: this_ptr is null");
    }

    auto& impl = *static_cast<TensorrtRtxSyncNotificationImpl*>(this_ptr);
    RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(impl.event_)));

    return nullptr;
}

/*static*/
void ORT_API_CALL TensorrtRtxSyncNotificationImpl::ReleaseImpl(_In_ OrtSyncNotificationImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null before deleting
    if (this_ptr == nullptr)
    {
        return;
    }
    delete static_cast<TensorrtRtxSyncNotificationImpl*>(this_ptr);
}

//
// TensorrtRtxSyncStreamImpl implementation
//

/*static*/
OrtStatus* TensorrtRtxSyncStreamImpl::Create(TensorrtRtxExecutionProviderFactory& factory, const OrtEp* ep,
                                             uint32_t device_id, const OrtKeyValuePairs* stream_options,
                                             std::unique_ptr<TensorrtRtxSyncStreamImpl>& stream_impl)
{
    // Security check: validate device_id is within valid range
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess || num_devices <= 0)
    {
        return factory.ort_api.CreateStatus(ORT_EP_FAIL,
                                            "[NvTensorRTRTX EP] SyncStream Create: Failed to get CUDA device count");
    }
    if (static_cast<int>(device_id) >= num_devices)
    {
        char error_msg[kErrorMsgBufferSize];
        snprintf(error_msg, kErrorMsgBufferSize,
                 "[NvTensorRTRTX EP] SyncStream Create: Invalid device_id %u, only %d devices available", device_id,
                 num_devices);
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error_msg);
    }

    stream_impl.reset(new TensorrtRtxSyncStreamImpl(factory, ep, device_id, stream_options));

    // Check if we have an EP and if it has an external stream
    const TensorrtRtxExecutionProvider* trt_ep = ep ? static_cast<const TensorrtRtxExecutionProvider*>(ep) : nullptr;
    if (trt_ep)
    {
        stream_impl->stream_ = trt_ep->stream_;
        stream_impl->stream_context_ = trt_ep->compute_stream_context_;
        stream_impl->own_stream_ = false;
    }
    else
    {
        cudaStream_t stream = nullptr;

#if ORT_API_VERSION >= 25
        // CIG graphics-interop was added in ORT API v25. The factory's
        // GetCigContext only exists when built against 1.25+ headers.
        // At runtime, on hosts older than 1.25 the CIG map stays empty
        // (host never calls InitGraphicsInterop), so this returns nullptr
        // and we fall through to the non-CIG path.
        CUcontext cig_context = factory.GetCigContext(static_cast<int32_t>(device_id));
        if (cig_context != nullptr)
        {
            OrtStatus* context_status = nullptr;
            ScopedCudaContext stream_context(factory.ort_api, cig_context, &context_status);
            RETURN_IF_ERROR(context_status);

            cudaError_t cuda_err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (cuda_err == cudaSuccess)
            {
                stream_impl->stream_context_ = cig_context;
            }

            RETURN_IF_ERROR(CUDA_CALL(cuda_err));
        }
        else
#endif
        {
            // Preserve an already-current CUDA context (e.g. an app-created CIG context that was made
            // current without going through InitGraphicsInterop). Calling cudaSetDevice unconditionally
            // would switch the thread to the device's primary context — clobbering the caller's context
            // and creating the stream on the wrong one. Only select the device when no
            // context is current; otherwise create the stream on the context the caller already set.
            CUcontext current_context = nullptr;
            if (cuCtxGetCurrent(&current_context) != CUDA_SUCCESS || current_context == nullptr)
            {
                RETURN_IF_ERROR(CUDA_CALL(cudaSetDevice(static_cast<int>(device_id))));
            }
            RETURN_IF_ERROR(CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)));
            RETURN_IF_ERROR(GetCudaStreamContext(factory.ort_api, stream, &stream_impl->stream_context_));
        }

        stream_impl->stream_ = stream;
        stream_impl->own_stream_ = true;
    }

    return nullptr;
}

TensorrtRtxSyncStreamImpl::TensorrtRtxSyncStreamImpl(TensorrtRtxExecutionProviderFactory& factory, const OrtEp* ep,
                                                     uint32_t device_id, const OrtKeyValuePairs* /*stream_options*/)
    : OrtSyncStreamImpl{}
    , ApiPtrs(factory)
    , factory_{&factory}
    , ep_{ep}
    , device_id_{device_id}
{
    ort_version_supported = NegotiatedOrtApiVersion();
    GetHandle = GetHandleImpl;
    CreateNotification = CreateNotificationImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = OnSessionRunEndImpl;
    Release = ReleaseImpl;
}

TensorrtRtxSyncStreamImpl::~TensorrtRtxSyncStreamImpl()
{
    if (own_stream_ && stream_ != nullptr)
    {
        ScopedCudaContextNoThrow stream_context(stream_context_);
        cudaStreamDestroy(stream_);
    }
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxSyncStreamImpl::CreateNotificationImpl(
    _In_ OrtSyncStreamImpl* this_ptr, _Outptr_ OrtSyncNotificationImpl** notification_impl) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] SyncStream CreateNotificationImpl: this_ptr is null");
    }

    auto& impl = *static_cast<TensorrtRtxSyncStreamImpl*>(this_ptr);

    // Security check: validate output parameter is not null
    if (notification_impl == nullptr)
    {
        return impl.ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "[NvTensorRTRTX EP] SyncStream CreateNotificationImpl: notification_impl output is null");
    }
    *notification_impl = nullptr;

    std::unique_ptr<TensorrtRtxSyncNotificationImpl> notification;
    RETURN_IF_ERROR(TensorrtRtxSyncNotificationImpl::Create(impl.stream_, impl, notification));
    *notification_impl = notification.release();

    return nullptr;
}

/*static*/
void* ORT_API_CALL TensorrtRtxSyncStreamImpl::GetHandleImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return nullptr;
    }

    auto& impl = *static_cast<TensorrtRtxSyncStreamImpl*>(this_ptr);
    return static_cast<void*>(impl.stream_);
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxSyncStreamImpl::FlushImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] SyncStream FlushImpl: this_ptr is null");
    }

    auto& impl = *static_cast<TensorrtRtxSyncStreamImpl*>(this_ptr);
    OrtStatus* context_status = nullptr;
    ScopedCudaContext stream_context(impl.ort_api, impl.stream_context_, &context_status);
    RETURN_IF_ERROR(context_status);
    RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(impl.stream_)));

    return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxSyncStreamImpl::OnSessionRunEndImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] SyncStream OnSessionRunEndImpl: this_ptr is null");
    }

    // Perform any cleanup or synchronization at end of session run
    // Currently no-op as we don't have special cleanup needs
    (void)this_ptr;
    return nullptr;
}

/*static*/
void ORT_API_CALL TensorrtRtxSyncStreamImpl::ReleaseImpl(_In_ OrtSyncStreamImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null before deleting
    if (this_ptr == nullptr)
    {
        return;
    }
    delete static_cast<TensorrtRtxSyncStreamImpl*>(this_ptr);
}

}  // namespace trt_rtx_ep
