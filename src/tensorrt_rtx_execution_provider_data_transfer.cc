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

#include "tensorrt_rtx_execution_provider_data_transfer.h"

#include "utils/cuda/cuda_call.h"
#include "utils/cuda/cuda_common.h"
#include "utils/ort_api_init.h"

#include <cuda_runtime_api.h>

#include <cstring>

namespace trt_rtx_ep
{

// Constructor
TensorrtRtxDataTransfer::TensorrtRtxDataTransfer(const ApiPtrs& api_ptrs,
                                                 std::vector<const OrtMemoryDevice*>& device_mem_devices,
                                                 std::vector<const OrtMemoryDevice*>& shared_mem_devices,
                                                 uint32_t nvidia_vendor_id)
    : OrtDataTransferImpl{},
      ApiPtrs{api_ptrs},
      device_mem_devices_{device_mem_devices},
      shared_mem_devices_{shared_mem_devices},
      nvidia_vendor_id_{nvidia_vendor_id}
{
    // Initialize OrtDataTransferImpl interface function pointers
    ort_version_supported = NegotiatedOrtApiVersion();
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
}

/*static*/
bool ORT_API_CALL TensorrtRtxDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                                       const OrtMemoryDevice* src_memory_device,
                                                       const OrtMemoryDevice* dst_memory_device) noexcept
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr || src_memory_device == nullptr || dst_memory_device == nullptr)
    {
        return false;
    }

    const auto& impl = *static_cast<const TensorrtRtxDataTransfer*>(this_ptr);

    // Logic copied from GPUDataTransfer::CanCopy in reference.cc
    OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);
    auto src_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
    auto dst_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);

    // If src or dst is GPU type, verify it has NVIDIA vendor ID
    if ((src_type == OrtMemoryInfoDeviceType_GPU && src_vendor_id != impl.nvidia_vendor_id_) ||
        (dst_type == OrtMemoryInfoDeviceType_GPU && dst_vendor_id != impl.nvidia_vendor_id_))
    {
        return false;
    }

    // Data transfer can happen between:
    // - GPU to GPU
    // - CPU to GPU
    // - GPU to CPU
    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
           (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
           (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                                 const OrtValue** src_tensors,
                                                                 OrtValue** dst_tensors,
                                                                 OrtSyncStream** streams,
                                                                 size_t num_tensors) noexcept
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CopyTensorsImpl: this_ptr is null");
    }

    auto& impl = *static_cast<TensorrtRtxDataTransfer*>(this_ptr);

    if (num_tensors > 0)
    {
        if (src_tensors == nullptr)
        {
            return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CopyTensorsImpl: src_tensors is null");
        }
        if (dst_tensors == nullptr)
        {
            return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CopyTensorsImpl: dst_tensors is null");
        }
        // Note: streams can be null (synchronous copy)
    }
    bool need_stream_sync = false;

    for (size_t idx = 0; idx < num_tensors; ++idx)
    {
        const OrtValue* src_tensor = src_tensors[idx];
        OrtValue* dst_tensor = dst_tensors[idx];
        OrtSyncStream* stream = streams ? streams[idx] : nullptr;

        // Security check: validate individual tensor pointers are not null
        if (src_tensor == nullptr)
        {
            return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CopyTensorsImpl: src_tensor at index is null");
        }
        if (dst_tensor == nullptr)
        {
            return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CopyTensorsImpl: dst_tensor at index is null");
        }

        const OrtMemoryDevice* src_device = impl.ep_api.Value_GetMemoryDevice(src_tensor);
        const OrtMemoryDevice* dst_device = impl.ep_api.Value_GetMemoryDevice(dst_tensor);

        // Security check: validate memory device info was retrieved successfully
        if (src_device == nullptr || dst_device == nullptr)
        {
            return impl.ort_api.CreateStatus(ORT_EP_FAIL, "[NvTensorRTRTX EP] CopyTensorsImpl: Failed to get memory device info from tensor");
        }

        size_t bytes;
        RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensor, &bytes));

        const void* src_data = nullptr;
        void* dst_data = nullptr;
        RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensor, &src_data));
        RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensor, &dst_data));

        OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_device);
        OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_device);
        OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_device);
        OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_device);

        const bool src_is_gpu_default = src_type == OrtMemoryInfoDeviceType_GPU &&
                                        src_mem_type == OrtDeviceMemoryType_DEFAULT;
        const bool dst_is_gpu_default = dst_type == OrtMemoryInfoDeviceType_GPU &&
                                        dst_mem_type == OrtDeviceMemoryType_DEFAULT;

        cudaStream_t cuda_stream = nullptr;
        if (stream)
        {
            cuda_stream = static_cast<cudaStream_t>(impl.ort_api.SyncStream_GetHandle(stream));
        }

        if (dst_is_gpu_default)
        {
            if (src_is_gpu_default)
            {
                // Copy only if the two addresses are different.
                if (dst_data != src_data)
                {
                    if (cuda_stream)
                    {
                        RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, cuda_stream)));
                    }
                    else
                    {
                        RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice)));

                        // For device memory to device memory copy, no host-side synchronization is performed by cudaMemcpy.
                        // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
                        need_stream_sync = true;
                    }
                }
            }
            else
            {
                // copy from pinned or non-pinned CPU memory to GPU
                if (cuda_stream)
                {
                    RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, cuda_stream)));
                }
                else
                {
                    RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice)));

                    if (src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE)
                    {
                        // For cudaMemcpy from pageable host memory to device memory, DMA to final destination may not
                        // have completed.
                        // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
                        need_stream_sync = true;
                    }
                }
            }
        }
        else if (src_is_gpu_default)
        {
            // copying from GPU to CPU memory, this is blocking

            if (cuda_stream)
            {
                RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, cuda_stream)));
            }
            else
            {
                RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost)));
            }
        }
        else
        {
            // copying between CPU accessible memory

            if (dst_data != src_data)
            {
                if (cuda_stream)
                {
                    if (src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE)
                    {
                        // sync the stream first to make sure the data arrived
                        RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(cuda_stream)));
                    }
                }

                memcpy(dst_data, src_data, bytes);
            }
        }
    }

    if (need_stream_sync)
    {
        RETURN_IF_ERROR(CUDA_CALL(cudaStreamSynchronize(nullptr)));
    }

    return nullptr;
}

/*static*/
void ORT_API_CALL TensorrtRtxDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept
{
    // Security check: validate this_ptr (though this is a no-op, good practice)
    if (this_ptr == nullptr)
    {
        return;
    }
    // No-op as we have a single shared instance in the factory which is returned from CreateDataTransferImpl,
    // and is owned by and freed by the factory.
    (void)this_ptr;
}

}  // namespace trt_rtx_ep
