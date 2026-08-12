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

#include "memcpy.h"

#include <cuda_runtime.h>

namespace trt_rtx_ep
{

// MemcpyFromHost implementation

// static
OrtStatus* MemcpyFromHost::Create(const OrtKernelInfo* info, void* state,
                                  /*out*/ std::unique_ptr<MemcpyFromHost>& kernel)
{
    try
    {
        kernel = std::make_unique<MemcpyFromHost>(info, state, PrivateTag{});
        return nullptr;
    }
    catch (const Ort::Exception& ex)
    {
        Ort::Status status(ex);
        return status.release();
    }
    catch (const std::exception& ex)
    {
        Ort::Status status(ex.what(), ORT_EP_FAIL);
        return status.release();
    }
}

MemcpyFromHost::MemcpyFromHost(const OrtKernelInfo* info, void* state, PrivateTag)
    : BaseKernelImpl(info, state)
{
}

OrtStatus* MemcpyFromHost::DoCompute(OrtKernelContext* kernel_ctx)
{
    try
    {
        const OrtApi& ort_api = Ort::GetApi();
        const OrtValue* input_tensor = nullptr;
        RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_ctx, 0, &input_tensor));

        // Get tensor shape and type
        OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input_tensor, &tensor_info));

        size_t element_count = 0;
        RETURN_IF_ERROR(ort_api.GetTensorShapeElementCount(tensor_info, &element_count));

        ONNXTensorElementDataType element_type;
        RETURN_IF_ERROR(ort_api.GetTensorElementType(tensor_info, &element_type));

        size_t num_dims = 0;
        RETURN_IF_ERROR(ort_api.GetDimensionsCount(tensor_info, &num_dims));

        std::vector<int64_t> dims(num_dims);
        RETURN_IF_ERROR(ort_api.GetDimensions(tensor_info, dims.data(), num_dims));
        ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

        // Get output tensor
        OrtValue* output_tensor = nullptr;
        RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_ctx, 0, dims.data(), num_dims, &output_tensor));

        // Get data pointers
        const void* input_data = nullptr;
        void* output_data = nullptr;
        RETURN_IF_ERROR(ort_api.GetTensorData(input_tensor, &input_data));
        RETURN_IF_ERROR(ort_api.GetTensorMutableData(output_tensor, &output_data));

        // Calculate size in bytes
        size_t bytes = 0;
        RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(input_tensor, &bytes));

        // Get CUDA stream from kernel context
        void* cuda_stream = nullptr;
        RETURN_IF_ERROR(ort_api.KernelContext_GetGPUComputeStream(kernel_ctx, &cuda_stream));
        cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

        // Copy from host (CPU) to device (GPU) asynchronously
        cudaError_t cuda_err = cudaMemcpyAsync(output_data, input_data, bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_err != cudaSuccess)
        {
            return ort_api.CreateStatus(ORT_EP_FAIL, cudaGetErrorString(cuda_err));
        }

        return nullptr;
    }
    catch (const Ort::Exception& ex)
    {
        Ort::Status status(ex);
        return status.release();
    }
    catch (const std::exception& ex)
    {
        Ort::Status status(ex.what(), ORT_EP_FAIL);
        return status.release();
    }
}

// MemcpyToHost implementation

// static
OrtStatus* MemcpyToHost::Create(const OrtKernelInfo* info, void* state,
                                /*out*/ std::unique_ptr<MemcpyToHost>& kernel)
{
    try
    {
        kernel = std::make_unique<MemcpyToHost>(info, state, PrivateTag{});
        return nullptr;
    }
    catch (const Ort::Exception& ex)
    {
        Ort::Status status(ex);
        return status.release();
    }
    catch (const std::exception& ex)
    {
        Ort::Status status(ex.what(), ORT_EP_FAIL);
        return status.release();
    }
}

MemcpyToHost::MemcpyToHost(const OrtKernelInfo* info, void* state, PrivateTag)
    : BaseKernelImpl(info, state)
{
}

OrtStatus* MemcpyToHost::DoCompute(OrtKernelContext* kernel_ctx)
{
    try
    {
        const OrtApi& ort_api = Ort::GetApi();
        const OrtValue* input_tensor = nullptr;
        RETURN_IF_ERROR(ort_api.KernelContext_GetInput(kernel_ctx, 0, &input_tensor));

        // Get tensor shape and type
        OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(input_tensor, &tensor_info));

        size_t num_dims = 0;
        RETURN_IF_ERROR(ort_api.GetDimensionsCount(tensor_info, &num_dims));

        std::vector<int64_t> dims(num_dims);
        RETURN_IF_ERROR(ort_api.GetDimensions(tensor_info, dims.data(), num_dims));
        ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

        // Get output tensor
        OrtValue* output_tensor = nullptr;
        RETURN_IF_ERROR(ort_api.KernelContext_GetOutput(kernel_ctx, 0, dims.data(), num_dims, &output_tensor));

        // Get data pointers
        const void* input_data = nullptr;
        void* output_data = nullptr;
        RETURN_IF_ERROR(ort_api.GetTensorData(input_tensor, &input_data));
        RETURN_IF_ERROR(ort_api.GetTensorMutableData(output_tensor, &output_data));

        // Calculate size in bytes
        size_t bytes = 0;
        RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(input_tensor, &bytes));

        // Get CUDA stream from kernel context
        void* cuda_stream = nullptr;
        RETURN_IF_ERROR(ort_api.KernelContext_GetGPUComputeStream(kernel_ctx, &cuda_stream));
        cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

        // Copy from device (GPU) to host (CPU) asynchronously
        cudaError_t cuda_err = cudaMemcpyAsync(output_data, input_data, bytes, cudaMemcpyDeviceToHost, stream);
        if (cuda_err != cudaSuccess)
        {
            return ort_api.CreateStatus(ORT_EP_FAIL, cudaGetErrorString(cuda_err));
        }

        return nullptr;
    }
    catch (const Ort::Exception& ex)
    {
        Ort::Status status(ex);
        return status.release();
    }
    catch (const std::exception& ex)
    {
        Ort::Status status(ex.what(), ORT_EP_FAIL);
        return status.release();
    }
}

}  // namespace trt_rtx_ep