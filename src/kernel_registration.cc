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

#include "kernel_registration.h"

#include <vector>

#include "kernels/memcpy.h"
#include "kernels/utils.h"

using namespace trt_rtx_ep;

// Forward declare the kernel classes
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyToHost);

// Define kernel builders
// Memory type specifications tell ORT where to allocate input/output buffers:
// - MemcpyFromHost: Input on CPU (OrtMemTypeCPUInput), output on GPU (default)
// - MemcpyToHost: Input on GPU (default), output on CPU (OrtMemTypeCPUOutput)
ONNX_OPERATOR_KERNEL_EX(MemcpyFromHost, kOnnxDomain, 1, Ort::KernelDefBuilder().SetInputMemType(0, OrtMemTypeCPUInput),
                        MemcpyFromHost);

ONNX_OPERATOR_KERNEL_EX(MemcpyToHost, kOnnxDomain, 1, Ort::KernelDefBuilder().SetOutputMemType(0, OrtMemTypeCPUOutput),
                        MemcpyToHost);

// Table of BuildKernelCreateInfo functions for each operator
static const BuildKernelCreateInfoFn build_kernel_create_info_funcs[] = {
    // MemcpyFromHost version 1
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyFromHost)>,

    // MemcpyToHost version 1
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyToHost)>,
};

size_t GetNumKernels()
{
    return std::size(build_kernel_create_info_funcs);
}

static OrtStatus* RegisterKernels(Ort::KernelRegistry& kernel_registry, const char* ep_name, void* create_kernel_state)
{
    for (auto& build_func : build_kernel_create_info_funcs)
    {
        KernelCreateInfo kernel_create_info = {};
        RETURN_IF_ERROR(build_func(ep_name, create_kernel_state, &kernel_create_info));

        if (kernel_create_info.kernel_def != nullptr)
        {
            RETURN_IF_ERROR(kernel_registry.AddKernel(kernel_create_info.kernel_def,
                                                      kernel_create_info.kernel_create_func,
                                                      kernel_create_info.kernel_create_func_state));
        }
    }

    return nullptr;
}

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state, OrtKernelRegistry** out_kernel_registry)
{
    *out_kernel_registry = nullptr;

    if (GetNumKernels() == 0)
    {
        return nullptr;
    }

    try
    {
        Ort::KernelRegistry kernel_registry;
        Ort::Status status{RegisterKernels(kernel_registry, ep_name, create_kernel_state)};

        *out_kernel_registry = status.IsOK() ? kernel_registry.release() : nullptr;
        return status.release();
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
