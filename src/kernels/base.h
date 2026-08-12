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

#include "onnxruntime_cxx_api.h"

#include "../utils/ep_utils.h"

namespace trt_rtx_ep
{

// Base class for kernel implementations.
//
// Note: BaseKernelImpl has virtual functions so care should be taken when casting BaseKernelImpl to a OrtKernelImpl,
// which is a C API struct type. Specifically, a static_cast or implicit cast should be used. A reinterpret_cast
// will result in an invalid object due to the presence of the vtable.
class BaseKernelImpl : public OrtKernelImpl
{
public:
    BaseKernelImpl(const OrtKernelInfo* info, void* state);
    virtual ~BaseKernelImpl() = default;

    static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
    static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

private:
    // Derived classes implement DoCompute.
    // DoCompute is called by BaseKernelImpl::ComputeImpl, which also catches exceptions thrown by DoCompute
    // implementations and converts them into OrtStatus*.
    virtual OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) = 0;

protected:
    const OrtKernelInfo* info_;
    void* state_;  // Custom state passed from OrtEp
};

}  // namespace trt_rtx_ep