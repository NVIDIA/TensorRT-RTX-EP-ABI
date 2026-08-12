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

#include "base.h"
#include "utils.h"

namespace trt_rtx_ep
{

//!
//! \brief Kernel implementation for copying data from host (CPU) to device (GPU).
//!
class MemcpyFromHost : public BaseKernelImpl
{
private:
    struct PrivateTag
    {
    };

public:
    static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<MemcpyFromHost>& kernel);
    MemcpyFromHost(const OrtKernelInfo* info, void* state, PrivateTag);

private:
    OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) override;
};

//!
//! \brief Kernel implementation for copying data from device (GPU) to host (CPU).
//!
class MemcpyToHost : public BaseKernelImpl
{
private:
    struct PrivateTag
    {
    };

public:
    static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<MemcpyToHost>& kernel);
    MemcpyToHost(const OrtKernelInfo* info, void* state, PrivateTag);

private:
    OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) override;
};

}  // namespace trt_rtx_ep