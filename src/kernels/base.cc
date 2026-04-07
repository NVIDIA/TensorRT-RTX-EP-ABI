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

#include "base.h"

namespace trt_rtx_ep
{

BaseKernelImpl::BaseKernelImpl(const OrtKernelInfo* info, void* state)
    : OrtKernelImpl{}, info_{info}, state_{state}
{
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

// static
OrtStatus* ORT_API_CALL BaseKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept
{
  try
  {
    auto* kernel = static_cast<BaseKernelImpl*>(this_ptr);
    return kernel->DoCompute(kernel_ctx);
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

// static
void ORT_API_CALL BaseKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept
{
  delete static_cast<BaseKernelImpl*>(this_ptr);
}

}  // namespace trt_rtx_ep