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

#include "onnxruntime_c_api.h"

// Forward declarations
struct OrtAllocator;
struct OrtMemoryInfo;

using DeviceId = int16_t;

namespace trt_rtx_ep
{

//!
//! \brief Device memory allocator implementing OrtAllocator interface.
//!
//! This allocates memory on the accelerator device (e.g., GPU).
//! Customize this for your specific device API (CUDA, ROCm, DirectML, etc.).
//!
struct TensorrtRtxAllocator : OrtAllocator
{
    TensorrtRtxAllocator(const OrtMemoryInfo* mem_info, DeviceId device_id);

    // OrtAllocator Interface implementations
    void* Alloc(size_t size);
    void Free(void* p) noexcept;
    const OrtMemoryInfo* Info() const noexcept;

    DeviceId GetDeviceId() const noexcept
    {
        return device_id_;
    }

private:
    // Delete copy/assignment
    TensorrtRtxAllocator(const TensorrtRtxAllocator&) = delete;
    TensorrtRtxAllocator& operator=(const TensorrtRtxAllocator&) = delete;

    // Device-specific helpers
    void CheckDevice(bool throw_when_fail) const;
    void SetDevice(bool throw_when_fail) const;

    const OrtMemoryInfo* mem_info_ = nullptr;
    DeviceId device_id_;
};

//!
//! \brief Pinned (host) memory allocator implementing OrtAllocator interface.
//!
//! This allocates page-locked host memory for faster transfers to/from device.
//! Optional: You can delete this if your device doesn't support pinned memory.
//!
struct TensorrtRtxPinnedAllocator : OrtAllocator
{
    explicit TensorrtRtxPinnedAllocator(const OrtMemoryInfo* mem_info);

    // OrtAllocator Interface implementations
    void* Alloc(size_t size);
    void Free(void* p) noexcept;
    const OrtMemoryInfo* Info() const noexcept;

private:
    // Delete copy/assignment
    TensorrtRtxPinnedAllocator(const TensorrtRtxPinnedAllocator&) = delete;
    TensorrtRtxPinnedAllocator& operator=(const TensorrtRtxPinnedAllocator&) = delete;

    const OrtMemoryInfo* mem_info_ = nullptr;
};

}  // namespace trt_rtx_ep
