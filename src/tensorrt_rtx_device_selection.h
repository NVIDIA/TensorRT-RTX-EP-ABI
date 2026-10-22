// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tensorrt_rtx_provider_options.h"

namespace trt_rtx_ep
{

inline constexpr uint32_t kNvidiaPciVendorId = 0x10DE;
// ORT's Linux PCI discovery reports WSL2 / Hyper-V display adapters with this vendor.
inline constexpr uint32_t kMicrosoftVirtualPciVendorId = 0x1414;

// Selection must use the same architecture gate the EP applies at device init, or it would
// publish a device the EP then refuses to run on. Keep this the single source of truth.
inline bool IsSupportedComputeCapability(int major, int minor)
{
    return onnxruntime::tensorrt_rtx::IsComputeCapabilitySupported(major, minor);
}

// Windows matches NVIDIA GPUs by LUID, so a Microsoft virtual vendor is not a candidate there.
// Linux WSL2 reports the NVIDIA GPU as vendor 0x1414 with a PCI id CUDA will not resolve.
inline bool IsEligibleGpuVendor(uint32_t vendor_id)
{
#if defined(_WIN32)
    return vendor_id == kNvidiaPciVendorId;
#else
    return vendor_id == kNvidiaPciVendorId || vendor_id == kMicrosoftVirtualPciVendorId;
#endif
}

enum class HardwareMatchKind
{
    kRejected,
    kMappedToCudaDevice,
    kVirtualAdapterUnmapped,
};

// One ORT hardware device after PCI / LUID probing. Compute capability is recorded for a mapped
// device even when it is below the minimum, so callers can explain the rejection.
struct HardwareDeviceMatch
{
    int ordinal = -1;
    HardwareMatchKind kind = HardwareMatchKind::kRejected;
    int cuda_index = -1;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
};

// A CUDA device visible to this process. CUDA_VISIBLE_DEVICES filtering is already applied:
// `index` is the runtime index cudaGetDeviceProperties accepts, not a physical PCI ordinal.
struct VisibleCudaDevice
{
    int index = -1;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
};

struct SelectedEpDevice
{
    int hardware_ordinal = -1;
    int cuda_index = -1;
};

// Assigns each supported CUDA device to at most one ORT hardware device.
//
// Mapped devices claim their resolved CUDA index first. Unmapped Microsoft virtual adapters then
// take the lowest-index remaining CUDA device whose compute capability the EP supports (the same
// IsComputeCapabilitySupported gate device init applies), in hardware-list order. Extra virtual
// adapters are left unselected. WSL2 can expose several identical 0x1414 adapters (the NVIDIA GPU
// and an iGPU) for a single CUDA device, and those adapters do not share a PCI id with the CUDA
// driver, so they cannot be matched individually.
//
// Results are returned in hardware-list order. `cuda_index` is the runtime CUDA index.
inline std::vector<SelectedEpDevice> SelectSupportedEpDevices(const std::vector<HardwareDeviceMatch>& hardware,
                                                              const std::vector<VisibleCudaDevice>& cuda_devices)
{
    std::vector<char> claimed(cuda_devices.size(), 0);

    auto position_of = [&cuda_devices](int cuda_index) -> int
    {
        for (size_t i = 0; i < cuda_devices.size(); ++i)
        {
            if (cuda_devices[i].index == cuda_index)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    auto lowest_unclaimed_supported = [&cuda_devices, &claimed]() -> int
    {
        int chosen = -1;
        int chosen_index = 0;
        for (size_t i = 0; i < cuda_devices.size(); ++i)
        {
            if (claimed[i] || !IsSupportedComputeCapability(cuda_devices[i].compute_capability_major,
                                                           cuda_devices[i].compute_capability_minor))
            {
                continue;
            }
            if (chosen < 0 || cuda_devices[i].index < chosen_index)
            {
                chosen = static_cast<int>(i);
                chosen_index = cuda_devices[i].index;
            }
        }
        return chosen;
    };

    std::vector<SelectedEpDevice> selected;
    std::vector<int> virtual_adapters;
    virtual_adapters.reserve(hardware.size());

    for (const HardwareDeviceMatch& device : hardware)
    {
        if (device.kind == HardwareMatchKind::kMappedToCudaDevice)
        {
            if (!IsSupportedComputeCapability(device.compute_capability_major, device.compute_capability_minor))
            {
                continue;
            }
            const int position = position_of(device.cuda_index);
            if (position < 0 || claimed[static_cast<size_t>(position)])
            {
                continue;
            }
            if (!IsSupportedComputeCapability(cuda_devices[static_cast<size_t>(position)].compute_capability_major,
                                              cuda_devices[static_cast<size_t>(position)].compute_capability_minor))
            {
                continue;
            }
            claimed[static_cast<size_t>(position)] = 1;
            selected.push_back(SelectedEpDevice{device.ordinal, device.cuda_index});
        }
        else if (device.kind == HardwareMatchKind::kVirtualAdapterUnmapped)
        {
            virtual_adapters.push_back(device.ordinal);
        }
    }

    for (int ordinal : virtual_adapters)
    {
        const int position = lowest_unclaimed_supported();
        if (position < 0)
        {
            break;
        }
        claimed[static_cast<size_t>(position)] = 1;
        selected.push_back(SelectedEpDevice{ordinal, cuda_devices[static_cast<size_t>(position)].index});
    }

    std::sort(selected.begin(), selected.end(),
              [](const SelectedEpDevice& lhs, const SelectedEpDevice& rhs)
              {
                  return lhs.hardware_ordinal < rhs.hardware_ordinal;
              });
    return selected;
}

}  // namespace trt_rtx_ep
