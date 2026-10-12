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

// Pairing rules for ORT hardware devices and visible CUDA devices.
// No GPU or live CUDA runtime is required: callers pass the probe results.

#include "tensorrt_rtx_device_selection.h"

#include <vector>

#include <gtest/gtest.h>

namespace
{

using trt_rtx_ep::HardwareDeviceMatch;
using trt_rtx_ep::HardwareMatchKind;
using trt_rtx_ep::SelectedEpDevice;
using trt_rtx_ep::SelectSupportedEpDevices;
using trt_rtx_ep::VisibleCudaDevice;

HardwareDeviceMatch Mapped(int ordinal, int cuda_index, int major, int minor = 0)
{
    return HardwareDeviceMatch{ordinal, HardwareMatchKind::kMappedToCudaDevice, cuda_index, major, minor};
}

HardwareDeviceMatch VirtualAdapter(int ordinal)
{
    return HardwareDeviceMatch{ordinal, HardwareMatchKind::kVirtualAdapterUnmapped, -1, 0, 0};
}

VisibleCudaDevice CudaDevice(int index, int major, int minor = 0)
{
    return VisibleCudaDevice{index, major, minor};
}

}  // namespace

TEST(DeviceSelection, TwoVirtualAdaptersAndOneCudaGpuRegisterOnce)
{
    // Docker Desktop / WSL2: ORT lists two Microsoft virtual display adapters, CUDA lists one GPU.
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({VirtualAdapter(0), VirtualAdapter(1)}, {CudaDevice(0, 12, 0)});

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 0);
}

TEST(DeviceSelection, TwoVirtualAdaptersTakeDistinctCudaIndices)
{
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({VirtualAdapter(0), VirtualAdapter(1)}, {CudaDevice(0, 8, 9), CudaDevice(1, 12, 0)});

    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 0);
    EXPECT_EQ(selected[1].hardware_ordinal, 1);
    EXPECT_EQ(selected[1].cuda_index, 1);
}

TEST(DeviceSelection, VirtualFallbackSkipsCudaDeviceBelowAmpere)
{
    // Device 0 is visible but too old. The fallback must not stop at index 0.
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({VirtualAdapter(0), VirtualAdapter(1)}, {CudaDevice(0, 7, 5), CudaDevice(1, 12, 0)});

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 1);
}

TEST(DeviceSelection, FilteredCudaVisibilityDoesNotInventHiddenDevices)
{
    // CUDA_VISIBLE_DEVICES has already reduced the runtime to a single device at index 0.
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({VirtualAdapter(0), VirtualAdapter(1)}, {CudaDevice(0, 8, 6)});

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0].cuda_index, 0);
}

TEST(DeviceSelection, MappedDevicesKeepTheirCudaIndex)
{
    // ORT enumeration order is the reverse of the CUDA runtime order.
    const std::vector<SelectedEpDevice> selected = SelectSupportedEpDevices(
        {Mapped(0, 1, 12, 0), Mapped(1, 0, 8, 9)}, {CudaDevice(0, 8, 9), CudaDevice(1, 12, 0)});

    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 1);
    EXPECT_EQ(selected[1].hardware_ordinal, 1);
    EXPECT_EQ(selected[1].cuda_index, 0);
}

TEST(DeviceSelection, VirtualAdapterDoesNotDuplicateAMappedCudaDevice)
{
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({Mapped(0, 0, 12, 0), VirtualAdapter(1), VirtualAdapter(2)}, {CudaDevice(0, 12, 0)});

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 0);
}

TEST(DeviceSelection, MappedDeviceClaimsItsIndexBeforeVirtualAdapters)
{
    const std::vector<SelectedEpDevice> selected = SelectSupportedEpDevices(
        {VirtualAdapter(0), Mapped(1, 1, 12, 0), VirtualAdapter(2)}, {CudaDevice(0, 8, 9), CudaDevice(1, 12, 0)});

    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 0);
    EXPECT_EQ(selected[1].hardware_ordinal, 1);
    EXPECT_EQ(selected[1].cuda_index, 1);
}

TEST(DeviceSelection, LowCapabilityMappedDeviceDoesNotBlockAnotherCudaGpu)
{
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({Mapped(0, 0, 7, 5), VirtualAdapter(1)}, {CudaDevice(0, 7, 5), CudaDevice(1, 8, 9)});

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0].hardware_ordinal, 1);
    EXPECT_EQ(selected[0].cuda_index, 1);
}

TEST(DeviceSelection, NoSupportedCudaDeviceSelectsNothing)
{
    EXPECT_TRUE(SelectSupportedEpDevices({VirtualAdapter(0), VirtualAdapter(1)}, {CudaDevice(0, 7, 5)}).empty());
    EXPECT_TRUE(SelectSupportedEpDevices({VirtualAdapter(0)}, {}).empty());
    EXPECT_TRUE(SelectSupportedEpDevices({Mapped(0, 0, 7, 5)}, {CudaDevice(0, 7, 5)}).empty());
}

TEST(DeviceSelection, DuplicateMappingKeepsTheFirstHardwareDevice)
{
    const std::vector<SelectedEpDevice> selected =
        SelectSupportedEpDevices({Mapped(0, 0, 8, 6), Mapped(1, 0, 8, 6)}, {CudaDevice(0, 8, 6)});

    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0].hardware_ordinal, 0);
    EXPECT_EQ(selected[0].cuda_index, 0);
}

TEST(DeviceSelection, VendorEligibilityAcceptsMicrosoftVirtualAdapterOnLinux)
{
    EXPECT_TRUE(trt_rtx_ep::IsEligibleGpuVendor(trt_rtx_ep::kNvidiaPciVendorId));
    EXPECT_FALSE(trt_rtx_ep::IsEligibleGpuVendor(0x8086));
#if defined(_WIN32)
    EXPECT_FALSE(trt_rtx_ep::IsEligibleGpuVendor(trt_rtx_ep::kMicrosoftVirtualPciVendorId));
#else
    EXPECT_TRUE(trt_rtx_ep::IsEligibleGpuVendor(trt_rtx_ep::kMicrosoftVirtualPciVendorId));
#endif
}
