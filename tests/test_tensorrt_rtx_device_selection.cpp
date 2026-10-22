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

// Device selection coverage, in two layers:
//
// 1. Pairing rules for ORT hardware devices and visible CUDA devices. Pure logic, no GPU or
//    live CUDA runtime required: callers pass the probe results.
// 2. An end-to-end binding test against real devices (needs a supported GPU).

#include "tensorrt_rtx_device_selection.h"

#include "tensorrt_rtx_provider_options.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

extern std::unique_ptr<Ort::Env> ort_env;

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

TEST(DeviceSelection, VirtualFallbackSkipsCudaDeviceBelowMinimum)
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

TEST(DeviceSelection, SelectionGateMatchesTheEpComputeCapabilityGate)
{
    // The pairing rules and the EP's own device-init gate must agree on which architectures are
    // supported, or selection would publish a device the EP then refuses to run on.
    for (int major = 7; major <= 12; ++major)
    {
        for (int minor = 0; minor <= 1; ++minor)
        {
            const bool ep_supports = onnxruntime::tensorrt_rtx::IsComputeCapabilitySupported(major, minor);
            const std::vector<SelectedEpDevice> selected =
                SelectSupportedEpDevices({VirtualAdapter(0)}, {CudaDevice(0, major, minor)});
            EXPECT_EQ(!selected.empty(), ep_supports)
                << "Pairing rules and IsComputeCapabilitySupported disagree for SM " << major << "." << minor;
        }
    }
}

// Regression coverage for a real production bug (DGX Station, two GPUs: an
// unsupported datacenter Blackwell part plus a supported RTX PRO 6000). Before
// this fix, the EP resolved which CUDA device to bind to from an independent
// `device_id` provider option that could silently disagree with the
// OrtHardwareDevice ORT had actually selected for the session -- so a session
// created against "RTX PRO 6000" could end up running its engine build on
// device 0 (the other, unsupported GPU) instead. On a single-GPU machine this
// class of bug is invisible, so this test is only meaningful on a host with
// more than one visible GPU; on a single-GPU host it still exercises the
// binding path, just without the ability to catch a mismatch.

// Independently resolves the CUDA device ordinal a given OrtEpDevice's hardware
// device describes, from its LUID (Windows) / pci_bus_id (Linux) metadata --
// deliberately reimplemented here rather than calling into the EP's own
// (file-local, untestable) resolution function, so this is a genuine
// cross-check of "did the session bind to *this* device" rather than a
// tautology that would pass even if the EP's internal resolution were broken.
// Returns -1 if the device can't be resolved.
static int ResolveExpectedCudaDevice(const OrtEpDevice* ep_device)
{
    const OrtApi& api = Ort::GetApi();
    const OrtHardwareDevice* hw = api.EpDevice_Device(ep_device);
    if (hw == nullptr)
    {
        return -1;
    }
    const OrtKeyValuePairs* metadata = api.HardwareDevice_Metadata(hw);

#if defined(_WIN32)
    const char* luid_str = api.GetKeyValue(metadata, "LUID");
    if (luid_str == nullptr)
    {
        return -1;
    }
    uint64_t target_luid = 0;
    try
    {
        target_luid = std::stoull(luid_str);
    }
    catch (const std::exception&)
    {
        return -1;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess)
    {
        return -1;
    }
    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess || prop.luidDeviceNodeMask == 0)
        {
            continue;
        }
        static_assert(sizeof(prop.luid) == sizeof(uint64_t), "cudaDeviceProp::luid should be 8 bytes");
        if (*reinterpret_cast<const uint64_t*>(prop.luid) == target_luid)
        {
            return i;
        }
    }
    return -1;
#else
    const char* pci_bus_id = api.GetKeyValue(metadata, "pci_bus_id");
    if (pci_bus_id == nullptr)
    {
        return -1;
    }
    int device_index = -1;
    if (cudaDeviceGetByPCIBusId(&device_index, pci_bus_id) != cudaSuccess)
    {
        // WSL2 reports a Microsoft virtual adapter: vendor 0x1414 with a bus id the CUDA driver
        // rejects (for example 3dc3:00:00.0), which is the exact condition the EP's WSL2 fallback
        // handles. There is no PCI identity to match on, so the EP pairs the adapter with the
        // lowest supported visible CUDA device. Reproduce that rule here so this cross-check stays
        // independent of the EP's own resolution instead of failing on an unmatchable bus id.
        if (api.HardwareDevice_VendorId(hw) != 0x1414)
        {
            return -1;
        }

        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess)
        {
            return -1;
        }
        for (int i = 0; i < device_count; ++i)
        {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
            {
                continue;
            }
            if (onnxruntime::tensorrt_rtx::IsComputeCapabilitySupported(prop.major, prop.minor))
            {
                return i;
            }
        }
        return -1;
    }
    return device_index;
#endif
}

// A session pinned to one specific OrtEpDevice must actually execute CUDA
// calls against the CUDA device that OrtEpDevice describes -- not whatever
// device happens to be ambient/default on the calling thread. This directly
// guards the DGX Station failure mode: verify, for every TRT RTX EP device
// this factory publishes, that (a) it only ever describes a device whose
// compute capability TensorRT RTX actually supports, and (b) a session
// created against it ends up bound to that same, specific CUDA device (not
// merely *a* supported one -- catches a wrong-but-still-supported bind on a
// multi-GPU host) and can run.
TEST(DeviceSelection, SessionBindsToSelectedCudaDevice)
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found. Is a supported GPU available?";

    int cuda_device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&cuda_device_count), cudaSuccess);
    if (cuda_device_count < 2)
    {
        GTEST_LOG_(INFO) << "Only " << cuda_device_count
                         << " CUDA device(s) visible; this test still runs but cannot "
                            "distinguish a wrong-device bind from a correct one on a single-GPU host.";
    }

    const std::string model_name = "device_selection_test_model.onnx";
    model_builder::CreateBaseModel(model_name, "device_selection_test", {1, 2, 2});

    for (size_t i = 0; i < devices.size(); ++i)
    {
        SCOPED_TRACE("TRT RTX EP device index " + std::to_string(i));

        // Pin the session to exactly this one EP device (not the full list), so a
        // wrong-device bind on a multi-GPU host would be observable per-device.
        Ort::SessionOptions so;
        std::vector<Ort::ConstEpDevice> single_device = {devices[i]};
        Ort::KeyValuePairs kv_options;
        so.AppendExecutionProvider_V2(*ort_env, single_device, kv_options);

        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

        // The EP constructor calls cudaSetDevice(device_id_) synchronously on this
        // thread for a normal (non-user-stream) session, so the CUDA device current
        // on this thread right after construction is the device the EP actually
        // bound to.
        int bound_device = -1;
        ASSERT_EQ(cudaGetDevice(&bound_device), cudaSuccess);

        // Direct identity check: the device we asked ORT to bind to (devices[i]) must
        // be the CUDA device that actually ended up bound, not merely some supported
        // one. This is what catches the DGX Station failure mode on a host where more
        // than one visible GPU is supported.
        const OrtEpDevice* raw_ep_device = static_cast<const OrtEpDevice*>(devices[i]);
        const int expected_device = ResolveExpectedCudaDevice(raw_ep_device);
        ASSERT_NE(expected_device, -1) << "Could not independently resolve a CUDA device ordinal for TRT RTX EP "
                                          "device "
                                       << i << " from its LUID/pci_bus_id metadata.";
        EXPECT_EQ(bound_device, expected_device)
            << "Session for TRT RTX EP device " << i << " bound to CUDA device " << bound_device
            << ", but that OrtEpDevice's own LUID/pci_bus_id metadata resolves to CUDA device " << expected_device
            << " -- wrong-device bind.";

        cudaDeviceProp bound_prop{};
        ASSERT_EQ(cudaGetDeviceProperties(&bound_prop, bound_device), cudaSuccess);
        EXPECT_TRUE(onnxruntime::tensorrt_rtx::IsComputeCapabilitySupported(bound_prop.major, bound_prop.minor))
            << "Session for TRT RTX EP device " << i << " bound to CUDA device " << bound_device << " (SM "
            << bound_prop.major << "." << bound_prop.minor
            << "), which IsComputeCapabilitySupported rejects -- the EP must never execute on an "
               "unsupported/wrong GPU regardless of which OrtEpDevice the session was created against.";

        // Confirm the bound device is actually usable, not just queryable.
        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }

    clearFileIfExists(model_name);
}
