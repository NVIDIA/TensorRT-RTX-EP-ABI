// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
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

#include "tensorrt_rtx_provider_options.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

extern std::unique_ptr<Ort::Env> ort_env;

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
