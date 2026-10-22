// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "trt_ep_perf_test_memory_monitor.h"

#include <cuda_runtime_api.h>
#ifdef _WIN32
#include <cuda.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_4.h>
#include <psapi.h>
#include <windows.h>
#else
#include <nvml.h>

#include <vector>

#include <unistd.h>
#endif

HostMemory GetHostMemory()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              sizeof(counters)))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    return {static_cast<uint64_t>(counters.PrivateUsage), static_cast<uint64_t>(counters.WorkingSetSize)};
#else
    std::ifstream status("/proc/self/status");
    std::string key;
    uint64_t value{};
    std::string unit;
    HostMemory result;
    while (status >> key >> value >> unit)
    {
        if (key == "VmRSS:")
            result.resident_bytes = value * 1024;
        if (key == "RssAnon:")
            result.private_bytes = value * 1024;
    }
    return result;
#endif
}

class MemoryMonitor::Impl
{
public:
    Impl()
    {
#ifndef _WIN32
        const nvmlReturn_t init = nvmlInit_v2();
        nvml_available_ = init == NVML_SUCCESS;
#endif
        // Establish the primary context before the load interval so its
        // one-time allocation does not appear as engine-load memory.
        (void)cudaFree(nullptr);
#ifdef _WIN32
        InitializeDxgiAdapter();
#endif
    }

    ~Impl()
    {
        Stop();
#ifdef _WIN32
        if (dxgi_adapter_)
            dxgi_adapter_->Release();
#else
        if (nvml_available_)
            (void)nvmlShutdown();
#endif
    }

    void Start()
    {
        peak_host_ = GetHostMemory();
        peak_gpu_ = CurrentGpuMemory();
        baseline_gpu_ = peak_gpu_;
        running_ = true;
        thread_ = std::thread(
            [this]
            {
                while (running_)
                {
                    const auto host = GetHostMemory();
                    peak_host_.private_bytes = std::max(peak_host_.private_bytes, host.private_bytes);
                    peak_host_.resident_bytes = std::max(peak_host_.resident_bytes, host.resident_bytes);
                    peak_gpu_ = std::max(peak_gpu_, CurrentGpuMemory());
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
    }

    void Stop()
    {
        running_ = false;
        if (thread_.joinable())
            thread_.join();
    }

    HostMemory PeakHost() const
    {
        return peak_host_;
    }
    uint64_t PeakGpu() const
    {
        return peak_gpu_;
    }
    uint64_t BaselineGpu() const
    {
        return baseline_gpu_;
    }

private:
#ifdef _WIN32
    void InitializeDxgiAdapter()
    {
        int cuda_ordinal{};
        CUdevice cuda_device{};
        LUID cuda_luid{};
        unsigned int cuda_node_mask{};
        if (cudaGetDevice(&cuda_ordinal) != cudaSuccess || cuDeviceGet(&cuda_device, cuda_ordinal) != CUDA_SUCCESS ||
            cuDeviceGetLuid(reinterpret_cast<char*>(&cuda_luid), &cuda_node_mask, cuda_device) != CUDA_SUCCESS ||
            cuda_node_mask == 0)
            return;
        while ((cuda_node_mask & 1U) == 0)
        {
            ++dxgi_node_index_;
            cuda_node_mask >>= 1;
        }

        IDXGIFactory1* factory{};
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            return;

        for (UINT index{};; ++index)
        {
            IDXGIAdapter1* adapter{};
            const HRESULT status = factory->EnumAdapters1(index, &adapter);
            if (status == DXGI_ERROR_NOT_FOUND)
                break;
            if (FAILED(status))
                continue;

            DXGI_ADAPTER_DESC1 description{};
            if (SUCCEEDED(adapter->GetDesc1(&description)) && description.AdapterLuid.LowPart == cuda_luid.LowPart &&
                description.AdapterLuid.HighPart == cuda_luid.HighPart)
            {
                (void)adapter->QueryInterface(IID_PPV_ARGS(&dxgi_adapter_));
                adapter->Release();
                break;
            }
            adapter->Release();
        }
        factory->Release();
    }

    uint64_t CurrentGpuMemory() const
    {
        if (!dxgi_adapter_)
            return 0;
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        return SUCCEEDED(
                   dxgi_adapter_->QueryVideoMemoryInfo(dxgi_node_index_, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory))
                   ? memory.CurrentUsage
                   : 0;
    }
#else
    uint64_t CurrentGpuMemory() const
    {
        if (!nvml_available_)
            return 0;
        unsigned int device_count{};
        if (nvmlDeviceGetCount_v2(&device_count) != NVML_SUCCESS)
            return 0;
        const unsigned int pid =
#ifdef _WIN32
            GetCurrentProcessId();
#else
            static_cast<unsigned int>(getpid());
#endif
        uint64_t total{};
        for (unsigned int index = 0; index < device_count; ++index)
        {
            nvmlDevice_t device{};
            if (nvmlDeviceGetHandleByIndex_v2(index, &device) != NVML_SUCCESS)
                continue;
            const auto process_memory = [device, pid](auto query)
            {
                unsigned int count{};
                if (query(device, &count, nullptr) != NVML_ERROR_INSUFFICIENT_SIZE || count == 0)
                    return uint64_t{};
                std::vector<nvmlProcessInfo_t> processes(count);
                if (query(device, &count, processes.data()) != NVML_SUCCESS)
                    return uint64_t{};
                for (const auto& process : processes)
                    if (process.pid == pid && process.usedGpuMemory != NVML_VALUE_NOT_AVAILABLE)
                        return static_cast<uint64_t>(process.usedGpuMemory);
                return uint64_t{};
            };
            // WDDM can report a CUDA process via either list. They represent the
            // same per-device allocation, so use the larger value rather than
            // summing and double-counting it.
            total += std::max(process_memory(nvmlDeviceGetComputeRunningProcesses_v3),
                              process_memory(nvmlDeviceGetGraphicsRunningProcesses_v3));
        }
        return total;
    }
#endif

#ifdef _WIN32
    IDXGIAdapter3* dxgi_adapter_{};
    UINT dxgi_node_index_{};
#else
    bool nvml_available_{};
#endif
    std::atomic<bool> running_{false};
    std::thread thread_;
    HostMemory peak_host_{};
    uint64_t peak_gpu_{};
    uint64_t baseline_gpu_{};
};

MemoryMonitor::MemoryMonitor()
    : impl_(std::make_unique<Impl>())
{
}
MemoryMonitor::~MemoryMonitor() = default;

void MemoryMonitor::Start()
{
    impl_->Start();
}
void MemoryMonitor::Stop()
{
    impl_->Stop();
}
HostMemory MemoryMonitor::PeakHost() const
{
    return impl_->PeakHost();
}
uint64_t MemoryMonitor::PeakGpu() const
{
    return impl_->PeakGpu();
}
uint64_t MemoryMonitor::BaselineGpu() const
{
    return impl_->BaselineGpu();
}
