// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

struct HostMemory
{
    uint64_t private_bytes{};
    uint64_t resident_bytes{};
};

HostMemory GetHostMemory();

class MemoryMonitor
{
public:
    MemoryMonitor();
    ~MemoryMonitor();

    MemoryMonitor(const MemoryMonitor&) = delete;
    MemoryMonitor& operator=(const MemoryMonitor&) = delete;

    void Start();
    void Stop();

    HostMemory PeakHost() const;
    uint64_t PeakGpu() const;
    uint64_t BaselineGpu() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
