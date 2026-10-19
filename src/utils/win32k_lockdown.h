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

#if defined(_WIN32)
// Keep windows.h from defining min/max as macros: this header is included by translation units
// that use std::numeric_limits<>::max().
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace trt_rtx_ep
{

//!
//! \brief Whether this process is barred from making win32k system calls.
//!
//! \details Windows applies that mitigation at process creation, so it cannot be lifted for
//! initialization: no code in the process, however early, reaches the display driver. Both ways of
//! asking the driver about a GPU are lost, and they fail differently — the CUDA runtime faults
//! inside cudaGetDeviceCount, while NVML reports NVML_ERROR_DRIVER_NOT_LOADED. A host that
//! isolates compilation this way is the case ORT's "allow_virtual_devices" covers, where an EP
//! describes the device it compiles for instead of discovering it.
//!
inline bool IsWin32kSystemCallsDisabled()
{
#if defined(_WIN32)
    PROCESS_MITIGATION_SYSTEM_CALL_DISABLE_POLICY policy{};
    if (!GetProcessMitigationPolicy(GetCurrentProcess(), ProcessSystemCallDisablePolicy, &policy, sizeof(policy)))
    {
        return false;
    }
    return policy.DisallowWin32kSystemCalls != 0;
#else
    return false;
#endif
}

//!
//! \brief IsWin32kSystemCallsDisabled(), cached.
//!
//! \details Cached because a process mitigation cannot be changed once the process is running.
//! Call this before any CUDA entry point that runs during option parsing, device enumeration or
//! allocator setup: those paths execute even for a session that only compiles, where a driver call
//! is both unnecessary and fatal.
//!
//! This answers one narrow question — whether this process is under the win32k mitigation — and
//! must not be read as a general test for driver availability. A machine with no NVIDIA driver
//! installed, or a container without one mapped in, reports no lockdown; those cases surface as
//! usual, through a CUDA call that reports zero devices.
//!
inline bool IsWin32kLockdownActive()
{
    static const bool active = IsWin32kSystemCallsDisabled();
    return active;
}

}  // namespace trt_rtx_ep
