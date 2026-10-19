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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

#include <onnxruntime_cxx_api.h>

// =============================================================================
// CUDA error checking
// =============================================================================

inline void cuda_error_check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        ::fprintf(stderr, "CUDA ERROR at %s[%d]: %s\n", file, line, cudaGetErrorString(err));
        abort();
    }
}

#define CUDA_CHECK(err) cuda_error_check((err), __FILE__, __LINE__)

// =============================================================================
// Path / EP helpers
// =============================================================================

// Convert a filesystem path to the string type expected by the ORT C++ API
// (wstring on Windows, string elsewhere).
inline auto toOrtString(const std::filesystem::path& p)
{
#ifdef _WIN32
    return p.wstring();
#else
    return p.string();
#endif
}

inline std::filesystem::path executableDirectory()
{
    namespace fs = std::filesystem;
#ifdef _WIN32
    std::vector<wchar_t> buffer(260);
    for (;;)
    {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
            std::fprintf(stderr, "ERROR: Could not determine the test executable path (error %lu).\n", GetLastError());
            return {};
        }
        if (length < buffer.size() - 1)
        {
            return fs::path(buffer.data(), buffer.data() + length).parent_path();
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    char buffer[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0)
    {
        std::fprintf(stderr, "ERROR: Could not determine the test executable path.\n");
        return {};
    }
    buffer[length] = '\0';
    return fs::path(buffer).parent_path();
#endif
}

inline std::filesystem::path resolveModelPath()
{
    namespace fs = std::filesystem;
    constexpr const char* kModelFileName = "resnet18_Opset18_timm.onnx";
    const fs::path model_path = executableDirectory() / kModelFileName;
    std::error_code ec;
    if (model_path.empty() || !fs::is_regular_file(model_path, ec))
    {
        std::fprintf(stderr, "ERROR: Required ResNet-18 model is not present next to the test executable: %s\n",
                     model_path.string().c_str());
    }
    return model_path;
}

inline const std::filesystem::path kModelPath = resolveModelPath();
constexpr const char* kEpName = "NvTensorRTRTXExecutionProvider";

// =============================================================================
// ONNX Runtime version check (runtime DLL, not compile-time headers)
// =============================================================================
//
// Returns true iff the currently-loaded onnxruntime.dll version parsed as
// "major.minor[.patch]" is greater than or equal to (min_major, min_minor).
// Writes the parsed version string to `version_out` for diagnostics.
//
// Intended for per-test gating (e.g. GTEST_SKIP) on features introduced in a
// specific ORT release without affecting tests that work on older runtimes.
inline bool ort_runtime_at_least(int min_major, int min_minor, std::string& version_out)
{
    const OrtApiBase* base = OrtGetApiBase();
    if (base == nullptr || base->GetVersionString == nullptr)
    {
        version_out = "<unknown>";
        return false;
    }
    const char* v = base->GetVersionString();
    version_out = (v != nullptr) ? v : "<null>";
    if (v == nullptr)
        return false;

    int major = 0;
    int minor = 0;
    if (std::sscanf(v, "%d.%d", &major, &minor) < 2)
        return false;

    if (major != min_major)
        return major > min_major;
    return minor >= min_minor;
}

// Shorthand that swallows the version string.
inline bool ort_runtime_at_least(int min_major, int min_minor)
{
    std::string unused;
    return ort_runtime_at_least(min_major, min_minor, unused);
}

// =============================================================================
// EP-side ORT API version check (read from EpDevice metadata)
// =============================================================================
//
// The TRT RTX EP publishes its negotiated ORT API version into EpDevice
// metadata under the key "nv_ep_ort_api_version" (set in
// src/tensorrt_rtx_provider_factory.cc::GetSupportedDevicesImpl). Tests that
// exercise version-gated EP features (graphics interop, mempool Shrink, ...)
// should read this value and skip when the EP was negotiated below the
// version that introduced the feature.
//
// Returns the negotiated EP API version (e.g. 24, 25, 26), or -1 if the EP is
// older and does not publish the key — callers should treat -1 as "unknown".
inline int ep_negotiated_ort_api_version(const OrtEpDevice* ep_device)
{
    if (ep_device == nullptr)
        return -1;
    const OrtApi& api = Ort::GetApi();
    const OrtKeyValuePairs* md = api.EpDevice_EpMetadata(ep_device);
    if (md == nullptr)
        return -1;
    const char* v = api.GetKeyValue(md, "nv_ep_ort_api_version");
    if (v == nullptr)
        return -1;
    return std::atoi(v);
}

// Returns the subset of EP devices that belong to the TRT RTX EP.
inline std::vector<Ort::ConstEpDevice> get_trt_rtx_devices(Ort::Env& env)
{
    std::vector<Ort::ConstEpDevice> result;
    for (auto& device : env.GetEpDevices())
    {
        if (std::strcmp(device.EpName(), kEpName) == 0)
        {
            result.push_back(device);
        }
    }
    return result;
}

// =============================================================================
// Session inspection
// =============================================================================

// Print input/output names, dtypes, and shapes to stdout.
void describe_session(Ort::Session& session);

// =============================================================================
// Type / size helpers
// =============================================================================

size_t ONNXDtypeToBytes(ONNXTensorElementDataType element_type);
size_t ORTValueToBytes(const Ort::Value& value);

// =============================================================================
// Inference runners
// =============================================================================

// Run `iterations` forward passes using CPU-side IoBinding.
// Inputs are zero-initialised. Prints average per-iteration time to stdout.
void run_with_cpu_bindings(Ort::Session& session, int iterations);

// Run `iterations` forward passes using GPU-side IoBinding with async
// upload/download streams. Prints average per-iteration time to stdout.
void run_with_gpu_bindings(Ort::Session& session, int iterations, cudaStream_t stream);

// =============================================================================
// File helpers
// =============================================================================

inline void clearFileIfExists(const std::filesystem::path& path)
{
    if (std::filesystem::exists(path))
    {
        std::filesystem::remove(path);
    }
}

// =============================================================================
// IoBinding helper
// =============================================================================

// Create an IoBinding from session metadata. Allocates input tensors on CPU
// (or with the given allocator). Dynamic dims (-1) are replaced with 1 unless
// overridden via shape_overwrites.
Ort::IoBinding generate_io_binding(Ort::Session& session,
                                   std::map<std::string, std::vector<int64_t>> shape_overwrites = {},
                                   OrtAllocator* allocator = nullptr);
