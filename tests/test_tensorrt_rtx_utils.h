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

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

// =============================================================================
// CUDA error checking
// =============================================================================

inline void cuda_error_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        ::fprintf(stderr, "CUDA ERROR at %s[%d]: %s\n", file, line,
                  cudaGetErrorString(err));
        abort();
    }
}

#define CUDA_CHECK(err) cuda_error_check((err), __FILE__, __LINE__)

// =============================================================================
// Path / EP helpers
// =============================================================================

// Convert a filesystem path to the string type expected by the ORT C++ API
// (wstring on Windows, string elsewhere).
inline auto toOrtString(const std::filesystem::path& p) {
#ifdef _WIN32
    return p.wstring();
#else
    return p.string();
#endif
}

// MODEL_FILE is injected as a compile definition by CMake.
inline const std::filesystem::path kModelPath{MODEL_FILE};
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
inline bool ort_runtime_at_least(int min_major, int min_minor,
                                 std::string& version_out) {
    const OrtApiBase* base = OrtGetApiBase();
    if (base == nullptr || base->GetVersionString == nullptr) {
        version_out = "<unknown>";
        return false;
    }
    const char* v = base->GetVersionString();
    version_out = (v != nullptr) ? v : "<null>";
    if (v == nullptr) return false;

    int major = 0;
    int minor = 0;
    if (std::sscanf(v, "%d.%d", &major, &minor) < 2) return false;

    if (major != min_major) return major > min_major;
    return minor >= min_minor;
}

// Shorthand that swallows the version string.
inline bool ort_runtime_at_least(int min_major, int min_minor) {
    std::string unused;
    return ort_runtime_at_least(min_major, min_minor, unused);
}

// Returns the subset of EP devices that belong to the TRT RTX EP.
inline std::vector<Ort::ConstEpDevice> get_trt_rtx_devices(Ort::Env& env) {
    std::vector<Ort::ConstEpDevice> result;
    for (auto& device : env.GetEpDevices()) {
        if (std::strcmp(device.EpName(), kEpName) == 0) {
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
void run_with_gpu_bindings(Ort::Session& session, int iterations,
                           cudaStream_t stream);

// =============================================================================
// File helpers
// =============================================================================

inline void clearFileIfExists(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

// =============================================================================
// IoBinding helper
// =============================================================================

// Create an IoBinding from session metadata. Allocates input tensors on CPU
// (or with the given allocator). Dynamic dims (-1) are replaced with 1 unless
// overridden via shape_overwrites.
Ort::IoBinding generate_io_binding(
    Ort::Session& session,
    std::map<std::string, std::vector<int64_t>> shape_overwrites = {},
    OrtAllocator* allocator = nullptr);
