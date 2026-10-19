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

#include <algorithm>
#include <string>
#include <string_view>

namespace onnxruntime
{
namespace tensorrt_rtx
{

/// Provider option name constants for TensorRT RTX EP session configuration.
namespace provider_option_names
{

/// @brief Set to `1` when supplying a user-managed CUDA compute stream via `user_compute_stream`.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — use an internally managed stream. `1` — use the stream supplied via `user_compute_stream`.
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";

/// @brief Pointer to a caller-owned `cudaStream_t`, encoded as a decimal string.
/// Requires `has_user_compute_stream=1`. In C++, encode the pointer with
/// `reinterpret_cast<uintptr_t>(ptr)`.
/// @par Type
/// String
/// @par Default
/// `0`
/// @par Accepted values
/// `0` means no user-managed stream pointer is supplied. A nonzero value is the decimal encoding
/// of a caller-owned `cudaStream_t` pointer and is used when `has_user_compute_stream` is `1`.
constexpr const char* kUserComputeStream = "user_compute_stream";

/// @brief Pointer to a caller-provided array of `cudaStream_t` used as TensorRT-RTX auxiliary
/// streams, encoded as a decimal string. Keeps auxiliary work inside the caller's CUDA
/// context, which is required for correct CIG graphics interop. In C++, encode the pointer
/// with `reinterpret_cast<uintptr_t>(ptr)`.
/// @par Type
/// String
/// @par Default
/// `0`
/// @par Accepted values
/// `0` means no user-provided auxiliary stream array is supplied. A nonzero value is the decimal
/// encoding of a pointer to an array containing `nv_length_aux_stream_array` streams.
constexpr const char* kUserAuxStreamArray = "user_aux_stream_array";

/// @brief Maximum GPU memory in bytes that TensorRT-RTX may use as a workspace during engine build.
/// The value must fit in the `uint64_t` range.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// Non-negative value. `0` lets TensorRT-RTX choose the workspace size automatically.
constexpr const char* kMaxWorkspaceSize = "nv_max_workspace_size";

/// @brief Maximum shared memory in bytes that TensorRT-RTX kernels are allowed to use.
/// The value must fit in the `uint64_t` range.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// Non-negative value. `0` uses the device maximum.
constexpr const char* kMaxSharedMemSize = "nv_max_shared_mem_size";

/// @brief Number of entries in `user_aux_stream_array`; also sets the maximum number of
/// TensorRT-RTX auxiliary streams.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// `0` configures no user-provided auxiliary streams. Other non-negative integers must match the
/// length of the array passed via `user_aux_stream_array`.
constexpr const char* kLengthAuxStreamArray = "nv_length_aux_stream_array";

/// @brief Dump partitioned subgraphs to disk for debugging.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — dump subgraphs to the current working directory.
constexpr const char* kDumpSubgraphs = "nv_dump_subgraphs";

/// @brief Enable verbose TensorRT-RTX engine build logging.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — emit detailed TensorRT-RTX build log to the ORT logger.
constexpr const char* kDetailedBuildLog = "nv_detailed_build_log";

/// @brief Minimum input shapes for TensorRT-RTX optimization profiles.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` infers minimum shapes from the model. Otherwise, use
/// `input:dim0xdim1,...;input2:dim0xdim1,...`.
constexpr const char* kProfilesMinShapes = "nv_profile_min_shapes";

/// @brief Maximum input shapes for TensorRT-RTX optimization profiles.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` infers maximum shapes from the model. Otherwise, use
/// `input:dim0xdim1,...;input2:dim0xdim1,...`.
constexpr const char* kProfilesMaxShapes = "nv_profile_max_shapes";

/// @brief Optimal input shapes for TensorRT-RTX optimization profiles.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` infers optimal shapes from the model. Otherwise, use
/// `input:dim0xdim1,...;input2:dim0xdim1,...`.
constexpr const char* kProfilesOptShapes = "nv_profile_opt_shapes";

/// @brief Enable CUDA graph capture to reduce kernel-launch overhead on repeated
/// inference runs with fixed input shapes.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — enable CUDA graph capture. Input shapes must be fixed across runs.
constexpr const char* kCudaGraphEnable = "enable_cuda_graph";

/// @brief Enable multi-profile support to handle multiple TensorRT-RTX optimization
/// profiles within a single session.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — single profile. `1` — enable multiple optimization profiles.
constexpr const char* kMultiProfileEnable = "nv_multi_profile_enable";

/// @brief Use external data initializers for model weights instead of embedding
/// them in the engine.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — use externally supplied weight initializers.
constexpr const char* kUseExternalDataInitializer = "nv_use_external_data_initializer";

/// @brief Directory path where compiled TensorRT-RTX engines are cached between sessions.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` disables caching. Any non-empty value specifies the cache directory path.
constexpr const char* kRuntimeCacheFile = "nv_runtime_cache_path";

/// @brief Pointer to a Vulkan `VkExternalComputeQueueDataParamsNV` blob used for CIG
/// context creation, encoded as a decimal string. In C++, encode the pointer with
/// `reinterpret_cast<uintptr_t>(ptr)`.
/// @par Type
/// String
/// @par Default
/// `0`
/// @par Accepted values
/// `0` means no external compute queue data is supplied. A nonzero value is the decimal encoding
/// of a pointer to a Vulkan `VkExternalComputeQueueDataParamsNV` blob.
constexpr const char* kExternalComputeQueueDataParamNV_data = "VkExternalComputeQueueDataParamsNV_data";

/// @brief GPU memory budget for TensorRT RTX weight streaming.
/// A nonzero value enables weight streaming when building engines. The budget can be
/// updated persistently for subsequent runs with `Ort::Session::SetEpDynamicOptions`.
/// If CUDA graph replay is enabled, TensorRT RTX determines graph eligibility from
/// the resolved budget when creating each execution context.
/// @par Type
/// String
/// @par Default
/// `0`
/// @par Dynamic updates
/// Supported
/// @par Accepted values
/// `0` disables streaming. `-1` uses TensorRT-RTX's automatic budget. `1` requests minimum VRAM
/// mode. Bare integers greater than `1` set an explicit byte budget. Suffixed values (`B`,
/// `K`, `M`, `G`) specify resident bytes (e.g. `512M`). Percentage values `0%`–`100%`
/// specify the fraction of streamable weights to keep resident in VRAM.
constexpr const char* kWeightStreamingBudget = "nv_weight_streaming_budget";

/// @brief Comma-separated list of ONNX op types that the EP should leave to other
/// execution providers during graph partitioning.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` excludes no operations. Otherwise, provide a case-sensitive, comma-separated list of ONNX
/// op type names, for example `Conv,Relu`.
constexpr const char* kOpTypesToExclude = "nv_op_types_to_exclude";

/// @brief Enable per-layer GPU timing via TensorRT-RTX's IProfiler. Incompatible with
/// CUDA graph capture; enabling both auto-disables CUDA graphs with a warning.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — enable per-layer profiling. Output path set via `nv_profiling_output_file`.
constexpr const char* kEnableProfiling = "nv_enable_profiling";

/// @brief File path for the Chrome tracing JSON profiling output.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` generates a timestamped output path automatically. Any non-empty value specifies the
/// output file path.
constexpr const char* kProfilingOutputFile = "nv_profiling_output_file";

/// @brief Install a synchronous GPU allocator (`cudaMalloc`/`cudaFree`) on the
/// TensorRT RTX runtime, replacing the default `cudaMallocAsync` path. Enable in
/// environments where asynchronous CUDA allocation is unreliable.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — use default async allocator (`cudaMallocAsync`). `1` — use synchronous allocator (`cudaMalloc`).
constexpr const char* kUseSyncGpuAllocator = "nv_use_sync_gpu_allocator";

/// @brief Retain execution context device memory between runs in both normal and EPContext execution.
/// Allocates the engine-wide memory requirement on first use, avoiding per-run allocations at the cost
/// of retaining GPU memory until the session is destroyed, including for smaller dynamic shapes.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — allocate memory per run. `1` — retain and reuse memory for each execution context.
constexpr const char* kPersistentContextMemory = "nv_persistent_context_memory";

/// @brief Token offset at which the EP switches from the short to the long rotary
/// position embedding (RoPE) cache. Required for LongRoPE models such as Phi-4.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// `0` applies no token offset. Other non-negative integers set the token offset at which the EP
/// switches to the long RoPE cache.
constexpr const char* kMultiRotaryCacheConcatOffset = "nv_multi_rotary_cache_concat_offset";

/// @brief Experimental: enable weight-stripped EPContext engine builds. The resulting
/// engine can be refitted at load time from the original ONNX model weights.
/// Requires TensorRT RTX 1.7+.
/// @par Type
/// Boolean
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — enable weight-stripped engine builds (experimental).
constexpr const char* kWeightStrippedEngineEnableExperimental = "nv_weight_stripped_engine_enable_experimental";

/// @brief Target GPU SM architecture for ahead-of-time (AOT) compilation. When set on a
/// compile-only session the EP skips all CUDA and device initialisation, allowing engine
/// builds in environments with no GPU driver access (e.g. a sandboxed compiler process).
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` uses the current GPU architecture via `kCURRENT`. `"sm86"` selects Ampere (RTX 30xx).
/// `"sm89"` selects Ada (RTX 40xx). `"sm120"` selects Blackwell (RTX 50xx). `"sm121"` selects
/// GB10 (RTX and DGX Spark).
constexpr const char* kTargetSM = "nv_target_sm";

}  // namespace provider_option_names

//!
//! \namespace onnxruntime::tensorrt_rtx::env_config_names
//! \brief Keys this EP reads from the environment's configuration entries rather than from
//! provider options.
//!
//! \details An application supplies these through OrtEnvCreationOptions::config_entries, spelled
//! with the prefix ORT reserves for an execution provider: "ep_factory.<ep_name>.<key>". They are
//! read during device enumeration, which runs when the library is registered and therefore before
//! any session exists, so a provider option cannot carry them. Both describe a device the process
//! cannot see, and are only consulted when it cannot reach the driver.
//!
namespace env_config_names
{
//! - `kTargetSM`: the architecture to build for, in the spelling provider_option_names::kTargetSM
//!   accepts. Republished as a default option on the described device, so it reaches every session
//!   created on that device.
constexpr const char* kTargetSM = provider_option_names::kTargetSM;

//! - `kHardwareDeviceId`: the OrtHardwareDevice identifier of the GPU being compiled for, as an
//!   unsigned decimal string. Optional; 0 is used when absent.
constexpr const char* kHardwareDeviceId = "nv_hardware_device_id";

}  // namespace env_config_names

// Single source of truth for SM validation — used by the option parser and device enumeration.
inline constexpr std::string_view kTargetSMValues[] = {"sm86", "sm89", "sm120", "sm121"};

inline bool IsTargetSMSupported(std::string_view value)
{
    return std::find(std::begin(kTargetSMValues), std::end(kTargetSMValues), value) != std::end(kTargetSMValues);
}

//! \brief True if a device/target with this compute capability is supported by TensorRT RTX.
//!
//! Named SMs (8.6, 8.9) are checked against kTargetSMValues above so this can't drift from the
//! AOT target list. CC 12.0 and newer is accepted unconditionally rather than requiring an exact
//! kTargetSMValues entry: kTargetSMValues only enumerates the specific "smNNN" strings recognized
//! for the nv_target_sm compile-only option, and new Blackwell-or-later silicon is expected to run
//! before a matching entry is added there.
inline bool IsComputeCapabilitySupported(int major, int minor)
{
    const int compute_capability = major * 10 + minor;
    if (compute_capability >= 120)
    {
        return true;
    }
    return IsTargetSMSupported("sm" + std::to_string(compute_capability));
}

/// Run option name constants for TensorRT RTX EP per-inference configuration.
namespace run_option_names
{

/// @brief TensorRT-RTX optimization profile index to use at runtime. Must be within
/// the range of profiles built into the engine.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// `0` selects the first optimization profile. Other non-negative integers select the corresponding
/// profile in the range `[0, number_of_profiles - 1]`.
constexpr const char* kProfileIndex = "nv_profile_index";

/// @brief CUDA graph annotation ID for graph capture and replay. Use different IDs
/// to maintain separate captured graphs within a session.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// `0` identifies the default captured graph. Other non-negative integers identify independent
/// captured graphs within the session.
constexpr const char* kCudaGraphAnnotation = "cuda_graph_annotation_id";

/// @brief Control memory arena shrinkage between inference runs.
/// @par Type
/// String
/// @par Default
/// `""`
/// @par Accepted values
/// `""` disables arena shrinkage. Use `gpu:0` to enable shrinkage on GPU device 0 or a semicolon-
/// separated list such as `gpu:0;gpu:1` for multiple devices.
constexpr const char* kMemoryArenaShrinkage = "memory.enable_memory_arena_shrinkage";

}  // namespace run_option_names

}  // namespace tensorrt_rtx
}  // namespace onnxruntime
