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

namespace onnxruntime
{
namespace tensorrt_rtx
{

/// Provider option name constants for TensorRT RTX EP session configuration.
namespace provider_option_names
{

/// @brief CUDA device index used by the execution provider.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// A non-negative device index available to the process.
constexpr const char* kDeviceId = "device_id";

/// @brief Set to `1` when supplying a user-managed CUDA compute stream via `user_compute_stream`.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — use an internally managed stream. `1` — use the stream supplied via `user_compute_stream`.
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";

/// @brief Pointer to a caller-owned `cudaStream_t`, encoded as a decimal string.
/// Requires `has_user_compute_stream=1`.
/// @par Type
/// String (decimal encoding of a 64-bit pointer; use ``reinterpret_cast<uintptr_t>(ptr)`` in C++)
/// @par Default
/// `0`
constexpr const char* kUserComputeStream = "user_compute_stream";

/// @brief Pointer to a caller-provided array of `cudaStream_t` used as TensorRT-RTX auxiliary
/// streams, encoded as a decimal string. Keeps auxiliary work inside the caller's CUDA
/// context, which is required for correct CIG graphics interop.
/// @par Type
/// String (decimal encoding of a 64-bit pointer; use ``reinterpret_cast<uintptr_t>(ptr)`` in C++)
/// @par Default
/// `0`
constexpr const char* kUserAuxStreamArray = "user_aux_stream_array";

/// @brief Maximum GPU memory in bytes that TensorRT-RTX may use as a workspace during engine build.
/// @par Type
/// Non-negative 64-bit integer (``uint64_t`` range)
/// @par Default
/// `0` (TensorRT-RTX default)
/// @par Accepted values
/// Non-negative value. `0` lets TensorRT-RTX choose the workspace size automatically.
constexpr const char* kMaxWorkspaceSize = "nv_max_workspace_size";

/// @brief Maximum shared memory in bytes that TensorRT-RTX kernels are allowed to use.
/// @par Type
/// Non-negative 64-bit integer (``uint64_t`` range)
/// @par Default
/// `0` (device default)
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
/// Non-negative integer. Must match the length of the array passed via `user_aux_stream_array`.
constexpr const char* kLengthAuxStreamArray = "nv_length_aux_stream_array";

/// @brief Dump partitioned subgraphs to disk for debugging.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — dump subgraphs to the current working directory.
constexpr const char* kDumpSubgraphs = "nv_dump_subgraphs";

/// @brief Enable verbose TensorRT-RTX engine build logging.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — emit detailed TensorRT-RTX build log to the ORT logger.
constexpr const char* kDetailedBuildLog = "nv_detailed_build_log";

/// @brief Minimum input shapes for TensorRT-RTX optimization profiles.
/// Specified as `input:dim0xdim1,...;input2:dim0xdim1,...`.
/// @par Type
/// String
/// @par Default
/// (empty — inferred from model)
constexpr const char* kProfilesMinShapes = "nv_profile_min_shapes";

/// @brief Maximum input shapes for TensorRT-RTX optimization profiles.
/// Specified as `input:dim0xdim1,...;input2:dim0xdim1,...`.
/// @par Type
/// String
/// @par Default
/// (empty — inferred from model)
constexpr const char* kProfilesMaxShapes = "nv_profile_max_shapes";

/// @brief Optimal input shapes for TensorRT-RTX optimization profiles.
/// Specified as `input:dim0xdim1,...;input2:dim0xdim1,...`.
/// @par Type
/// String
/// @par Default
/// (empty — inferred from model)
constexpr const char* kProfilesOptShapes = "nv_profile_opt_shapes";

/// @brief Enable CUDA graph capture to reduce kernel-launch overhead on repeated
/// inference runs with fixed input shapes.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — enable CUDA graph capture. Input shapes must be fixed across runs.
constexpr const char* kCudaGraphEnable = "enable_cuda_graph";

/// @brief Enable multi-profile support to handle multiple TensorRT-RTX optimization
/// profiles within a single session.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — single profile. `1` — enable multiple optimization profiles.
constexpr const char* kMultiProfileEnable = "nv_multi_profile_enable";

/// @brief Use external data initializers for model weights instead of embedding
/// them in the engine.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — use externally supplied weight initializers.
constexpr const char* kUseExternalDataInitializer = "nv_use_external_data_initializer";

/// @brief Directory path where compiled TensorRT-RTX engines are cached between sessions.
/// An empty string disables caching.
/// @par Type
/// String (directory path)
/// @par Default
/// (empty — caching disabled)
constexpr const char* kRuntimeCacheFile = "nv_runtime_cache_path";

/// @brief Pointer to a Vulkan `VkExternalComputeQueueDataParamsNV` blob used for CIG
/// context creation, encoded as a decimal string.
/// @par Type
/// String (decimal encoding of a 64-bit pointer; use ``reinterpret_cast<uintptr_t>(ptr)`` in C++)
/// @par Default
/// `0`
constexpr const char* kExternalComputeQueueDataParamNV_data = "VkExternalComputeQueueDataParamsNV_data";

/// @brief GPU memory budget for TensorRT RTX weight streaming.
/// @par Type
/// String
/// @par Default
/// `0` (weight streaming disabled)
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
/// (empty)
/// @par Accepted values
/// Comma-separated ONNX op type names, e.g. `Conv,Relu`. Case-sensitive.
constexpr const char* kOpTypesToExclude = "nv_op_types_to_exclude";

/// @brief Enable per-layer GPU timing via TensorRT-RTX's IProfiler. Incompatible with
/// CUDA graph capture; enabling both auto-disables CUDA graphs with a warning.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — enable per-layer profiling. Output path set via `nv_profiling_output_file`.
constexpr const char* kEnableProfiling = "nv_enable_profiling";

/// @brief File path for the Chrome tracing JSON profiling output. A timestamped
/// path is generated automatically when not specified.
/// @par Type
/// String (file path)
/// @par Default
/// (auto-generated)
constexpr const char* kProfilingOutputFile = "nv_profiling_output_file";

/// @brief Install a synchronous GPU allocator (`cudaMalloc`/`cudaFree`) on the
/// TensorRT RTX runtime, replacing the default `cudaMallocAsync` path. Enable in
/// environments where asynchronous CUDA allocation is unreliable.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — use default async allocator (`cudaMallocAsync`). `1` — use synchronous allocator (`cudaMalloc`).
constexpr const char* kUseSyncGpuAllocator = "nv_use_sync_gpu_allocator";

/// @brief Token offset at which the EP switches from the short to the long rotary
/// position embedding (RoPE) cache. Required for LongRoPE models such as Phi-4.
/// Most users can leave this at the default of ``0``.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// Non-negative integer.
constexpr const char* kMultiRotaryCacheConcatOffset = "nv_multi_rotary_cache_concat_offset";

/// @brief Experimental: enable weight-stripped EPContext engine builds. The resulting
/// engine can be refitted at load time from the original ONNX model weights.
/// Requires TensorRT RTX 1.6+.
/// @par Type
/// Boolean (`0` or `1`)
/// @par Default
/// `0`
/// @par Accepted values
/// `0` — disabled. `1` — enable weight-stripped engine builds (experimental).
constexpr const char* kWeightStrippedEngineEnableExperimental = "nv_weight_stripped_engine_enable_experimental";

}  // namespace provider_option_names

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
/// Non-negative integer in the range `[0, number_of_profiles - 1]`.
constexpr const char* kProfileIndex = "nv_profile_index";

/// @brief CUDA graph annotation ID for graph capture and replay. Use different IDs
/// to maintain separate captured graphs within a session.
/// @par Type
/// Integer
/// @par Default
/// `0`
/// @par Accepted values
/// Non-negative integer. Each unique ID maintains an independent captured graph.
constexpr const char* kCudaGraphAnnotation = "cuda_graph_annotation_id";

/// @brief Control memory arena shrinkage between inference runs.
/// @par Type
/// String
/// @par Default
/// (empty — shrinkage disabled)
/// @par Accepted values
/// `gpu:0` to enable shrinkage on GPU device 0, `gpu:0;gpu:1` for multiple devices.
constexpr const char* kMemoryArenaShrinkage = "memory.enable_memory_arena_shrinkage";

}  // namespace run_option_names

}  // namespace tensorrt_rtx
}  // namespace onnxruntime
