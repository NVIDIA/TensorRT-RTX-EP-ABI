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

//!
//! \namespace onnxruntime::tensorrt_rtx::provider_option_names
//! \brief Contains provider option name constants for TensorRT RTX EP configuration.
//!
//! \details The `provider_option_names` namespace contains the following constants:
//! - `kDeviceId`: Specifies the GPU device ID to use.
//! - `kHasUserComputeStream`: Indicates whether a user-provided compute stream is used.
//! - `kUserComputeStream`: Specifies the user-provided compute stream.
//! - `kMaxWorkspaceSize`: Sets the maximum workspace size for GPU memory allocation.
//! - `kMaxSharedMemSize`: Sets the maximum amount of shared memory that TensorRT kernels are allowed to use.
//! - `kDumpSubgraphs`: Enables or disables dumping of subgraphs for debugging.
//! - `kDetailedBuildLog`: Enables or disables detailed build logs for debugging.
//! - `kProfilesMinShapes`: Specifies the minimum shapes for profiling.
//! - `kProfilesMaxShapes`: Specifies the maximum shapes for profiling.
//! - `kProfilesOptShapes`: Specifies the optimal shapes for profiling.
//! - `kCudaGraphEnable`: Enables or disables CUDA graph optimizations.
//! - `kMultiProfileEnable`: Enables or disables multi-profile support.
//! - `kUseExternalDataInitializer`: Enables or disables use of external data initializers.
//! - `kRuntimeCacheFile`: Specifies the path to the runtime cache file.
//!
//! \namespace onnxruntime::tensorrt_rtx::run_option_names
//! \brief Contains run option name constants for TensorRT RTX EP runtime configuration.
//!
//! \details The `run_option_names` namespace contains the following constants:
//! - `kProfileIndex`: Specifies the profile index to use at runtime.
//! - `kCudaGraphAnnotation`: Specifies the CUDA graph annotation ID for graph capture/replay.
//!
namespace onnxruntime
{
namespace tensorrt_rtx
{
namespace provider_option_names
{
constexpr const char* kDeviceId = "device_id";
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";
constexpr const char* kUserComputeStream = "user_compute_stream";
constexpr const char* kMaxWorkspaceSize = "nv_max_workspace_size";
constexpr const char* kMaxSharedMemSize = "nv_max_shared_mem_size";
constexpr const char* kDumpSubgraphs = "nv_dump_subgraphs";
constexpr const char* kDetailedBuildLog = "nv_detailed_build_log";
constexpr const char* kProfilesMinShapes = "nv_profile_min_shapes";
constexpr const char* kProfilesMaxShapes = "nv_profile_max_shapes";
constexpr const char* kProfilesOptShapes = "nv_profile_opt_shapes";
constexpr const char* kCudaGraphEnable = "enable_cuda_graph";
constexpr const char* kMultiProfileEnable = "nv_multi_profile_enable";
constexpr const char* kUseExternalDataInitializer = "nv_use_external_data_initializer";
constexpr const char* kRuntimeCacheFile = "nv_runtime_cache_path";

}  // namespace provider_option_names

namespace run_option_names
{
constexpr const char* kProfileIndex = "nv_profile_index";
constexpr const char* kCudaGraphAnnotation = "cuda_graph_annotation_id";
constexpr const char* kMemoryArenaShrinkage = "memory.enable_memory_arena_shrinkage";

}  // namespace run_option_names

}  // namespace tensorrt_rtx
}  // namespace onnxruntime
