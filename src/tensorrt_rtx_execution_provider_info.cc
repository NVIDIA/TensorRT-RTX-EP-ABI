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

#include "tensorrt_rtx_execution_provider_info.h"

#include "tensorrt_rtx_provider_options.h"
#include "utils/cuda/cuda_call.h"
#include "utils/provider_options_utils.h"
#include "utils/ep_utils.h"

#include "onnxruntime_session_options_config_keys.h"

#include <cuda_runtime.h>

#include <sstream>

TensorrtRtxExecutionProviderInfo TensorrtRtxExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options)
{
    TensorrtRtxExecutionProviderInfo info{};

    void* user_compute_stream = nullptr;
    void* onnx_bytestream = nullptr;
    void* external_data_bytestream = nullptr;
    THROW_IF_ERROR(
        ProviderOptionsParser{}
            .AddValueParser(
                onnxruntime::tensorrt_rtx::provider_option_names::kDeviceId,
                [&info](const std::string& value_str) -> OrtStatus*
                {
                    RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                    int num_devices{};
                    RETURN_IF_ERROR(CUDA_CALL(cudaGetDeviceCount(&num_devices)));
                    RETURN_IF_NOT(
                        0 <= info.device_id && info.device_id < num_devices,
                        "Invalid device ID: ", info.device_id,
                        ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                    return nullptr;
                })
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kHasUserComputeStream, info.has_user_compute_stream)
            .AddValueParser(
                onnxruntime::tensorrt_rtx::provider_option_names::kUserComputeStream,
                [&user_compute_stream](const std::string& value_str) -> OrtStatus*
                {
                    size_t address;
                    RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                    user_compute_stream = reinterpret_cast<void*>(address);
                    return nullptr;
                })
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kMaxWorkspaceSize, info.max_workspace_size)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kMaxSharedMemSize, info.max_shared_mem_size)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kDumpSubgraphs, info.dump_subgraphs)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kDetailedBuildLog, info.detailed_build_log)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kProfilesMinShapes, info.profile_min_shapes)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kProfilesMaxShapes, info.profile_max_shapes)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kProfilesOptShapes, info.profile_opt_shapes)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kCudaGraphEnable, info.cuda_graph_enable)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kUseExternalDataInitializer, info.use_external_data_initializer)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kMultiProfileEnable, info.multi_profile_enable)
            .AddAssignmentToReference(onnxruntime::tensorrt_rtx::provider_option_names::kRuntimeCacheFile, info.runtime_cache_path)
            .AddAssignmentToReference(kOrtSessionOptionEpContextEnable, info.dump_ep_context_model)
            .AddAssignmentToReference(kOrtSessionOptionEpContextFilePath, info.ep_context_file_path)
            .AddAssignmentToReference(kOrtSessionOptionEpContextEmbedMode, info.ep_context_embed_mode)
            // .AddAssignmentToReference(kOrtSessionOptionsDisableModelCompile, info.disable_model_compile)

            .Parse(options));  // add new provider option here.

    info.user_compute_stream = user_compute_stream;
    info.has_user_compute_stream = (user_compute_stream != nullptr);
    info.onnx_bytestream = onnx_bytestream;
    info.external_data_bytestream = external_data_bytestream;

    return info;
}
