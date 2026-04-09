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

#include "utils/provider_options.h"

#include <string>
#include <unordered_map>

//!
//! \brief Information needed to construct TRT RTX execution providers.
//!
struct TensorrtRtxExecutionProviderInfo
{
    int device_id{0};
    bool has_user_compute_stream{false};
    void* user_compute_stream{nullptr};
    int max_partition_iterations{1000};
    int min_subgraph_size{1};
    size_t max_workspace_size{0};
    size_t max_shared_mem_size{0};
    bool dump_subgraphs{false};
    std::string engine_cache_path{""};
    bool weight_stripped_engine_enable{false};
    std::string onnx_model_folder_path{""};
    const void* onnx_bytestream{nullptr};
    size_t onnx_bytestream_size{0};
    bool use_external_data_initializer{true};
    const void* external_data_bytestream{nullptr};
    size_t external_data_bytestream_size{0};
    bool engine_decryption_enable{false};
    std::string engine_decryption_lib_path{""};
    bool force_sequential_engine_build{false};
    std::string runtime_cache_path{""};
    bool detailed_build_log{false};
    bool sparsity_enable{false};
    int auxiliary_streams{-1};
    std::string extra_plugin_lib_paths{""};
    std::string profile_min_shapes{""};
    std::string profile_max_shapes{""};
    std::string profile_opt_shapes{""};
    bool cuda_graph_enable{true};
    bool multi_profile_enable{false};
    bool dump_ep_context_model{false};
    std::string ep_context_file_path{""};
    int ep_context_embed_mode{0};
    std::string engine_cache_prefix{""};
    std::string op_types_to_exclude{""};

    static TensorrtRtxExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
};
