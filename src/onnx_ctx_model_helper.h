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

#include "tensorrt_rtx_execution_provider.h"
#include "ep_utils.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace trt_rtx_ep
{

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string MAIN_CONTEXT = "main_context";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string COMPUTE_CAPABILITY = "hardware_architecture";
static const std::string ONNX_MODEL_FILENAME = "onnx_model_filename";
static const std::string PARTITION_NAME = "partition_name";
static const std::string SDK_VERSION = "ep_sdk_version";
static const std::string SOURCE = "source";
static const std::string EPCONTEXT_OP_DOMAIN = "com.microsoft";

bool IsAbsolutePath(const std::string& path_string);
bool IsRelativePathToParentPath(const std::string& path_string);
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::filesystem::path& ep_context_file_path);
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::string&) = delete;

std::filesystem::path GetCtxModelPath(const std::filesystem::path& ep_context_file_path,
                                      const std::filesystem::path& original_model_path);
std::filesystem::path GetCtxModelPath(const std::string&, const std::string&) = delete;

//!
//! \brief Class to create an EPContext node from an ORT's fused_node.
//!
//! \note The class can be instantiated many times during EP's Compile() as to generate the EPContext nodes from
//!       fused_nodes/subgraphs and returns them to ORT via Compile(). ORT will end up creating the EPContext model.
//!
class EPContextNodeHelper : public ApiPtrs
{
public:
    EPContextNodeHelper(TensorrtRtxExecutionProvider& ep,
                        const OrtGraph* graph,
                        const OrtNode* fused_node)
        : ApiPtrs{static_cast<const ApiPtrs&>(ep)}, ep_(ep), graph_(graph), fused_node_(fused_node)
    {
    }

    OrtStatus* CreateEPContextNode(const std::filesystem::path& engine_cache_path,
                                   char* engine_data,
                                   size_t size,
                                   const int64_t embed_mode,
                                   const std::string& compute_capability,
                                   const std::filesystem::path& onnx_model_path,
                                   OrtNode** ep_context_node);
    OrtStatus* CreateEPContextNode(const std::string&, char*, size_t, const int64_t,
                                   const std::string&, const std::string&, OrtNode**) = delete;

private:
    TensorrtRtxExecutionProvider& ep_;
    const OrtGraph* graph_ = nullptr;
    const OrtNode* fused_node_ = nullptr;
};

//!
//! \brief Class to read an OrtGraph that contains an EPContext node and get the engine binary accordingly.
//!
class EPContextNodeReader : public ApiPtrs
{
public:
    EPContextNodeReader(TensorrtRtxExecutionProvider& ep,
                        const OrtLogger& logger,
                        std::unique_ptr<nvinfer1::ICudaEngine>* trt_rtx_engine,
                        nvinfer1::IRuntime* trt_rtx_runtime,
                        std::string ep_context_model_path,
                        std::string compute_capability,
                        bool weight_stripped_engine_refit,
                        std::string onnx_model_folder_path,
                        const void* onnx_model_bytestream,
                        size_t onnx_model_bytestream_size,
                        const void* onnx_external_data_bytestream,
                        size_t onnx_external_data_bytestream_size,
                        bool detailed_build_log)
        : ApiPtrs{static_cast<const ApiPtrs&>(ep)},
          ep_(ep),
          logger_(logger),
          trt_rtx_engine_(trt_rtx_engine),
          trt_rtx_runtime_(trt_rtx_runtime),
          ep_context_model_path_(ep_context_model_path),
          compute_capability_(compute_capability),
          weight_stripped_engine_refit_(weight_stripped_engine_refit),
          onnx_model_folder_path_(onnx_model_folder_path),
          onnx_model_bytestream_(onnx_model_bytestream),
          onnx_model_bytestream_size_(onnx_model_bytestream_size),
          onnx_external_data_bytestream_(onnx_external_data_bytestream),
          onnx_external_data_bytestream_size_(onnx_external_data_bytestream_size),
          detailed_build_log_(detailed_build_log)
    {
    }

    static bool GraphHasCtxNode(const OrtGraph* graph, const OrtApi& ort_api);

    bool ValidateEPCtxNode(const OrtGraph* graph) const;

    OrtStatus* GetEpContextFromGraph(const OrtGraph& graph);

    //! \brief Partition name (fused node name at build time) from the EPContext subgraph node.
    //! Populated by GetEpContextFromGraph(). Use for runtime cache path so it matches the name used when the cache was written.
    const std::string& GetPartitionName() const { return partition_name_; }

private:
    TensorrtRtxExecutionProvider& ep_;
    const OrtLogger& logger_;
    std::unique_ptr<nvinfer1::ICudaEngine>* trt_rtx_engine_;
    nvinfer1::IRuntime* trt_rtx_runtime_;
    std::string ep_context_model_path_;  //!< If using context model, it implies context model and engine cache is in the same directory
    std::string compute_capability_;
    bool weight_stripped_engine_refit_;
    std::string onnx_model_folder_path_;
    const void* onnx_model_bytestream_;
    size_t onnx_model_bytestream_size_;
    const void* onnx_external_data_bytestream_;
    size_t onnx_external_data_bytestream_size_;
    bool detailed_build_log_;
    std::string partition_name_;  //!< Set by GetEpContextFromGraph() from subgraph node's partition_name attribute
};  // EPContextNodeReader

}  // namespace trt_rtx_ep
