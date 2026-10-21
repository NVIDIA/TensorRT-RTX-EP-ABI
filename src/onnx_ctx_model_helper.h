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

#include "utils/path_validation.h"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ep_utils.h"
#include "weightless_refit.h"

namespace trt_rtx_ep
{

class FileStreamWriter;

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
    EPContextNodeHelper(TensorrtRtxExecutionProvider& ep, const OrtGraph* graph, const OrtNode* fused_node);

    ~EPContextNodeHelper();

    //! Configures the serialized engine artifact. External file output streams directly to its final cache;
    //! embedded and callback output use a temporary file until their bytes are requested, or are built in memory
    //! when the temporary file cannot be created (e.g. in sandboxed processes). A final-cache file is retained only
    //! after CreateEPContextNode() has successfully created the node that references it.
    OrtStatus* InitializeEngineOutput(const std::filesystem::path& cache_root, const std::string& cache_prefix,
                                      bool create_ep_context_node, int64_t embed_mode);
    //! True when the engine must be written through GetEngineStreamWriter(); false when it must be built in memory
    //! and handed over with SetSerializedEngine().
    bool StreamsEngineOutput() const noexcept;
    nvinfer1::IStreamWriter& GetEngineStreamWriter();
    void SetSerializedEngine(std::unique_ptr<nvinfer1::IHostMemory> serialized_engine) noexcept;
    OrtStatus* FinalizeEngineOutput();
    size_t GetEngineSize() const noexcept;
    //! Reads only the fixed-size compatibility header unless the complete engine has already been materialized.
    const std::vector<char>& GetEngineHeader();
    //! Materializes the complete serialized engine on first use and reuses that buffer for subsequent consumers.
    std::span<const char> GetEngineData();

    //! \param refit_records Weightless refit records to persist. May be pruned in place: a record whose
    //!        source initializer was not preserved as a node input is dropped rather than failing the
    //!        whole export when `refit_table_filtered_against_engine` is false (see that param).
    //! \param persist_refit_table Write the ep_refit_table attribute even when `refit_records` is empty.
    //!        Set for weight-stripped compiles: the attribute marks the node as weightless for the load
    //!        path, and an engine with no refittable weights legitimately has an empty table.
    //! \param refit_table_filtered_against_engine True when the caller already filtered `refit_records`
    //!        against the live built engine's IRefitter (i.e. a live engine was available -- not
    //!        compile-only mode). In that case every remaining record is expected to be genuinely
    //!        needed, so a record whose source was not preserved indicates a real bug and export is
    //!        aborted. When false (compile-only: no live engine to filter against), `refit_records` is
    //!        still the parser's unfiltered superset, so a missing source may simply belong to a weight
    //!        the builder folded away; that record is dropped instead of aborting the export.
    OrtStatus* CreateEPContextNode(const std::string& compute_capability, const std::filesystem::path& onnx_model_path,
                                   std::vector<WeightlessRefitRecord>& refit_records, bool persist_refit_table,
                                   bool refit_table_filtered_against_engine, OrtNode** ep_context_node);

private:
    TensorrtRtxExecutionProvider& ep_;
    const OrtGraph* graph_ = nullptr;
    const OrtNode* fused_node_ = nullptr;
    int64_t embed_mode_ = 0;
    std::filesystem::path engine_cache_path_;
    std::filesystem::path serialized_engine_path_;
    // Keep the engine only after the EPContext node that references it has been created. This ensures a failed
    // build or node-creation path does not leave a partial or orphaned external cache file behind.
    bool preserve_serialized_engine_on_destroy_ = false;
    size_t serialized_engine_size_ = 0;
    std::unique_ptr<FileStreamWriter> engine_stream_writer_;
    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine_;
    std::vector<char> engine_header_;
    std::vector<char> engine_data_;
};

//!
//! \brief Class to read an OrtGraph that contains an EPContext node and get the engine binary accordingly.
//!
class EPContextNodeReader : public ApiPtrs
{
public:
    EPContextNodeReader(TensorrtRtxExecutionProvider& ep, const OrtLogger& logger,
                        std::unique_ptr<nvinfer1::ICudaEngine>* trt_rtx_engine, nvinfer1::IRuntime* trt_rtx_runtime,
                        std::string ep_context_model_path, std::string compute_capability,
                        bool weight_stripped_engine_refit, std::string onnx_model_folder_path,
                        const void* onnx_model_bytestream, size_t onnx_model_bytestream_size,
                        const void* onnx_external_data_bytestream, size_t onnx_external_data_bytestream_size,
                        bool detailed_build_log, bool deferred_weights_loading)
        : ApiPtrs{static_cast<const ApiPtrs&>(ep)}
        , ep_(ep)
        , logger_(logger)
        , trt_rtx_engine_(trt_rtx_engine)
        , trt_rtx_runtime_(trt_rtx_runtime)
        , ep_context_model_path_(ep_context_model_path)
        , compute_capability_(compute_capability)
        , weight_stripped_engine_refit_(weight_stripped_engine_refit)
        , onnx_model_folder_path_(onnx_model_folder_path)
        , onnx_model_bytestream_(onnx_model_bytestream)
        , onnx_model_bytestream_size_(onnx_model_bytestream_size)
        , onnx_external_data_bytestream_(onnx_external_data_bytestream)
        , onnx_external_data_bytestream_size_(onnx_external_data_bytestream_size)
        , detailed_build_log_(detailed_build_log)
        , deferred_weights_loading_(deferred_weights_loading)
    {
    }

    static bool GraphHasCtxNode(const OrtGraph* graph, const OrtApi& ort_api);

    bool ValidateEPCtxNode(const OrtGraph* graph) const;

    OrtStatus* GetEpContextFromGraph(const OrtGraph& graph);

    //! Complete a deferred external-engine load after execution-context creation.
    OrtStatus* LoadDeferredWeights(cudaStream_t stream);

    //! \brief Partition name (fused node name at build time) from the EPContext subgraph node.
    //! Populated by GetEpContextFromGraph(). Use for runtime cache path so it matches the name used when the cache was
    //! written.
    const std::string& GetPartitionName() const
    {
        return partition_name_;
    }

private:
    //! \brief If `node` carries an EP_REFIT_TABLE attribute, replay it via
    //!        TensorrtRtxExecutionProvider::WeightlessRefitEngineImpl (no original ONNX needed) and
    //!        set `handled` to true. If the attribute is absent, sets `handled` to false so the
    //!        caller falls back to the legacy RefitEngineImpl path unchanged.
    OrtStatus* TryWeightlessRefit(const Ort::ConstNode& node, bool& handled);

    TensorrtRtxExecutionProvider& ep_;
    const OrtLogger& logger_;
    std::unique_ptr<nvinfer1::ICudaEngine>* trt_rtx_engine_;
    nvinfer1::IRuntime* trt_rtx_runtime_;
    std::string ep_context_model_path_;  //!< If using context model, it implies context model and engine cache is in
                                         //!< the same directory
    std::string compute_capability_;
    bool weight_stripped_engine_refit_;
    std::string onnx_model_folder_path_;
    const void* onnx_model_bytestream_;
    size_t onnx_model_bytestream_size_;
    const void* onnx_external_data_bytestream_;
    size_t onnx_external_data_bytestream_size_;
    bool detailed_build_log_;
    bool deferred_weights_loading_;
    std::unique_ptr<nvinfer1::IStreamReaderV2> engine_stream_reader_;
    std::string partition_name_;  //!< Set by GetEpContextFromGraph() from subgraph node's partition_name attribute
};  // EPContextNodeReader

}  // namespace trt_rtx_ep
