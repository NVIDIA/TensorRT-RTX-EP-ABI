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

#include "tensorrt_rtx_provider_factory.h"
#include "tensorrt_rtx_execution_provider_info.h"
#include "utils/ep_utils.h"
#include "utils/filesystem_utils.h"
#include "cuda_graph.h"
#include "nv_includes.h"

#include "onnxruntime_cxx_api.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

using HashValue = uint64_t;

// Import CUDA Graph types from trt_rtx_ep namespace
using trt_rtx_ep::CudaGraphAnnotation_t;
using trt_rtx_ep::CUDAGraph;
using trt_rtx_ep::kCudaGraphAnnotationDefault;
using trt_rtx_ep::kCudaGraphAnnotationSkip;

// All ORT types are already declared in onnxruntime_cxx_api.h (which we include above)

namespace trt_rtx_ep
{

// SubGraph_t and SubGraphCollection_t were removed from NvOnnxParser.h between
// TRT-RTX 1.5.0.97 and 1.5.0.99. Define them here for those builds and all 1.6.x+.
#if TRT_MAJOR_RTX >= 2 || TRT_MINOR_RTX >= 6 || (TRT_MINOR_RTX == 5 && TRT_BUILD_RTX >= 99)
using SubGraph_t = std::pair<std::vector<size_t>, bool>;
using SubGraphCollection_t = std::vector<SubGraph_t>;
#endif

constexpr size_t kTensorRTEngineHeaderSize = 64;

// Helper functions for engine header validation
std::string BinaryToHexString(const void* data, size_t size);
std::vector<uint8_t> HexStringToBinary(const std::string& hex);

//!
//! \brief Class to allocate memory for outputs with data-dependent shapes.
//!
//! The sizes of those are unknown so pre-allocation is not possible.
//!
class OutputAllocator : public nvinfer1::IOutputAllocator
{
public:
    OutputAllocator() = delete;
    OutputAllocator(OrtAllocator* allocator) : alloc_(allocator)
    {
        if (alloc_ == nullptr)
        {
            throw std::invalid_argument("OutputAllocator: allocator parameter cannot be null");
        }
    }

    void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
                                cudaStream_t stream) noexcept override;

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;

    void* getBuffer() const
    {
        return outputPtr;
    }

    const std::vector<int64_t>& getOutputShape() const
    {
        return output_shapes;
    }

    uint64_t getSize() const
    {
        return allocated_size;
    }

    ~OutputAllocator() override
    {
        if (alloc_ != nullptr)
        {
            alloc_->Free(alloc_, outputPtr);
        }
    }

private:
    OrtAllocator* alloc_;
    void* outputPtr{nullptr};
    uint64_t allocated_size = 0;
    std::vector<int64_t> output_shapes;
};

using DDSOutputAllocatorMap = std::unordered_map<std::string, std::unique_ptr<OutputAllocator>>;

class TensorrtRtxLogger : public nvinfer1::ILogger
{
    Severity verbosity_;
    mutable std::shared_mutex logger_mutex_;
    const OrtLogger* ort_default_logger_{nullptr};
    const OrtApi* ort_api_{nullptr};

public:
    explicit TensorrtRtxLogger(Severity verbosity = Severity::kWARNING)
        : verbosity_(verbosity)
    {
    }

    // Called by EP instances to register a currently-alive OrtLogger.
    void set_ort_logger(const OrtLogger* logger, const OrtApi* api)
    {
        std::unique_lock lock(logger_mutex_);
        ort_default_logger_ = logger;
        ort_api_ = api;
    }

    // Called by EP destructors. Clears the pointer only if it still matches,
    // so a newer EP's logger isn't accidentally removed by an older EP's teardown.
    void clear_ort_logger(const OrtLogger* logger)
    {
        std::unique_lock lock(logger_mutex_);
        if (ort_default_logger_ == logger)
        {
            ort_default_logger_ = nullptr;
            ort_api_ = nullptr;
        }
    }

    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= verbosity_)
        {
            time_t rawtime = std::time(0);
            struct tm stm;
#if defined(_MSC_VER)
            gmtime_s(&stm, &rawtime);
#else
            gmtime_r(&rawtime, &stm);
#endif
            char buf[256];
            strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", &stm);
            const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG"
                                  : severity == Severity::kERROR        ? "  ERROR"
                                  : severity == Severity::kWARNING      ? "WARNING"
                                  : severity == Severity::kINFO         ? "   INFO"
                                                                        : "UNKNOWN");
            OrtLoggingLevel ort_severity;
            if (severity <= Severity::kERROR)
            {
                ort_severity = ORT_LOGGING_LEVEL_ERROR;
            }
            else
            {
                ort_severity = ORT_LOGGING_LEVEL_WARNING;
            }

            std::string message = "[" + std::string(buf) + " " + std::string(sevstr) + "] " + std::string(msg);

            std::shared_lock lock(logger_mutex_);
            if (ort_api_ && ort_default_logger_)
            {
                try {
                    ort_api_->Logger_LogMessage(ort_default_logger_, ort_severity,
                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                }
                catch (const Ort::Exception& ex) {
                    fprintf(stderr, "[TensorRT Logger] Ort::Exception: %s\n", ex.what());
                }
            }
            else
            {
                fprintf(stderr, "%s\n", message.c_str());
            }
        }
    }

    void set_level(Severity verbosity)
    {
        verbosity_ = verbosity;
    }

    Severity get_level() const
    {
        return verbosity_;
    }
};

TensorrtRtxLogger& GetTensorrtRtxLogger(bool verbose_log);

namespace tensorrt_ptr
{
//!
//! \brief Custom deleter that will dump the optimized runtime cache when the execution context is destructed.
//!
struct IExecutionContextDeleter
{
    IExecutionContextDeleter() = default;
    IExecutionContextDeleter(const std::string& runtime_cache_path, std::unique_ptr<nvinfer1::IRuntimeCache>&& runtime_cache,
                             const OrtLogger& logger, const OrtApi& ort_api)
        : runtime_cache_path_(runtime_cache_path), runtime_cache_(std::move(runtime_cache)), logger_(logger), ort_api_(ort_api)
    {
    }

    void operator()(nvinfer1::IExecutionContext* context)
    {
        if (context != nullptr)
        {
            if (!runtime_cache_path_.empty())
            {
                auto serialized_cache_data = std::unique_ptr<nvinfer1::IHostMemory>(runtime_cache_->serialize());
                utils::WriteFile(runtime_cache_path_, serialized_cache_data->data(), serialized_cache_data->size(), logger_, ort_api_);
            }
            delete context;
        }
    }

private:
    std::string runtime_cache_path_;
    std::unique_ptr<nvinfer1::IRuntimeCache> runtime_cache_;
    const OrtLogger& logger_;
    const OrtApi& ort_api_;
};

struct TensorrtInferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete obj;
        }
    }
};

template <typename T>
using unique_pointer = std::unique_ptr<T, TensorrtInferDeleter>;
using unique_pointer_exec_ctx = std::unique_ptr<nvinfer1::IExecutionContext, IExecutionContextDeleter>;
}  // namespace tensorrt_ptr

//!
//! \brief This map saves the dimension range of the shape of the shape tensor or execution tensor.
//!
//! tensor name -> ( dimension -> [min, max, opt] )
//!
using ShapeRangesMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>>;

//!
//! \brief Container for tensor data and their shape.
//!
struct TensorParams
{
    const void* data{nullptr};
    nvinfer1::Dims dims;

    TensorParams() = default;

    TensorParams(const void* data_ptr, const std::vector<int64_t>& shape)
    {
        // Initialize data and dims from the Ort::ConstValue
        data = data_ptr;

        dims.nbDims = static_cast<int32_t>(shape.size());
        for (int i = 0; i < dims.nbDims; ++i)
        {
            dims.d[i] = static_cast<int32_t>(shape[i]);
        }
    }

    TensorParams(const void* data_ptr, nvinfer1::Dims& shape)
    {
        // Initialize data and dims from the Ort::ConstValue
        data = data_ptr;

        dims = shape;
    }

    bool operator!=(const TensorParams& other) const
    {
        if (data != other.data || dims.nbDims != other.dims.nbDims)
        {
            return true;
        }

        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] != other.dims.d[i])
            {
                return true;
            }
        }
        return false;
    }
};

//!
//! \brief Data structure to hold user weights when ModelProtos are serialized with external data.
//!
class TensorrtUserWeights
{
public:
    TensorrtUserWeights(const std::string& name, const std::string& data)
        : name_(name), data_cpy_(data), data_(nullptr), size_(0)
    {
    }

    TensorrtUserWeights(const std::string& name, const void* data, size_t size)
        : name_(name), data_cpy_(), data_(data), size_(size)
    {
    }

    const char* Name() const
    {
        return name_.c_str();
    }

    const void* Data() const
    {
        if (!data_cpy_.empty())
        {
            return data_cpy_.data();
        }
        return data_;
    }

    int64_t Size() const
    {
        if (!data_cpy_.empty())
        {
            return static_cast<int64_t>(data_cpy_.size());
        }
        return static_cast<int64_t>(size_);
    }

private:
    std::string name_{};
    std::string data_cpy_{};
    void const* data_;
    size_t size_;
};

//!
//! \brief Information to construct kernel function state.
//!
struct TensorrtRtxComputeState
{
    std::string fused_node_name;
    nvinfer1::IBuilder* builder;
    std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
    tensorrt_ptr::unique_pointer_exec_ctx* context = nullptr;
    std::unique_ptr<nvinfer1::INetworkDefinition>* network = nullptr;
    std::vector<std::unordered_map<std::string, size_t>> input_info;
    std::vector<std::unordered_map<std::string, size_t>> output_info;
    std::unordered_map<std::string, std::unordered_map<size_t, std::vector<std::vector<int64_t>>>> input_shape_ranges;
    std::mutex* tensorrt_mu_ptr = nullptr;
    bool engine_cache_enable = false;
    std::string engine_cache_path;
    nvinfer1::IRuntime* runtime = nullptr;
    std::vector<nvinfer1::IOptimizationProfile*> profiles;
    bool engine_decryption_enable = false;
    int (*engine_decryption)(const char*, char*, size_t*) = nullptr;
    int (*engine_encryption)(const char*, char*, size_t) = nullptr;
    bool detailed_build_log = false;
    bool sparsity_enable = false;
    int device_id = 0;
    int auxiliary_streams = -1;
    bool cuda_graph_enable = 0;
    bool multi_profile_enable = false;
    int trt_profile_index_ = 0;
    bool is_dynamic_shape = false;
    std::string cache_prefix;
    std::string cache_suffix;
    // runtime parameters
    std::vector<AllocatorUniquePtr<void>> scratch_buffers;
    std::vector<TensorParams> input_tensors;
    std::vector<TensorParams> output_tensors;
    bool is_first_run = true;              //!< Indicates if this is the first run of the engine
    bool skip_io_binding_allowed = false;  //!< Indicates if input/output binding can be skipped
};

//!
//! \brief Minimum information to construct kernel function state for direct engine load code path.
//!
struct TensorrtRtxEpContextNodeComputeState
{
    uint32_t device_id;
    std::string fused_node_name;
    std::unique_ptr<nvinfer1::ICudaEngine>* engine = nullptr;
    tensorrt_ptr::unique_pointer_exec_ctx* context = nullptr;
    std::vector<std::unordered_map<std::string, size_t>> input_info;
    std::vector<std::unordered_map<std::string, size_t>> output_info;
    std::mutex* tensorrt_mu_ptr = nullptr;
    bool is_dynamic_shape = false;
    // runtime parameters
    std::vector<AllocatorUniquePtr<void>> scratch_buffers;
    std::vector<TensorParams> input_tensors;
    std::vector<TensorParams> output_tensors;
    bool is_first_run = true;              //!< Indicates if this is the first run of the engine
    bool skip_io_binding_allowed = false;  //!< Indicates if input/output binding can be skipped
};

//!
//! \brief Plugin TensorRT RTX Execution Provider implementing OrtEp.
//!
//! This is a boilerplate that you can customize for your own execution provider.
//!
struct TensorrtRtxExecutionProvider : public OrtEp, public ApiPtrs
{
    // Constructor - Initialize your EP here
    TensorrtRtxExecutionProvider(TensorrtRtxExecutionProviderFactory& factory,
                                 const std::string& name,
                                 const OrtSessionOptions& session_options,
                                 const OrtLogger& logger);

    ~TensorrtRtxExecutionProvider();

    TensorrtRtxExecutionProviderFactory& factory_;
    std::string name_;
    const OrtSessionOptions& session_options_;
    const OrtLogger& logger_;
    bool external_stream_ = false;
    cudaStream_t stream_ = nullptr;

    //!< Call cudaStreamSynchronize() after TRT enqueueV3().
    mutable bool sync_stream_after_enqueue_ = true;

    //!< The OrtAllocator object will be obtained during EP compute time.
    OrtAllocator* alloc_ = nullptr;

    std::unordered_map<std::string, std::unique_ptr<TensorrtRtxComputeState>> compute_states_;
    std::unordered_map<std::string, std::unique_ptr<TensorrtRtxEpContextNodeComputeState>> compute_states_for_ep_context_;
    std::unordered_map<std::string, DDSOutputAllocatorMap> dds_output_allocator_maps_;

    // Refit engine with new weights from ONNX model
    OrtStatus* RefitEngineImpl(_In_ std::string onnx_model_filename,
                               _In_ std::string onnx_model_folder_path,
                               _In_ bool path_check,
                               _In_ const void* onnx_model_bytestream,
                               _In_ size_t onnx_model_bytestream_size,
                               _In_ const void* onnx_external_data_bytestream,
                               _In_ size_t onnx_external_data_bytestream_size,
                               _In_ nvinfer1::ICudaEngine* trt_engine,
                               _In_ bool detailed_build_log) noexcept;

private:
    // ========================================
    // Required ORT EP Interface implementations
    // ========================================

    static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
        _In_ OrtEp* this_ptr,
        _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept;

    static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                     const OrtGraph* graph,
                                                     OrtEpGraphSupportInfo* graph_support_info) noexcept;

    static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr,
                                               _In_ const OrtGraph** graphs,
                                               _In_ const OrtNode** fused_nodes,
                                               _In_ size_t count,
                                               _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                               _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

    static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                         OrtNodeComputeInfo** node_compute_infos,
                                                         size_t num_node_compute_infos) noexcept;

    static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                                 _In_ const OrtMemoryDevice* memory_device,
                                                                 _Outptr_ OrtSyncStreamImpl** stream) noexcept;

    static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(_In_ OrtEp* this_ptr,
                                                                          _In_ const OrtGraph* graph) noexcept;

    static OrtStatus* ORT_API_CALL OnRunStartImpl(_In_ OrtEp* this_ptr,
                                                  _In_ const OrtRunOptions* run_options) noexcept;

    static OrtStatus* ORT_API_CALL OnRunEndImpl(_In_ OrtEp* this_ptr,
                                                _In_ const OrtRunOptions* run_options,
                                                _In_ bool sync_stream) noexcept;

    mutable TensorrtRtxExecutionProviderInfo info_;
    int device_id_ = 0;

    int max_partition_iterations_ = 1000;
    size_t min_subgraph_size_ = 1;
    size_t max_workspace_size_ = 0;
    size_t max_shared_mem_size_ = 0;
    bool force_sequential_engine_build_ = false;
    bool dump_subgraphs_ = false;
    bool engine_cache_enable_ = false;
    bool weight_stripped_engine_enable_ = false;
    std::string onnx_model_folder_path_;
    const void* onnx_model_bytestream_;
    size_t onnx_model_bytestream_size_;
    const void* onnx_external_data_bytestream_ = nullptr;
    size_t onnx_external_data_bytestream_size_ = 0;
    bool sparsity_enable_ = false;
    int auxiliary_streams_ = -1;
    std::string cache_path_, engine_decryption_lib_path_;
    std::unique_ptr<nvinfer1::IRuntime> trt_rtx_runtime_ = nullptr;
    std::mutex tensorrt_rtx_mu_;

    std::string compute_capability_;
    size_t max_ctx_mem_size_ = 0;
    mutable char model_path_[4096] = {};  //!< Reserved for max path length
    bool engine_decryption_enable_ = false;
    int (*engine_decryption_)(const char*, char*, size_t*) = nullptr;
    int (*engine_encryption_)(const char*, char*, size_t) = nullptr;
    bool detailed_build_log_ = false;
    bool cuda_graph_enable_ = false;
    bool multi_profile_enable_ = false;
    int trt_profile_index_ = 0;
    bool dump_ep_context_model_ = false;
    int ep_context_embed_mode_ = 0;
    std::string engine_cache_prefix_;
    std::string op_types_to_exclude_;
    std::filesystem::path runtime_cache_;
    std::string cache_prefix_;

    // Following maps that hold TRT objects will be accessible by different threads if ORT is using multithreading.
    // In general, TensorRT objects are not thread safe; accesses to an object from different threads must be serialized by the client.
    // But there are still some thread safe operations, please see here https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    // For those non thread safe operations, TRT EP uses (1) lock_guard or (2) PerThreadContext to make sure synchronization.
    std::unordered_map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> engines_;
    std::unordered_map<std::string, tensorrt_ptr::unique_pointer_exec_ctx> contexts_;
    std::unordered_map<std::string, std::unique_ptr<nvinfer1::IBuilder>> builders_;
    std::unordered_map<std::string, std::unique_ptr<nvinfer1::INetworkDefinition>> networks_;

    std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> input_info_;
    std::unordered_map<std::string, std::vector<std::unordered_map<std::string, size_t>>> output_info_;
    std::unordered_map<std::string, ShapeRangesMap> input_shape_ranges_;  //!< The profile shape ranges that the engine is built with
    std::unordered_map<std::string, std::vector<nvinfer1::IOptimizationProfile*>> profiles_;
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_min_shapes_;
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_max_shapes_;
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>> profile_opt_shapes_;

    // Storage for engine headers (64 bytes) for compatibility validation
    // Maps fused_node_name -> hex-encoded engine header
    mutable std::unordered_map<std::string, std::string> engine_headers_;
    mutable std::string compatibility_info_cache_;

    // For create/dump EP context node model.

    // Cuda Graph stuff
    CUDAGraph cuda_graph_;
    // Map of graph id to regular_run_count_before_graph_capture
    std::unordered_map<CudaGraphAnnotation_t, int> graph_id_to_run_count_;
    bool is_graph_captured_ = false;
    int regular_run_count_before_graph_capture_ = 0;
    // Current graph annotation ID for this run
    CudaGraphAnnotation_t current_graph_annotation_id_ = 0;
    // There is chance (currently only happens in CUDA EP) that the second regular run allocates GPU memory for causes like:
    // (1) memory pattern is enabled. (2) arena allocation for stream.
    // Since no GPU memory allocation is allowed during graph capturing, we need at least two regular runs
    // to allocate enough memory in Arena before graph capturing.
    const int min_num_runs_before_cuda_graph_capture_ = 2;  //!< required min regular runs before graph capture for the necessary memory allocations.
    // https://github.com/NVIDIA/TensorRT/blob/main/samples/common/sampleInference.cpp#L1258-L1291 Based on the trtexec code

    // For create/dump EP context node model
    std::string ep_context_file_path_;
    std::string ctx_model_path_;
    std::string engine_cache_relative_path_to_context_model_dir_;

    std::unordered_set<std::string> control_flow_op_set_ = {"If", "Loop", "Scan"};

    // TensorRT RTX runtime and builder.
    std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    mutable std::unique_ptr<nvinfer1::IBuilder> builder_ = nullptr;

    // The format is as for TENSORRT_VERSION: (MAJOR * 100 + MINOR) * 100 + PATCH.
    int32_t trt_version_;
    int32_t cuda_version_;

    //!
    //! \brief Get a unique_lock object to control the concurrency behavior.
    //!
    //! Every API call not in the thread-safe operations (https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading)
    //! should be protected by a lock when invoked by multiple threads concurrently.
    //!
    std::unique_lock<std::mutex> GetApiLock() const;

    // Returns an OrtGraph subgraph view created using Graph_GetGraphView.
    Ort::Graph GetSubgraph(SubGraph_t graph_nodes_index, const Ort::ConstGraph& graph) const;

    bool AllNodesAssignedToSpecificEP(const OrtGraph* graph, const std::string& provider_type) const;

    SubGraphCollection_t GetSupportedList(SubGraphCollection_t nodes_vector_input, int iterations, const int max_iterations,
                                          const OrtGraph* graph, bool* early_termination) const;

    bool DetectTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const Ort::ConstGraph& graph,
                                   const HashValue& model_hash, bool remove_cycles = true) const;

    // Check the graph is the subgraph of control flow op.
    bool IsSubGraphOfControlFlowOp(const OrtGraph* graph) const;

    // Check whether all the nodes of subgraph are supported.
    bool IsSubGraphFullySupported(const OrtGraph* graph, SubGraphCollection_t supported_nodes_vector) const;

    OrtStatus* CreateNodeComputeInfoFromPrecompiledEngine(OrtEp* this_ptr, const OrtGraph* graph,
                                                          const OrtNode* fused_node,
                                                          std::unordered_map<std::string, size_t>& input_map,
                                                          std::unordered_map<std::string, size_t>& output_map,
                                                          OrtNodeComputeInfo** node_compute_info);

    OrtStatus* CreateNodeComputeInfoFromGraph(OrtEp* this_ptr, const OrtGraph* graph, const OrtNode* fused_node,
                                              std::unordered_map<std::string, size_t>& input_map,
                                              std::unordered_map<std::string, size_t>& output_map,
                                              OrtNodeComputeInfo** node_compute_info,
                                              OrtNode** ep_context_node);

    nvinfer1::IBuilder* GetBuilder(TensorrtRtxLogger& trt_logger) const;

public:
    // CUDA Graph related functions
    void HandleCudaGraphStart(cudaStream_t stream, bool require_io_binding, CudaGraphAnnotation_t cuda_graph_annotation_id,
                              bool& graph_replay_on_this_run, bool& should_start_capture);
    void SetCudaGraphStream(cudaStream_t stream)
    {
        cuda_graph_.SetStream(stream);
    }
    bool IsGraphCaptureEnabled() const
    {
        return cuda_graph_enable_;
    }
    bool IsGraphCaptureAllowed(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    CudaGraphAnnotation_t GetCudaGraphAnnotationId(const OrtRunOptions* run_options) const;
    void SetCurrentGraphAnnotationId(CudaGraphAnnotation_t cuda_graph_annotation_id);
    CudaGraphAnnotation_t GetCurrentGraphAnnotationId() const;
    void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
    bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    OrtStatus* ReplayGraph(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync_status_flag);
    void IncrementRegularRunCountBeforeGraphCapture(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void ResetWarmupRuns(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void DeleteCapturedGraph(CudaGraphAnnotation_t cuda_graph_annotation_id);

    const std::string& GetEpContextFilePath() const
    {
        return ep_context_file_path_;
    }
};

//!
//! \brief OrtNodeComputeInfo that represents the computation function for a compiled subgraph.
//!
//! This is created during the Compile phase and invoked during inference.
//!
struct TensorRtRtxEpNodeComputeInfo : OrtNodeComputeInfo
{
    explicit TensorRtRtxEpNodeComputeInfo(TensorrtRtxExecutionProvider& ep);

    // OrtNodeComputeInfo Interface implementations.
    static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state);

    static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                               void* compute_state,
                                               OrtKernelContext* kernel_context);

    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr,
                                              void* compute_state);

    TensorrtRtxExecutionProvider& ep;
};

//!
//! \brief Optional: OrtNodeComputeInfo for EP Context nodes (precompiled models).
//!
//! Only implement this if you want to support loading precompiled models.
//! You can delete this struct if not needed.
//!
struct TensorRtRtxEpContextNodeComputeInfo : OrtNodeComputeInfo
{
    explicit TensorRtRtxEpContextNodeComputeInfo(TensorrtRtxExecutionProvider& ep);

    static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state);

    static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                               void* compute_state,
                                               OrtKernelContext* kernel_context);

    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr,
                                              void* compute_state);

    TensorrtRtxExecutionProvider& ep;
};

}  // namespace trt_rtx_ep
