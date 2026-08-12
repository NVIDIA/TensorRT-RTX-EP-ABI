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

#include "tensorrt_rtx_execution_provider.h"

#include "onnx_ctx_model_helper.h"
#include "tensorrt_rtx_allocator.h"
#include "tensorrt_rtx_execution_provider_stream_support.h"
#include "tensorrt_rtx_provider_factory.h"
#include "tensorrt_rtx_provider_options.h"

#include "utils/cuda/cuda_call.h"
#include "utils/cuda/cuda_context.h"
#include "utils/ep_utils.h"
#include "utils/filesystem_utils.h"
#include "utils/ort_api_init.h"
#include "utils/path_string.h"
#include "utils/provider_options.h"

#include "gpu_sync_allocator.h"
#include "proto_node_id_utils.h"
#include "trt_proto_preprocessing.h"

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "tensorrt_rtx_execution_provider_utils.h"

#include "utils/ort_graph_to_proto.h"
#include "utils/ort_model_dump.h"

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) LoadLibrary(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#define CLOSELIB(lib) FreeLibrary((HMODULE)(lib))
#else
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#define CLOSELIB(lib) dlclose(lib)
#endif

namespace trt_rtx_ep
{

const OrtApi* g_ort_api = nullptr;
const OrtEpApi* g_ep_api = nullptr;
const OrtModelEditorApi* g_model_editor_api = nullptr;
const OrtLogger* g_logger = nullptr;

namespace
{
//!
//! \brief Resolves a model-derived runtime-cache filename without making caching mandatory.
//!
//! \param runtime_cache_dir Configured runtime-cache directory.
//! \param raw_name Model-derived node or partition name.
//! \param logger ORT logger used when the name cannot be used safely.
//! \param ort_api ORT API used for best-effort warning logging.
//! \return The complete cache path, or std::nullopt when caching must be disabled for this node.
//!
std::optional<std::filesystem::path> ResolveRuntimeCacheFile(const std::filesystem::path& runtime_cache_dir,
                                                             const std::string& raw_name, const OrtLogger& logger,
                                                             const OrtApi& ort_api)
{
    auto cache_name = SanitizeCacheFilename(raw_name);
    if (cache_name)
    {
        return runtime_cache_dir / *cache_name;
    }

    // Runtime caching is optional. Consume any logging status without throwing through CompileImpl noexcept.
    static_cast<void>(Ort::Status{ort_api.Logger_LogMessage(
        &logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        "[NvTensorRTRTX EP] Runtime cache disabled for this node: model-supplied node/partition name is unsafe for "
        "use as a cache filename (absolute path or parent-directory traversal).",
        ORT_FILE, __LINE__, __FUNCTION__)});
    return std::nullopt;
}

//!
//! \brief Creates, restores, and attaches an optional TensorRT RTX runtime cache.
//!
//! A missing cache file is treated as an empty cache. Failures that prevent safe cache use are logged and disable
//! caching for the execution context without failing compilation.
//!
//! \param runtime_config Runtime configuration that creates and receives the cache.
//! \param runtime_cache_file File from which an existing cache is restored.
//! \param logger ORT logger used for cache setup diagnostics.
//! \param ort_api ORT API used for logging and file-error status creation.
//! \return The attached cache with ownership retained by the caller, or nullptr when caching is disabled.
//!
std::unique_ptr<nvinfer1::IRuntimeCache> CreateAndAttachRuntimeCache(nvinfer1::IRuntimeConfig& runtime_config,
                                                                     const std::filesystem::path& runtime_cache_file,
                                                                     const OrtLogger& logger, const OrtApi& ort_api)
{
    auto runtime_cache = std::unique_ptr<nvinfer1::IRuntimeCache>(runtime_config.createRuntimeCache());
    if (runtime_cache == nullptr)
    {
        static_cast<void>(Ort::Status{ort_api.Logger_LogMessage(
            &logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            "[NvTensorRTRTX EP] Runtime caching is disabled: createRuntimeCache returned null.", ORT_FILE, __LINE__,
            __FUNCTION__)});
        return nullptr;
    }

    std::vector<char> cache_data;
    try
    {
        cache_data = utils::ReadFile(runtime_cache_file, ort_api);
    }
    catch (const std::exception& ex)
    {
        char message[2048];
        std::snprintf(message, sizeof(message),
                      "[NvTensorRTRTX EP] Runtime caching is disabled because the configured cache could not be "
                      "read: %s",
                      ex.what());
        static_cast<void>(Ort::Status{ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                                message, ORT_FILE, __LINE__, __FUNCTION__)});
        return nullptr;
    }
    catch (...)
    {
        static_cast<void>(Ort::Status{ort_api.Logger_LogMessage(
            &logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            "[NvTensorRTRTX EP] Runtime caching is disabled because the configured cache could not be read: unknown "
            "error.",
            ORT_FILE, __LINE__, __FUNCTION__)});
        return nullptr;
    }

    if (!cache_data.empty() && !runtime_cache->deserialize(cache_data.data(), cache_data.size()))
    {
        const std::string message = "[NvTensorRTRTX EP] TensorRT RTX failed to deserialize runtime cache '" +
                                    PathToUTF8String(runtime_cache_file.native()) + "'; it will be overwritten.";
        static_cast<void>(Ort::Status{ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__)});

        // TRT-RTX 1.6 leaves a rejected cache unchanged, but 1.5 does not make that guarantee.
        if (!runtime_cache->reset())
        {
            static_cast<void>(Ort::Status{ort_api.Logger_LogMessage(
                &logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                "[NvTensorRTRTX EP] Runtime caching is disabled: failed to reset the rejected cache.", ORT_FILE,
                __LINE__, __FUNCTION__)});
            return nullptr;
        }
    }

    if (!runtime_config.setRuntimeCache(*runtime_cache))
    {
        static_cast<void>(Ort::Status{
            ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                      "[NvTensorRTRTX EP] Runtime caching is disabled: setRuntimeCache returned false.",
                                      ORT_FILE, __LINE__, __FUNCTION__)});
        return nullptr;
    }
    return runtime_cache;
}

//!
//! \brief Creates an execution context whose deleter owns its runtime configuration and cache.
//!
//! \param engine Engine used to create the execution context.
//! \param runtime_cache_file File to which the final optimized cache is persisted.
//! \param runtime_cache Cache attached to runtime_config, if caching is enabled.
//! \param runtime_config Runtime configuration referenced non-owningly by the execution context.
//! \param ort_api ORT API used if final cache persistence fails.
//! \return An execution-context pointer carrying the required configuration and cache ownership.
//!
tensorrt_ptr::unique_pointer_exec_ctx
CreateOwnedExecutionContext(nvinfer1::ICudaEngine& engine, const std::filesystem::path& runtime_cache_file,
                            std::unique_ptr<nvinfer1::IRuntimeCache> runtime_cache,
                            std::unique_ptr<nvinfer1::IRuntimeConfig> runtime_config, const OrtApi& ort_api)
{
    auto* runtime_config_ptr = runtime_config.get();
    auto deleter = tensorrt_ptr::IExecutionContextDeleter(runtime_cache_file, std::move(runtime_cache),
                                                          std::move(runtime_config), ort_api);
    return {engine.createExecutionContext(runtime_config_ptr), std::move(deleter)};
}
}  // namespace

namespace tensorrt_ptr
{
IExecutionContextDeleter::IExecutionContextDeleter(const std::filesystem::path& runtime_cache_path,
                                                   std::unique_ptr<nvinfer1::IRuntimeCache>&& runtime_cache,
                                                   std::unique_ptr<nvinfer1::IRuntimeConfig>&& runtime_config,
                                                   const OrtApi& ort_api)
    : runtime_cache_path_(runtime_cache_path)
    , runtime_cache_(std::move(runtime_cache))
    , runtime_config_(std::move(runtime_config))
    , ort_api_(ort_api)
{
}

void IExecutionContextDeleter::operator()(nvinfer1::IExecutionContext* context) noexcept
{
    if (context == nullptr)
    {
        return;
    }

    // IExecutionContext stores a non-owning IRuntimeConfig pointer. Keep both dependencies alive while the context
    // synchronizes its stream and releases its TensorRT/Myelin resources, then serialize the still-owned cache.
    delete context;
    SaveCache();
}

void IExecutionContextDeleter::SaveCache() noexcept
{
    if (runtime_cache_path_.empty() || runtime_cache_ == nullptr)
    {
        return;
    }

    try
    {
        auto serialized_cache_data = std::unique_ptr<nvinfer1::IHostMemory>(runtime_cache_->serialize());
        if (serialized_cache_data == nullptr)
        {
            std::fprintf(stderr,
                         "[NvTensorRTRTX EP] Runtime cache serialization returned null; cache save was skipped.\n");
            return;
        }

        const void* data = serialized_cache_data->data();
        const size_t size = serialized_cache_data->size();
        if (data == nullptr || size == 0)
        {
            std::fprintf(stderr,
                         "[NvTensorRTRTX EP] Runtime cache serialization returned empty or invalid data; cache save "
                         "was skipped.\n");
            return;
        }

        utils::WriteFile(runtime_cache_path_, data, size, ort_api_);
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "[NvTensorRTRTX EP] Failed to save runtime cache: %s\n", ex.what());
    }
    catch (...)
    {
        std::fprintf(stderr, "[NvTensorRTRTX EP] Failed to save runtime cache: unknown error\n");
    }
}
}  // namespace tensorrt_ptr

void CUDA_RETURN_IF_ERROR(cudaError_t res)
{
    if (res != cudaSuccess)
    {
        THROW("CUDA Error: ", cudaGetErrorString(res), " (error code: ", static_cast<int>(res), ")");
    }
}
//
// Get number of profile setting.
//
// profile_min_shapes/profile_max_shapes/profile_opt_shapes may contain multiple profile settings.
// Note: TRT EP currently only supports one profile setting.
//
// {
//   tensor_a: [[dim_0_value_0, dim_1_value_1, dim_2_value_2]],
//   tensor_b: [[dim_0_value_3, dim_1_value_4, dim_2_value_5]]
// }
//
int GetNumProfiles(std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_shapes)
{
    int num_profile = 0;
    for (auto it = profile_shapes.begin(); it != profile_shapes.end(); it++)
    {
        num_profile = static_cast<int>(it->second.size());
        if (num_profile > 0)
        {
            break;
        }
    }
    return num_profile;
}

// Anonymous namespace for local helpers used during graph serialization/remap.
namespace
{
constexpr const char* kExternalMemAddrLocation = "_MEM_ADDR_";
std::atomic<uint64_t> s_subgraph_dump_counter{0};

int64_t ResolveWeightStreamingBudget(nvinfer1::ICudaEngine& trt_engine,
                                     const TensorrtRtxWeightStreamingBudget& requested_budget)
{
    switch (requested_budget.mode)
    {
    case TensorrtRtxWeightStreamingBudgetMode::Automatic:
        return trt_engine.getWeightStreamingAutomaticBudget();
    case TensorrtRtxWeightStreamingBudgetMode::MinimumVram:
        // Provider option value 1 means minimum-VRAM mode. TRT represents that
        // mode as budget 0: stream as much weight data as possible, while still
        // allowing TRT to keep any required small state in GPU memory.
        return 0;
    case TensorrtRtxWeightStreamingBudgetMode::Bytes:
        return requested_budget.bytes;
    case TensorrtRtxWeightStreamingBudgetMode::Percent:
        return static_cast<int64_t>((requested_budget.percent / 100.0) * trt_engine.getStreamableWeightsSize());
    case TensorrtRtxWeightStreamingBudgetMode::Disabled:
        return 0;
    }

    return 0;
}

OrtStatus* ApplyWeightStreamingBudget(nvinfer1::ICudaEngine& trt_engine,
                                      const TensorrtRtxWeightStreamingBudget& requested_budget,
                                      const std::string& engine_name, const OrtLogger& logger, const OrtApi& ort_api)
{
    if (!requested_budget.IsEnabled())
    {
        return nullptr;
    }

    const int64_t resolved_budget = ResolveWeightStreamingBudget(trt_engine, requested_budget);
    if (!trt_engine.setWeightStreamingBudgetV2(resolved_budget))
    {
        const int64_t streamable_size = trt_engine.getStreamableWeightsSize();
        const std::string message =
            "[NvTensorRTRTX EP] Failed to set nv_weight_streaming_budget=" + requested_budget.requested_value +
            " (resolved to " + std::to_string(resolved_budget) + " bytes) for engine '" + engine_name +
            "'. Streamable weights size=" + std::to_string(streamable_size) + " bytes.";
        return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    const int64_t streamable_size = trt_engine.getStreamableWeightsSize();
    const int64_t actual_budget = trt_engine.getWeightStreamingBudgetV2();
    const int64_t scratch_bytes = trt_engine.getWeightStreamingScratchMemorySize();
    const std::string message =
        "[NvTensorRTRTX EP] Weight streaming budget applied for engine '" + engine_name +
        "': requested=" + requested_budget.requested_value + ", resolved=" + std::to_string(resolved_budget) +
        ", actual=" + std::to_string(actual_budget) + ", streamable_weights_size=" + std::to_string(streamable_size) +
        ", scratch_memory_size=" + std::to_string(scratch_bytes) + ".";
    RETURN_IF_ERROR(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, message.c_str(),
                                              ORT_FILE, __LINE__, __FUNCTION__));

    return nullptr;
}

// Helper function to detect cycles in graph using DFS
bool FindCycleHelper(size_t i, const std::list<size_t>* adjacency_map, bool visited[], bool* st,
                     std::vector<size_t>& cycles)
{
    if (!visited[i])
    {
        visited[i] = true;
        st[i] = true;
        for (auto iter = adjacency_map[i].begin(); iter != adjacency_map[i].end(); ++iter)
        {
            if (!visited[*iter] && FindCycleHelper(*iter, adjacency_map, visited, st, cycles))
            {
                cycles.push_back(*iter);
                return true;
            }
            else if (st[*iter])
            {
                cycles.push_back(*iter);
                return true;
            }
        }
    }
    st[i] = false;
    return false;
}

// OrtGraphToProto stores initializer payloads as external references, so we
// encode the original memory address in the TensorProto offset field.
OrtEpUtils::HandleInitializerDataFunc CreateInitializerDataHandler()
{
    return [](const OrtValueInfo* value_info, const void* data, size_t bytes, bool& is_external, std::string& location,
              int64_t& offset) -> Ort::Status
    {
        (void)value_info;
        (void)bytes;
        offset = reinterpret_cast<int64_t>(data);
        location = kExternalMemAddrLocation;
        is_external = true;
        return Ort::Status{nullptr};
    };
}

}  // namespace

void* OutputAllocator::reallocateOutputAsync(char const* /*tensorName*/, void* /*currentMemory*/, uint64_t size,
                                             uint64_t alignment, cudaStream_t /*stream*/) noexcept
{
    // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
    // even for empty tensors, so allocate a dummy byte.
    size = (std::max)(size, static_cast<uint64_t>(1));

    // TensorRT requires the returned buffer to satisfy the requested alignment (e.g. 512 bytes for
    // data-dependent-shape outputs such as NonZero). The ORT allocator does not guarantee this alignment,
    // so over-allocate by (alignment - 1) and return a pointer rounded up to the alignment boundary.
    // The original base pointer (outputPtrBase) is kept for Free(); the aligned pointer (outputPtr) is
    // what is handed to TensorRT and exposed via getBuffer().
    if (alignment == 0)
    {
        alignment = 1;
    }
    uint64_t const required_size = size + (alignment - 1);

    if (required_size > allocated_size)
    {
        alloc_->Free(alloc_, outputPtrBase);
        outputPtrBase = nullptr;
        outputPtr = nullptr;
        allocated_size = 0;
        outputPtrBase = alloc_->Alloc(alloc_, required_size);
        if (outputPtrBase)
        {
            allocated_size = required_size;
        }
    }

    // Compute the aligned pointer from the (possibly reused) base allocation.
    if (outputPtrBase != nullptr)
    {
        uintptr_t const base = reinterpret_cast<uintptr_t>(outputPtrBase);
        uintptr_t const aligned = ((base + (alignment - 1)) / alignment) * alignment;
        outputPtr = reinterpret_cast<void*>(aligned);
    }
    else
    {
        // if allocation fails, returns nullptr.
        outputPtr = nullptr;
    }
    return outputPtr;
}

void OutputAllocator::notifyShape(char const* /*tensorName*/, nvinfer1::Dims const& dims) noexcept
{
    output_shapes.clear();
    output_shapes.reserve(dims.nbDims);
    for (int i = 0; i < dims.nbDims; i++)
    {
        output_shapes.push_back(dims.d[i]);
    }
}

//
// Apply TensorRT optimization profile shapes from provider options.
//
// This function supports single/multiple profile(s).
// (Note: An optimization profile describes a range of dimensions for each network input)
//
bool ApplyProfileShapesFromProviderOptions(
    std::vector<nvinfer1::IOptimizationProfile*>& trt_profiles, nvinfer1::ITensor* input,
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
    std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes,
    ShapeRangesMap& input_explicit_shape_ranges, bool& cuda_graph_flag, const OrtLogger& logger, const OrtApi& ort_api)
{
    if (trt_profiles.size() == 0)
    {
        Ort::ThrowOnError(ort_api.Logger_LogMessage(
            &logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            "[NvTensorRTRTX EP] Number of optimization profiles should be greater than 0, but it's 0.", ORT_FILE,
            __LINE__, __FUNCTION__));
        return false;
    }

    const std::string& input_name = input->getName();
    if (profile_min_shapes.find(input_name) == profile_min_shapes.end())
    {
        return false;
    }

    if (input_explicit_shape_ranges.find(input_name) == input_explicit_shape_ranges.end())
    {
        std::unordered_map<size_t, std::vector<std::vector<int64_t>>> inner_map;
        input_explicit_shape_ranges[input_name] = inner_map;
    }

    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                "[NvTensorRTRTX EP] Begin to apply profile shapes ...", ORT_FILE,
                                                __LINE__, __FUNCTION__));

    std::string message = "[NvTensorRTRTX EP] Input tensor name is '" + input_name + "', number of profiles found is " +
                          std::to_string(trt_profiles.size());
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, message.c_str(),
                                                ORT_FILE, __LINE__, __FUNCTION__));

    for (size_t i = 0; i < trt_profiles.size(); i++)
    {
        nvinfer1::Dims dims = input->getDimensions();
        int nb_dims = dims.nbDims;

        auto trt_profile = trt_profiles[i];

        // Shape tensor
        if (input->isShapeTensor())
        {
            std::string shape_message;
            if (cuda_graph_flag)
            {
                shape_message = std::string("[NvTensorRTRTX EP] Shape tensor detected on input '") + input->getName() +
                                "'. Disabling CUDA Graph.";
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                            shape_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
                cuda_graph_flag = false;
            }
            int shape_size = nb_dims == 0 ? 1 : static_cast<int>(profile_min_shapes[input_name][i].size());
            std::vector<int64_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
            shape_message = "[NvTensorRTRTX EP] shape size of this shape tensor is " + std::to_string(shape_size);
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                        shape_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

            for (int j = 0; j < shape_size; j++)
            {
                auto min_value = profile_min_shapes[input_name][i][j];
                auto max_value = profile_max_shapes[input_name][i][j];
                auto opt_value = profile_opt_shapes[input_name][i][j];
                shapes_min[j] = static_cast<int64_t>(min_value);
                shapes_max[j] = static_cast<int64_t>(max_value);
                shapes_opt[j] = static_cast<int64_t>(opt_value);

                shape_message =
                    "[NvTensorRTRTX EP] shapes_min.d[" + std::to_string(j) + "] is " + std::to_string(shapes_min[j]);
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            shape_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

                if (input_explicit_shape_ranges[input_name].find(j) == input_explicit_shape_ranges[input_name].end())
                {
                    std::vector<std::vector<int64_t>> profile_vector(trt_profiles.size());
                    input_explicit_shape_ranges[input_name][j] = profile_vector;
                }
                input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(min_value);
                input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(max_value);
                input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(opt_value);
            }

            trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0],
                                          shape_size);
            trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0],
                                          shape_size);
            trt_profile->setShapeValuesV2(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0],
                                          shape_size);
        }
        // Execution tensor
        else
        {
            std::string exec_message;
            nvinfer1::Dims dims_min, dims_opt, dims_max;
            dims_min.nbDims = nb_dims;
            dims_max.nbDims = nb_dims;
            dims_opt.nbDims = nb_dims;

            exec_message =
                "[NvTensorRTRTX EP] number of dimension of this execution tensor is " + std::to_string(nb_dims);
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                        exec_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

            for (int j = 0; j < nb_dims; j++)
            {
                if (dims.d[j] == -1)
                {
                    auto min_value = profile_min_shapes[input_name][i][j];
                    auto max_value = profile_max_shapes[input_name][i][j];
                    auto opt_value = profile_opt_shapes[input_name][i][j];
                    dims_min.d[j] = static_cast<int32_t>(min_value);
                    dims_max.d[j] = static_cast<int32_t>(max_value);
                    dims_opt.d[j] = static_cast<int32_t>(opt_value);
                    exec_message =
                        "[NvTensorRTRTX EP] dims_min.d[" + std::to_string(j) + "] is " + std::to_string(dims_min.d[j]);
                    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                                exec_message.c_str(), ORT_FILE, __LINE__,
                                                                __FUNCTION__));
                    exec_message =
                        "[NvTensorRTRTX EP] dims_max.d[" + std::to_string(j) + "] is " + std::to_string(dims_max.d[j]);
                    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                                exec_message.c_str(), ORT_FILE, __LINE__,
                                                                __FUNCTION__));
                    exec_message =
                        "[NvTensorRTRTX EP] dims_opt.d[" + std::to_string(j) + "] is " + std::to_string(dims_opt.d[j]);
                    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                                exec_message.c_str(), ORT_FILE, __LINE__,
                                                                __FUNCTION__));

                    if (input_explicit_shape_ranges[input_name].find(j) ==
                        input_explicit_shape_ranges[input_name].end())
                    {
                        std::vector<std::vector<int64_t>> profile_vector(trt_profiles.size());
                        input_explicit_shape_ranges[input_name][j] = profile_vector;
                    }
                    input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(min_value);
                    input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(max_value);
                    input_explicit_shape_ranges[input_name][static_cast<int64_t>(j)][i].push_back(opt_value);
                }
                else
                {
                    dims_min.d[j] = dims.d[j];
                    dims_max.d[j] = dims.d[j];
                    dims_opt.d[j] = dims.d[j];
                }
            }

            trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
            trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
            trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
        }
    }
    return true;
}

// Per TensorRT documentation, logger needs to be a singleton.
// EP instances register/deregister their OrtLogger via set_ort_logger/clear_ort_logger
// so the singleton never holds a dangling pointer.
TensorrtRtxLogger& GetTensorrtRtxLogger(bool verbose_log)
{
    const auto log_level = verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING;
    static TensorrtRtxLogger trt_logger(log_level);
    if (log_level != trt_logger.get_level())
    {
        trt_logger.set_level(log_level);
    }
    return trt_logger;
}

//! Extract a TRT-layer-name -> ONNX-node-names mapping from an engine built with kDETAILED verbosity.
//! Returns an empty map if the engine has no ONNX metadata (e.g. built with kLAYER_NAMES_ONLY).
static std::unordered_map<std::string, std::string> ExtractLayerOnnxMapping(nvinfer1::ICudaEngine& engine)
{
    std::unordered_map<std::string, std::string> mapping;

    auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine.createEngineInspector());
    if (!inspector)
    {
        return mapping;
    }

    for (int32_t i = 0, nb_layers = engine.getNbLayers(); i < nb_layers; ++i)
    {
        const char* info_raw = inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON);
        if (!info_raw)
        {
            continue;
        }
        const std::string_view info(info_raw);

        // Extract "Name" value - first occurrence in the JSON.
        const std::string name_key = "\"Name\": \"";
        const auto name_start = info.find(name_key);
        if (name_start == std::string::npos)
        {
            continue;
        }
        const auto name_val_start = name_start + name_key.size();
        const auto name_val_end = info.find('"', name_val_start);
        if (name_val_end == std::string::npos)
        {
            continue;
        }
        const std::string_view layer_name = info.substr(name_val_start, name_val_end - name_val_start);

        // Extract "Metadata" value.
        const std::string meta_key = "\"Metadata\": \"";
        const auto meta_start = info.find(meta_key);
        if (meta_start == std::string::npos)
        {
            continue;
        }
        const auto meta_val_start = meta_start + meta_key.size();
        const auto meta_val_end = info.find('"', meta_val_start);
        if (meta_val_end == std::string::npos)
        {
            continue;
        }
        const std::string_view metadata = info.substr(meta_val_start, meta_val_end - meta_val_start);
        if (metadata.empty())
        {
            continue;
        }

        // Metadata format: "[ONNX Layer: /node1]\x1f[ONNX Layer: /node2]..."
        // Split on unit separator (\x1f = ASCII 31), extract node names, join with ", ".
        std::string onnx_nodes;
        constexpr std::string_view prefix = "[ONNX Layer: ";
        const char delim = '\x1f';
        std::string_view::size_type pos = 0;
        while (pos < metadata.size())
        {
            auto end = metadata.find(delim, pos);
            if (end == std::string_view::npos)
            {
                end = metadata.size();
            }
            const std::string_view token = metadata.substr(pos, end - pos);
            // token looks like "[ONNX Layer: /conv1/Conv]"
            if (token.size() > prefix.size() + 1 && token.substr(0, prefix.size()) == prefix)
            {
                const std::string_view node_name = token.substr(prefix.size(), token.size() - prefix.size() - 1);
                if (!onnx_nodes.empty())
                {
                    onnx_nodes += ", ";
                }
                onnx_nodes += node_name;
            }
            pos = (end == metadata.size()) ? end : end + 1;
        }

        if (!onnx_nodes.empty())
        {
            mapping.emplace(layer_name, std::move(onnx_nodes));
        }
    }

    return mapping;
}

// Helper function to check if a data type is supported by input output nodes ofNvTensorRTRTX EP
static bool IsSupportedInputOutputDataType(ONNXTensorElementDataType data_type)
{
    switch (data_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:         // kFLOAT - 32-bit floating point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:       // kHALF - IEEE 16-bit floating-point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:      // kBF16 - Brain float 16
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:          // kBOOL - 8-bit boolean
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:          // kINT4 - 4-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:          // kINT8 - 8-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:         // kUINT8 - 8-bit unsigned integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:         // kINT32 - 32-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:         // kINT64 - 64-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:  // kFP8 - 8-bit floating point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:    // kFP4 - 4-bit floating point
        return true;
    default:
        return false;
    }
}

// Helper function to check if a data type is supported by NvTensorRTRTX EP
static bool IsSupportedDataType(ONNXTensorElementDataType data_type)
{
    switch (data_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:         // kFLOAT - 32-bit floating point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:       // kHALF - IEEE 16-bit floating-point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:      // kBF16 - Brain float 16
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:          // kBOOL - 8-bit boolean
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:          // kINT4 - 4-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:          // kINT8 - 8-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:         // kUINT8 - 8-bit unsigned integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:         // kINT32 - 32-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:         // kINT64 - 64-bit signed integer
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:  // kFP8 - 8-bit floating point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:        // kDOUBLE - 64-bit floating point
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:    // kFP4 - 4-bit floating point
        return true;
    default:
        return false;
    }
}

// Helper function to get data type name as string
static std::string GetDataTypeName(ONNXTensorElementDataType data_type)
{
    switch (data_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return "BFLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return "BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
        return "INT4";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "INT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "UINT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "INT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "INT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        return "FLOAT8E4M3FN";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "DOUBLE";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return "STRING";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "UINT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "UINT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "UINT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "INT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return "COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return "COMPLEX128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
        return "FLOAT4E2M1";
    default:
        return "UNKNOWN(" + std::to_string(static_cast<int>(data_type)) + ")";
    }
}

SubGraphCollection_t TensorrtRtxExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input,
                                                                    int iterations, const int max_iterations,
                                                                    const OrtGraph* graph,
                                                                    bool* early_termination) const
{
    // Walk the input subgraphs and recursively refine TensorRT-supported regions.
    // High-level flow per group:
    // 1) Build an ONNX ModelProto for the current graph or subgraph.
    // 2) Ask the TRT parser for supported subgraphs (in proto node indices).
    // 3) Map proto node indices back to ORT node indices.
    // 4) Recurse into each supported/unsupported region to further split.
    // 5) Remap child indices back to the parent graph's node index space.
    /// Return if iterations are exceeding predefined number
    SubGraphCollection_t nodes_list_output;
    if (iterations > max_iterations)
    {
        *early_termination = true;
        return nodes_list_output;
    }

    for (const auto& group : nodes_vector_input)
    {
        // Skip empty groups.
        if (group.first.empty())
        {
            continue;
        }

        // If already marked supported, keep as-is.
        if (group.second)
        {
            nodes_list_output.push_back(group);
            continue;
        }
        auto ort_graph = Ort::ConstGraph(graph);

        // Note: has_control_flow_op and load_initializers_inline_true are reserved for future use
        // bool has_control_flow_op = false;
        // constexpr const bool load_initializers_inline_true = true;

        ONNX_NAMESPACE::ModelProto model_proto;

        // Keep Ort::Graph alive to prevent dangling pointer (only used when iterations > 0)
        Ort::Graph subgraph_owner{nullptr};
        std::vector<size_t> subgraph_parent_indices;

        // Create handler for initializer data (external references with memory addresses)
        auto handle_initializer_data = CreateInitializerDataHandler();

        // Decide whether we are operating on the full graph or a derived subgraph.
        const OrtGraph* subgraph_view = graph;

        // Store the Ort::Graph object to keep it alive (prevents dangling pointer)
        subgraph_owner = GetSubgraph(group, ort_graph);
        subgraph_view = static_cast<const OrtGraph*>(subgraph_owner);

        // Build a mapping from ORT node id -> index in the parent graph so we can
        // later remap child indices back to the parent graph's index space.
        auto parent_nodes = ort_graph.GetNodes();
        std::unordered_map<size_t, size_t> parent_id_to_index;
        parent_id_to_index.reserve(parent_nodes.size());
        for (size_t idx = 0; idx < parent_nodes.size(); ++idx)
        {
            parent_id_to_index[parent_nodes[idx].GetId()] = idx;
        }

        auto subgraph_node_views = subgraph_owner.GetNodes();
        subgraph_parent_indices.reserve(subgraph_node_views.size());
        for (const auto& node : subgraph_node_views)
        {
            auto it = parent_id_to_index.find(node.GetId());
            if (it != parent_id_to_index.end())
            {
                subgraph_parent_indices.push_back(it->second);
            }
        }

        // Serialize the chosen graph to ModelProto so TRT can parse it.
        const OrtGraph& graph_to_serialize = *subgraph_owner;
        auto status = OrtEpUtils::OrtGraphToProto(graph_to_serialize, model_proto, handle_initializer_data);
        if (!status.IsOK())
        {
            std::string message =
                "[NvTensorRTRTX EP] OrtGraphToProto failed; skipping subgraph. Error: " + status.GetErrorMessage();
            OrtStatus* log_status = ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                              message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
            if (log_status)
            {
                ort_api.ReleaseStatus(log_status);
            }
            continue;
        }
        // Capability discovery must see the same proto preprocessing sequence
        // that engine build will later apply. Today that includes policy-driven
        // Q/DQ lowering plus logical-output compatibility cleanup for
        // Chromium's bool->uint8 bridge, Clip bound compatibility for WebNN
        // clamp defaults, and dilated pooling compatibility lowering.
        TensorRtProtoPreprocessingOptions preprocessing_options{};
        preprocessing_options.multi_rotary_cache_concat_offset = multi_rotary_cache_concat_offset_;
        const auto lowered_qdq_info = RunTensorRtProtoPreprocessing(model_proto, preprocessing_options);

        // TRT parser consumes serialized model bytes.
        std::string string_buf;
        model_proto.SerializeToString(&string_buf);
        std::vector<TensorrtUserWeights> userWeights;

        // Pass initializer blobs separately as external data.
        {
            auto initializers = subgraph_owner.GetInitializers();
            userWeights.reserve(initializers.size());
            for (auto& initializer : initializers)
            {
                Ort::ConstValue ort_value{nullptr};
                THROW_IF_ERROR(initializer.GetInitializer(ort_value));
                if (ort_value.IsTensor())
                {
                    userWeights.emplace_back(TensorrtUserWeights(initializer.GetName(), ort_value.GetTensorRawData(),
                                                                 ort_value.GetTensorSizeInBytes()));
                }
            }

            // GetInitializers() returns only local-scope initializers. Outer-scope
            // constant initializers (e.g. Gather indices defined in a parent Loop
            // scope) are serialized into the subgraph proto by OrtGraphToProto but
            // never returned by GetInitializers(), so TRT falls back to file-based
            // lookup and fails. Walk node inputs to collect them explicitly.
            std::unordered_set<std::string> registered;
            for (const auto& w : userWeights)
                registered.insert(w.Name());

            auto collect_outer_scope = [&](Ort::ConstValueInfo vi)
            {
                if (!vi || !vi.IsFromOuterScope() || !vi.IsConstantInitializer())
                    return;
                if (!registered.insert(vi.GetName()).second)
                    return;
                Ort::ConstValue ort_value{nullptr};
                OrtStatus* s = vi.GetInitializer(ort_value);
                if (s != nullptr)
                {
                    std::string msg = "[NvTensorRTRTX EP] GetSupportedList: failed to fetch outer-scope initializer '";
                    msg += vi.GetName();
                    msg += "'; weight will be absent from TRT engine";
                    OrtStatus* log_s = ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                                 msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                    if (log_s)
                        ort_api.ReleaseStatus(log_s);
                    Ort::GetApi().ReleaseStatus(s);
                    return;
                }
                if (ort_value.IsTensor() && ort_value.GetTensorSizeInBytes() > 0)
                    userWeights.emplace_back(TensorrtUserWeights(vi.GetName(), ort_value.GetTensorRawData(),
                                                                 ort_value.GetTensorSizeInBytes()));
            };

            for (const auto& node : subgraph_owner.GetNodes())
            {
                for (const auto& vi : node.GetInputs())
                    collect_outer_scope(vi);
                for (const auto& vi : node.GetImplicitInputs())
                    collect_outer_scope(vi);
            }
        }

        // Ask TRT for supported subgraphs and then recurse on them.
        SubGraphCollection_t parser_nodes_list;
        TensorrtRtxLogger& trt_logger = GetTensorrtRtxLogger(detailed_build_log_);
        auto trt_builder = GetBuilder(trt_logger);
        auto network_flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        auto trt_network = std::unique_ptr<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(network_flags));
        bool is_model_supported = false;

        // Lowering can expand one original ORT node into several serialized ONNX nodes:
        //
        //   ORT node 2: QuantizeLinear
        //        |
        //        v
        //   proto: Div(doc=2) -> Add(doc=2) -> Round(doc=2) -> Clamp(doc=2) -> Cast(doc=2)
        //
        // TRT reports support in terms of proto node indices, but ORT capability
        // partitioning still needs the original ORT node index. We therefore read
        // the original ORT node id back out of doc_string and map:
        //
        //   proto node index -> original ORT node id -> ORT node index
        //
        // Any proto node without a valid original ORT id is a lowering-only helper
        // and must not claim ownership of an unrelated ORT node.
        auto build_proto_to_ort_index = [&model_proto](const std::unordered_map<size_t, size_t>& ort_node_id_to_index)
        {
            std::vector<std::optional<size_t>> proto_idx_to_ort_idx;
            const auto& graph_proto = model_proto.graph();
            proto_idx_to_ort_idx.reserve(graph_proto.node_size());
            for (int proto_idx = 0; proto_idx < graph_proto.node_size(); ++proto_idx)
            {
                const auto& node_proto = graph_proto.node(proto_idx);
                if (const auto node_id = TryParseNodeId(node_proto.doc_string()))
                {
                    auto it = ort_node_id_to_index.find(*node_id);
                    proto_idx_to_ort_idx.push_back(
                        (it != ort_node_id_to_index.end()) ? std::optional<size_t>(it->second) : std::nullopt);
                }
                else
                {
                    proto_idx_to_ort_idx.push_back(std::nullopt);
                }
            }

            return proto_idx_to_ort_idx;
        };

        // Limit the scope of trt_parser so that model gets unloaded from memory asap.
        {
            auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(
                nvonnxparser::createParser(*trt_network, trt_logger));
            trt_parser->loadModelProto(string_buf.data(), string_buf.size(), model_path_);
            for (auto const& userWeight : userWeights)
            {
                trt_parser->loadInitializer(userWeight.Name(), userWeight.Data(), userWeight.Size());
            }
            is_model_supported = trt_parser->parseModelProto();

            // Get ORT nodes from subgraph
            Ort::ConstGraph ort_subgraph_view{subgraph_view};
            auto ort_nodes = ort_subgraph_view.GetNodes();
            std::unordered_map<size_t, size_t> ort_node_id_to_index;
            ort_node_id_to_index.reserve(ort_nodes.size());
            for (size_t idx = 0; idx < ort_nodes.size(); ++idx)
            {
                ort_node_id_to_index[ort_nodes[idx].GetId()] = idx;
            }
            auto proto_idx_to_ort_idx = build_proto_to_ort_index(ort_node_id_to_index);

            // Only query subgraph info when parsing succeeded.
            // Calling getNbSubgraphs()/getSubgraphNodes() on a failed parser
            // crashes — leave parser_nodes_list empty so the recursion correctly
            // treats this group as fully unsupported.
            if (is_model_supported)
            {
                // TRT returns subgraph nodes as indices into the serialized model graph.
                // Convert those to ORT node indices for recursive processing.
                // Note: Calling getNbSubgraphs or getSubgraphNodes before calling supportsModelV2 results in
                // undefined behavior.
                auto num_subgraphs = trt_parser->getNbSubgraphs();
                parser_nodes_list.reserve(num_subgraphs);

                for (int64_t i = 0; i < num_subgraphs; ++i)
                {
                    int64_t subgraph_len = 0;
                    int64_t* subgraph_nodes = trt_parser->getSubgraphNodes(i, subgraph_len);
                    parser_nodes_list.emplace_back();
                    parser_nodes_list.back().first.reserve(subgraph_len);
                    std::unordered_set<size_t> seen_ort_node_indices;
                    for (int64_t j = 0; j < subgraph_len; ++j)
                    {
                        // Lowering can expand one ORT node into several proto nodes. We only
                        // keep parser nodes that still carry a valid original ORT node id.
                        size_t proto_node_idx = static_cast<size_t>(subgraph_nodes[j]);
                        if (proto_node_idx >= proto_idx_to_ort_idx.size() ||
                            !proto_idx_to_ort_idx[proto_node_idx].has_value())
                        {
                            continue;
                        }

                        size_t ort_node_idx = *proto_idx_to_ort_idx[proto_node_idx];
                        if (seen_ort_node_indices.insert(ort_node_idx).second)
                        {
                            parser_nodes_list.back().first.push_back(static_cast<int64_t>(ort_node_idx));
                        }
                    }
                    parser_nodes_list.back().second = is_model_supported;
                }
            }

            if (!lowered_qdq_info.folded_constant_nodes.empty())
            {
                // Constant-folded DQ nodes can disappear from the parser's node list entirely.
                // Reattach the original ORT node to every supported subgraph that still references
                // the folded tensor so partition ownership matches the pre-lowered graph.
                //
                //   before lowering: dq_node --> consumer
                //   after folding   : folded_initializer -- Identity? --> consumer
                //
                // If TRT prunes the helper Identity, the parser may only report the consumer.
                // We recover ownership by looking for subgraphs that still mention the folded
                // tensor name and then reattaching the original ORT DQ node to those subgraphs.
                std::unordered_map<std::string, std::vector<size_t>> tensor_to_parser_subgraphs;
                const auto& graph_proto = model_proto.graph();
                for (size_t parser_idx = 0; parser_idx < parser_nodes_list.size(); ++parser_idx)
                {
                    for (const auto proto_node_index_i64 : parser_nodes_list[parser_idx].first)
                    {
                        const size_t ort_node_index = static_cast<size_t>(proto_node_index_i64);
                        if (ort_node_index >= ort_nodes.size())
                        {
                            continue;
                        }
                        const auto ort_node_id = ort_nodes[ort_node_index].GetId();
                        for (int proto_idx = 0; proto_idx < graph_proto.node_size(); ++proto_idx)
                        {
                            const auto& node_proto = graph_proto.node(proto_idx);
                            const auto node_id = TryParseNodeId(node_proto.doc_string());
                            if (!node_id || *node_id != ort_node_id)
                            {
                                continue;
                            }
                            for (const auto& input_name : node_proto.input())
                            {
                                if (!input_name.empty())
                                {
                                    tensor_to_parser_subgraphs[input_name].push_back(parser_idx);
                                }
                            }
                            for (const auto& output_name : node_proto.output())
                            {
                                if (!output_name.empty())
                                {
                                    tensor_to_parser_subgraphs[output_name].push_back(parser_idx);
                                }
                            }
                        }
                    }
                }

                std::vector<std::unordered_set<size_t>> seen_ort_node_indices(parser_nodes_list.size());
                for (size_t parser_idx = 0; parser_idx < parser_nodes_list.size(); ++parser_idx)
                {
                    seen_ort_node_indices[parser_idx].insert(parser_nodes_list[parser_idx].first.begin(),
                                                             parser_nodes_list[parser_idx].first.end());
                }

                for (const auto& folded_node : lowered_qdq_info.folded_constant_nodes)
                {
                    auto ort_it = ort_node_id_to_index.find(folded_node.original_node_id);
                    if (ort_it == ort_node_id_to_index.end())
                    {
                        continue;
                    }

                    auto subgraph_it = tensor_to_parser_subgraphs.find(folded_node.output_name);
                    if (subgraph_it == tensor_to_parser_subgraphs.end())
                    {
                        continue;
                    }

                    for (const size_t parser_idx : subgraph_it->second)
                    {
                        if (parser_idx >= parser_nodes_list.size())
                        {
                            continue;
                        }
                        if (seen_ort_node_indices[parser_idx].insert(ort_it->second).second)
                        {
                            parser_nodes_list[parser_idx].first.push_back(static_cast<int64_t>(ort_it->second));
                        }
                    }
                }
            }
        }

        // Recurse into the returned subgraphs to refine support boundaries.
        const int next_iteration = iterations + 1;
        auto next_nodes_list =
            GetSupportedList(parser_nodes_list, next_iteration, max_iterations, subgraph_view, early_termination);

        // Remap indices from child graph scope back to the parent graph.
        // Root call (iterations == 0) uses group.first, which maps child indices to parent indices directly.
        // Recursive calls use subgraph_parent_indices, which preserves the actual subgraph node order
        // returned by GetGraphView (this order can differ from group.first).
        const auto& parent_index_map = (subgraph_parent_indices.empty()) ? group.first : subgraph_parent_indices;

        for (auto& child_group : next_nodes_list)
        {
            for (auto& node_idx : child_group.first)
            {
                node_idx = static_cast<int64_t>(parent_index_map[static_cast<size_t>(node_idx)]);
            }
            nodes_list_output.push_back(child_group);
        }
    }

    return nodes_list_output;
}

// Helper function to check if a node has supported data types
static bool CheckNodeDataTypes(const Ort::ConstNode& node)
{
    // Check input data types

    for (auto input : node.GetInputs())
    {
        const OrtValueInfo* input_ptr = (const OrtValueInfo*)(input);

        if (input_ptr != nullptr)
        {
            auto type_info = input.TypeInfo();
            if (!type_info || type_info.GetONNXType() != ONNX_TYPE_TENSOR)
                continue;  // skip unconnected or non-tensor inputs

            if (!IsSupportedDataType(type_info.GetTensorTypeAndShapeInfo().GetElementType()))
            {
                return false;
            }
        }
    }
    // Check output data types
    for (auto output : node.GetOutputs())
    {
        const OrtValueInfo* output_ptr = (const OrtValueInfo*)(output);
        if (output_ptr == nullptr)
            continue;  // optional unconnected output (e.g., LSTM Y sequence output)

        auto type_info = output.TypeInfo();
        if (!type_info)
            continue;  // no type info for unconnected optional output

        if (type_info.GetONNXType() != ONNX_TYPE_TENSOR)
            continue;  // non-tensor outputs (sequence, map, etc.) are not type-checked here

        if (!IsSupportedDataType(type_info.GetTensorTypeAndShapeInfo().GetElementType()))
        {
            return false;
        }
    }
    return true;
}

std::unique_lock<std::mutex> TensorrtRtxExecutionProvider::GetApiLock() const
{
    static std::mutex singleton;
    return std::unique_lock<std::mutex>(singleton);
}

nvinfer1::IBuilder* TensorrtRtxExecutionProvider::GetBuilder(TensorrtRtxLogger& trt_logger) const
{
    if (!builder_)
    {
        {
            auto lock = GetApiLock();
            builder_ = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
            unsigned int num_threads = std::thread::hardware_concurrency();
            builder_->setMaxThreads(num_threads / 2);

            // Force synchronous GPU allocation during engine build if requested.
            if (sync_gpu_allocator_)
            {
                builder_->setGpuAllocator(sync_gpu_allocator_.get());
            }
        }
    }
    return builder_.get();
}

Ort::Graph TensorrtRtxExecutionProvider::GetSubgraph(SubGraph_t graph_nodes_index, const Ort::ConstGraph& graph) const
{

    auto all_nodes = graph.GetNodes();

    // Build a vector of ConstNode for the subgraph
    std::vector<Ort::ConstNode> subgraph_const_nodes;
    subgraph_const_nodes.reserve(graph_nodes_index.first.size());
    for (const auto& index : graph_nodes_index.first)
    {
        if (index < all_nodes.size())
        {
            subgraph_const_nodes.push_back(all_nodes[index]);
        }
    }

    // Use Graph_GetGraphView via the C++ API to create an OrtGraph subgraph view
    return graph.GetGraphView(subgraph_const_nodes);
}

// Check the graph is the subgraph of control flow op
bool TensorrtRtxExecutionProvider::IsSubGraphOfControlFlowOp(const OrtGraph* graph) const
{
    const OrtNode* parent_node = nullptr;
    THROW_IF_ERROR(ort_api.Graph_GetParentNode(graph, &parent_node));
    if (parent_node)
    {
        const char* op_type = nullptr;
        THROW_IF_ERROR(ort_api.Node_GetOperatorType(parent_node, &op_type));

        if (control_flow_op_set_.find(std::string(op_type)) != control_flow_op_set_.end())
        {
            return true;
        }
    }
    return false;
}

// Check whether all the nodes of subgraph are supported
bool TensorrtRtxExecutionProvider::IsSubGraphFullySupported(const OrtGraph* graph,
                                                            SubGraphCollection_t supported_nodes_vector) const
{
    size_t num_nodes = 0;
    THROW_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

    int number_of_trt_nodes = 0;
    for (const auto& group : supported_nodes_vector)
    {
        if (!group.first.empty())
        {
            number_of_trt_nodes += static_cast<int>(group.first.size());
        }
    }

    return number_of_trt_nodes == num_nodes;
}

OrtStatus* TensorrtRtxExecutionProvider::CreateNodeComputeInfoFromPrecompiledEngine(
    OrtEp* this_ptr, const OrtGraph* graph, const OrtNode* fused_node,
    std::unordered_map<std::string, size_t>& input_map, std::unordered_map<std::string, size_t>& output_map,
    OrtNodeComputeInfo** node_compute_info)
{
    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);
    ScopedCudaContext compute_stream_context(ep->compute_stream_context_);

    // Compile-only sessions never run inference. The input is already an EPContext model,
    // so no engine save is needed either — skip deserialization and execution context creation
    // and register a stub compute info that will not be invoked.
    if (ep->compile_only_mode_)
    {
        Ort::ThrowOnError(
            ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                      "[NvTensorRTRTX EP] Compile-only session on EPContext input: skipping engine "
                                      "deserialization and execution-context creation; registering stub compute info.",
                                      ORT_FILE, __LINE__, __FUNCTION__));
        auto stub = std::make_unique<TensorRtRtxCompileOnlyNodeComputeInfo>();
        *node_compute_info = stub.release();
        return nullptr;
    }

    const char* name = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &name));
    std::string fused_node_name = name;

    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
    std::unordered_map<std::string, size_t> input_indexes;  // TRT engine input name -> ORT kernel context input index
    std::unordered_map<std::string, size_t>
        output_indexes;                                    // TRT engine output name -> ORT kernel context output index
    std::unordered_map<std::string, size_t> output_types;  // TRT engine output name -> ORT output tensor type

    // Get engine binary data and deserialize it
    std::unique_ptr<EPContextNodeReader> ep_context_node_reader = std::make_unique<EPContextNodeReader>(
        *ep, logger_, &trt_engine, runtime_.get(), model_path_, compute_capability_, weight_stripped_engine_enable_,
        onnx_model_folder_path_, onnx_model_bytestream_, onnx_model_bytestream_size_, onnx_external_data_bytestream_,
        onnx_external_data_bytestream_size_, detailed_build_log_);
    RETURN_IF_ERROR(ep_context_node_reader->GetEpContextFromGraph(*graph));
    RETURN_IF_ERROR(
        ApplyWeightStreamingBudget(*trt_engine, weight_streaming_budget_, fused_node_name, ep->logger_, ep->ort_api));

    std::unique_ptr<nvinfer1::IRuntimeCache> trt_runtime_cache;
    auto trt_runtime_config = std::unique_ptr<nvinfer1::IRuntimeConfig>(trt_engine->createRuntimeConfig());
    if (!trt_runtime_config)
    {
        return ort_api.CreateStatus(
            ORT_EP_FAIL, "[NvTensorRTRTX EP] createRuntimeConfig returned null; cannot build execution context.");
    }
    if (cuda_graph_enable_)
    {
        trt_runtime_config->setDynamicShapesKernelSpecializationStrategy(
            nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
#if TRT_MAJOR_RTX > 1 || (TRT_MAJOR_RTX == 1 && TRT_MINOR_RTX >= 3)
        auto cuda_strategy_flag =
            trt_runtime_config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
        Ort::ThrowOnError(
            ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                      ("[NvTensorRTRTX EP] CUDA graph strategy with RTX Graph capture enabled : " +
                                       std::to_string(cuda_strategy_flag))
                                          .c_str(),
                                      ORT_FILE, __LINE__, __FUNCTION__));
#else
        Ort::ThrowOnError(ort_api.Logger_LogMessage(
            &ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            "[NvTensorRTRTX EP] CUDA graph is enabled but RTX Graph capture is not available. "
            "The current TRT RTX version does not support RTX Graph. "
            "Please upgrade to TRT RTX >= 1.3 to use RTX Graph capture feature for optimal CUDA graph performance.",
            ORT_FILE, __LINE__, __FUNCTION__));
#endif
    }
    trt_runtime_config->setExecutionContextAllocationStrategy(
        nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    std::filesystem::path runtime_cache_file;
    if (!runtime_cache_.empty())
    {
        // Use partition_name from EPContext node for runtime cache path so it matches the name
        const std::string& partition_name = ep_context_node_reader->GetPartitionName();
        const std::string& runtime_cache_name_for_path = partition_name.empty() ? fused_node_name : partition_name;
        auto resolved_runtime_cache_file =
            ResolveRuntimeCacheFile(runtime_cache_, runtime_cache_name_for_path, ep->logger_, ep->ort_api);
        if (resolved_runtime_cache_file)
        {
            runtime_cache_file = std::move(*resolved_runtime_cache_file);
            trt_runtime_cache =
                CreateAndAttachRuntimeCache(*trt_runtime_config, runtime_cache_file, ep->logger_, ep->ort_api);
        }
    }

    // Build context
    // Note: Creating an execution context from an engine is thread safe per TRT doc
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading

    auto trt_context = CreateOwnedExecutionContext(*trt_engine, runtime_cache_file, std::move(trt_runtime_cache),
                                                   std::move(trt_runtime_config), ep->ort_api);
    if (!trt_context)
    {
        std::string message = "NvTensorRTRTX EP could not build execution context for fused node: " + fused_node_name;
        return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    if (ep->profiling_enable_)
    {
        trt_context->setProfiler(ep->profiler_.get());
        ep->profiler_->SetLayerOnnxMapping(ExtractLayerOnnxMapping(*trt_engine));
    }

    bool is_dynamic_shape_context = false;
    // Create input/output to index maps
    for (int32_t i = 0; i < trt_engine->getNbIOTensors(); ++i)
    {
        auto const& tensor_name = trt_engine->getIOTensorName(i);
        auto const& mode = trt_engine->getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            is_dynamic_shape_context |= checkTrtDimIsDynamic(trt_engine->getTensorShape(tensor_name));
            const auto& iter = input_map.find(tensor_name);
            if (iter != input_map.end())
            {
                input_indexes[tensor_name] = iter->second;
            }
        }
        else
        {
            const auto& iter = output_map.find(tensor_name);
            if (iter != output_map.end())
            {
                output_indexes[tensor_name] = iter->second;
            }
        }
    }

    // Create output to type map
    Ort::ConstNode ort_fused_node(fused_node);
    for (auto output : ort_fused_node.GetOutputs())
    {
        auto output_name = output.GetName();
        auto type_info = output.TypeInfo();
        output_types[output_name] = type_info.GetTensorTypeAndShapeInfo().GetElementType();
    }
    engines_.emplace(fused_node_name, std::move(trt_engine));
    contexts_.emplace(fused_node_name, std::move(trt_context));
    input_info_[fused_node_name].push_back(input_indexes);
    output_info_[fused_node_name].push_back(output_indexes);
    output_info_[fused_node_name].push_back(output_types);

    auto compute_state = std::make_unique<TensorrtRtxEpContextNodeComputeState>();
    compute_state->device_id = device_id_;
    compute_state->fused_node_name = fused_node_name;
    compute_state->engine = &engines_.at(fused_node_name);
    compute_state->context = &contexts_.at(fused_node_name);
    compute_state->input_info = input_info_[fused_node_name];
    compute_state->output_info = output_info_[fused_node_name];
    compute_state->tensorrt_mu_ptr = &tensorrt_rtx_mu_;
    compute_state->is_dynamic_shape = is_dynamic_shape_context;

    ep->compute_states_for_ep_context_[fused_node_name] = std::move(compute_state);
    auto ep_node_compute_info = std::make_unique<TensorRtRtxEpContextNodeComputeInfo>(*ep);
    *node_compute_info = ep_node_compute_info.release();

    return nullptr;
}

OrtStatus* TensorrtRtxExecutionProvider::CreateNodeComputeInfoFromGraph(
    OrtEp* this_ptr, const OrtGraph* graph, const OrtNode* fused_node,
    std::unordered_map<std::string, size_t>& input_map, std::unordered_map<std::string, size_t>& output_map,
    OrtNodeComputeInfo** node_compute_info, OrtNode** ep_context_node)
{
    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);
    ScopedCudaContext compute_stream_context(ep->compute_stream_context_);
    // Construct ModelProto from OrtGraph
    ONNX_NAMESPACE::ModelProto model_proto;

    auto ort_graph = Ort::ConstGraph(graph);

    // Collect initializer weights as external data
    std::vector<TensorrtUserWeights> userWeights;
    {
        auto initializers = ort_graph.GetInitializers();
        userWeights.reserve(initializers.size());
        for (auto& initializer : initializers)
        {
            Ort::ConstValue ort_value{nullptr};
            THROW_IF_ERROR(initializer.GetInitializer(ort_value));
            if (ort_value.IsTensor())
            {
                userWeights.emplace_back(TensorrtUserWeights(initializer.GetName(), ort_value.GetTensorRawData(),
                                                             ort_value.GetTensorSizeInBytes()));
            }
        }

        // GetInitializers() returns only local-scope initializers. Outer-scope
        // constant initializers (e.g. Gather indices defined in a parent Loop
        // scope) are serialized into the subgraph proto by OrtGraphToProto but
        // never returned by GetInitializers(), so TRT falls back to file-based
        // lookup and fails. Walk node inputs to collect them explicitly.
        std::unordered_set<std::string> registered;
        for (const auto& w : userWeights)
            registered.insert(w.Name());

        auto collect_outer_scope = [&](Ort::ConstValueInfo vi)
        {
            if (!vi || !vi.IsFromOuterScope() || !vi.IsConstantInitializer())
                return;
            if (!registered.insert(vi.GetName()).second)
                return;
            Ort::ConstValue ort_value{nullptr};
            OrtStatus* s = vi.GetInitializer(ort_value);
            if (s != nullptr)
            {
                std::string msg = "[NvTensorRTRTX EP] Compile: failed to fetch outer-scope initializer '";
                msg += vi.GetName();
                msg += "'; weight will be absent from TRT engine";
                OrtStatus* log_s = ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                             msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                if (log_s)
                    ort_api.ReleaseStatus(log_s);
                Ort::GetApi().ReleaseStatus(s);
                return;
            }
            if (ort_value.IsTensor() && ort_value.GetTensorSizeInBytes() > 0)
                userWeights.emplace_back(
                    TensorrtUserWeights(vi.GetName(), ort_value.GetTensorRawData(), ort_value.GetTensorSizeInBytes()));
        };

        for (const auto& node : ort_graph.GetNodes())
        {
            for (const auto& vi : node.GetInputs())
                collect_outer_scope(vi);
            for (const auto& vi : node.GetImplicitInputs())
                collect_outer_scope(vi);
        }
    }

    // NOTE: a ValueInfo_GetExternalInitializerInfo probe here returned NULL (no external
    // info) for ALL initializers at build — even weights that genuinely came from an external
    // .onnx.data file (llama-int4: 0/235 external; candy_ext: 0/72). ORT fully materializes external
    // data into memory (userWeights via GetTensorRawData) before the EP's compile sees the graph, so
    // the original file/offset/length address book is NOT recoverable. This kills the "recipe carries
    // per-source offsets, fully ONNX-independent" delivery shape.

    // Create handler for initializer data (external references with memory addresses)
    auto handle_initializer_data = CreateInitializerDataHandler();

    auto proto_status = OrtEpUtils::OrtGraphToProto(*graph, model_proto, handle_initializer_data);
    if (!proto_status.IsOK())
    {
        std::string message = "[NvTensorRTRTX EP] OrtGraphToProto failed in Compile; cannot build engine. Error: " +
                              proto_status.GetErrorMessage();
        OrtStatus* log_s = ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                     message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
        if (log_s)
            ort_api.ReleaseStatus(log_s);
        return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }
    // Engine build must replay the same preprocessing sequence as
    // GetSupportedList so TRT compiles the exact proto whose support we
    // advertised during partitioning.
    TensorRtProtoPreprocessingOptions preprocessing_options{};
    preprocessing_options.multi_rotary_cache_concat_offset = multi_rotary_cache_concat_offset_;
    (void)RunTensorRtProtoPreprocessing(model_proto, preprocessing_options);
    std::string string_buf;
    model_proto.SerializeToString(&string_buf);

    if (dump_subgraphs_)
    {
        // Dump TensorRT subgraphs
        const char* name = nullptr;
        RETURN_IF_ERROR(ort_api.Node_GetName(fused_node, &name));
        const uint64_t dump_index = s_subgraph_dump_counter.fetch_add(1, std::memory_order_relaxed);
        const std::string subgraph_base_name = std::string(name) + "_dump_" + std::to_string(dump_index);
        const std::string subgraph_name = subgraph_base_name + ".onnx";
        const std::string subgraph_data = subgraph_base_name + ".data";
        const auto dump_status = OrtEpUtils::DumpModelWithInitializers(model_proto, subgraph_name, subgraph_data);
        if (!dump_status.IsOK())
        {
            const std::string message =
                "[NvTensorRTRTX EP] Failed to dump subgraph '" + subgraph_name + "': " + dump_status.GetErrorMessage();
            OrtStatus* log_s = ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                         message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
            if (log_s)
                ort_api.ReleaseStatus(log_s);
        }
    }

    auto& trt_logger = GetTensorrtRtxLogger(detailed_build_log_);
    auto trt_builder = ep->GetBuilder(trt_logger);
    auto network_flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto trt_network = std::unique_ptr<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(network_flags));
    if (!trt_network)
    {
        return ort_api.CreateStatus(ORT_EP_FAIL,
                                    "[NvTensorRTRTX EP] createNetworkV2 returned null; cannot build engine.");
    }
    auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
    if (!trt_config)
    {
        return ort_api.CreateStatus(ORT_EP_FAIL,
                                    "[NvTensorRTRTX EP] createBuilderConfig returned null; cannot build engine.");
    }
    auto trt_parser =
        tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));

    trt_parser->loadModelProto(string_buf.data(), string_buf.size(), model_path_);
    for (auto const& userWeight : userWeights)
    {
        trt_parser->loadInitializer(userWeight.Name(), userWeight.Data(), userWeight.Size());
    }
    trt_parser->parseModelProto();

    if (max_workspace_size_ > 0)
    {
        trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size_);
    }
    if (max_shared_mem_size_ > 0)
    {
        trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY, max_shared_mem_size_);
    }

    // Set compute capability to kCURRENT by default
    // Must set the number of compute capabilities before setting the capability itself
    constexpr int kDefaultNumComputeCapabilities = 1;
    if (trt_config->getNbComputeCapabilities() == 0)
    {
        trt_config->setNbComputeCapabilities(kDefaultNumComputeCapabilities);
    }
    trt_config->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT, 0);

    int num_inputs = trt_network->getNbInputs();
    int num_outputs = trt_network->getNbOutputs();
    std::unordered_map<std::string, size_t> input_indexes(num_inputs);
    std::unordered_map<std::string, size_t> output_indexes(num_outputs);
    std::unordered_map<std::string, size_t> output_types(num_outputs);
    /*
     * Initialize shape range for each dynamic shape input tensor:
     *   1) If user explicitly specifies optimization profiles via provider options, TRT EP will create those profiles
     * during EP compile time. It won't make adjustment for profile values during EP compute time.
     *
     *   2) If no explicit optimization profiles provided by user, TRT EP will firstly set min/max/opt shape to
     * [INT_MAX, INT_MIN, INT_MIN]. Later in EP compute time, the shape will be adjusted to [min_input_value,
     * max_input_value, max_input_value] based on input tensor value.
     *
     *
     * Once the TRT profiles are created:
     *   1) If all the dynamic shape input tensors have associated profiles explicitly provided by user, those profiles
     * will be applied to TRT builder config and the engine will be built at EP compile time.
     *
     *   2) As long as one of the dynamic shape input tensors has no explicitly associated profile, TRT EP will create
     * default shape as described above, and all the profiles won't be applied and engine won't be built until EP
     * compute time.
     */
    bool has_explicit_profile = false;
    bool has_implicit_profile = false;
    int num_profiles = 0;
    std::vector<nvinfer1::IOptimizationProfile*> trt_profiles;

    // Following c++ map data structure is used to help serialize/deserialize profiles where it saves dynamic shape
    // dimension(s) and min/max/opt values for dynamic shape input tensor.
    //
    // (1) Single profile case:
    // For example, assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2, and tensor_b
    // has one dynamic shape dimension: dim_1. The data will be:
    // {
    //   tensor_a: {
    //              dim_0: [[min_shape, max_shape, opt_shape]],
    //              dim_2: [[min_shape, max_shape, opt_shape]]
    //   },
    //   tensor_b: {
    //              dim_1: [[min_shape, max_shape, opt_shape]]
    //   }
    // }
    //
    // (2) Multiple profiles case:
    // For example, assume tensor_a has one dynamic shap dimension: dim 0, and tensor_b has one dynamic shape dimension:
    // dim_1, and both of the tensors have two profiles. The data will be:
    // {
    //   tensor_a: {
    //     dim_0: [[min_shape_0, max_shape_0, opt_shape_0], [min_shape_1, max_shape_1, opt_shape_1]]
    //   },
    //   tensor_b: {
    //     dim_1: [[min_shape_2, max_shape_2, opt_shape_2], [min_shape_3, max_shape_3, opt_shape_3]]
    //   }
    // }
    ShapeRangesMap input_explicit_shape_ranges;
    ShapeRangesMap input_implicit_shape_ranges;

    bool has_dynamic_shape =
        false;  // True if input tensor has dynamic shape and no explicit profile is specified, otherwise false
    if ((!profile_min_shapes_.empty()) && (!profile_max_shapes_.empty()) && (!profile_opt_shapes_.empty()))
    {
        has_explicit_profile = true;
        has_dynamic_shape = true;
        num_profiles = GetNumProfiles(profile_min_shapes_);
        for (int i = 0; i < num_profiles; i++)
        {
            trt_profiles.push_back(trt_builder->createOptimizationProfile());
        }
    }
    else
    {
        for (unsigned int i = 0, end = num_inputs; i < end; ++i)
        {
            auto input = trt_network->getInput(i);
            has_dynamic_shape |= checkTrtTensorIsDynamic(input);
        }
        if (has_dynamic_shape)
        {

            std::string message =
                "[NvTensorRTRTX EP] No explicit optimization profile was specified. "
                "We will assume a single profile with fully dynamic range. "
                "This feature is experimental and may change in the future."
                "If you plan to use this model as fixed shape we recommend using a free dimension override: "
                "https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides.";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

            trt_profiles.push_back(trt_builder->createOptimizationProfile());
        }
    }

    if (has_dynamic_shape)
    {
        // Iterate all input tensors to check dynamic shape
        for (unsigned int i = 0, end = num_inputs; i < end; ++i)
        {
            auto input = trt_network->getInput(i);
            const std::string& input_name = input->getName();
            nvinfer1::Dims dims = input->getDimensions();

            // Apply explicit optimization profiles provided by user
            bool apply_profile = false;
            bool tensor_has_profile = profile_min_shapes_.find(input_name) != profile_min_shapes_.end() &&
                                      profile_opt_shapes_.find(input_name) != profile_opt_shapes_.end() &&
                                      profile_max_shapes_.find(input_name) != profile_max_shapes_.end();
            if (has_explicit_profile && tensor_has_profile)
            {
                apply_profile = ApplyProfileShapesFromProviderOptions(
                    trt_profiles, input, profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_,
                    input_explicit_shape_ranges, cuda_graph_enable_, ep->logger_, ep->ort_api);
            }
            else
            {
                std::string info_msg = "[NvTensorRTRTX EP] Creating implicit profile for tensor " + input_name;
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                            info_msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
                profile_min_shapes_[input_name] = std::vector<std::vector<int64_t>>{{}};
                profile_min_shapes_[input_name][0].resize(dims.nbDims);
                profile_opt_shapes_[input_name] = std::vector<std::vector<int64_t>>{{}};
                profile_opt_shapes_[input_name][0].resize(dims.nbDims);
                profile_max_shapes_[input_name] = std::vector<std::vector<int64_t>>{{}};
                profile_max_shapes_[input_name][0].resize(dims.nbDims);
                for (int idx_dim = 0; idx_dim < dims.nbDims; ++idx_dim)
                {
                    auto dim_value = dims.d[idx_dim];
                    if (dim_value == -1)
                    {
                        has_implicit_profile = true;
                        // TODO: this is needed until we have a wildcard in the API to support dynamic
                        // shapes
                        profile_min_shapes_[input_name][0][idx_dim] = 0;
                        // TODO: This can be buggy since shape inference can fail with 1 being used as
                        // optimal shape
                        //        [2025-04-04 15:41:58   ERROR] IBuilder::buildSerializedNetwork: Error Code 4: Internal
                        //        Error (kOPT values for profile 0 violate shape constraints: /conv1/Conv: spatial
                        //        dimension of convolution/deconvolution output cannot be negative (build-time output
                        //        dimension of axis 2 is
                        //        (+ (CEIL_DIV (+ h -6) 2) 1)) Condition '<' violated: 2 >= 1.)
                        profile_opt_shapes_[input_name][0][idx_dim] = 1;
                        profile_max_shapes_[input_name][0][idx_dim] = (std::numeric_limits<int16_t>::max)();
                    }
                    else
                    {
                        profile_min_shapes_[input_name][0][idx_dim] = dim_value;
                        profile_opt_shapes_[input_name][0][idx_dim] = dim_value;
                        profile_max_shapes_[input_name][0][idx_dim] = dim_value;
                    }
                }
                apply_profile = ApplyProfileShapesFromProviderOptions(
                    trt_profiles, input, profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_,
                    input_explicit_shape_ranges, cuda_graph_enable_, ep->logger_, ep->ort_api);
            }
            if (!apply_profile)
            {
                std::ostringstream msg;
                msg << "Optimization profile could not be applied for tensor:\n";
                msg << input_name;
                msg << "\n[";
                for (int idx_dim = 0; idx_dim < dims.nbDims; ++idx_dim)
                {
                    msg << dims.d[idx_dim] << ",";
                }
                msg << "]";
                return ort_api.CreateStatus(OrtErrorCode::ORT_EP_FAIL, msg.str().c_str());
            }
        }
        // Set explicit profiles in TRT config if all dynamic shape inputs have associated profiles provided by user
        if (has_explicit_profile || has_implicit_profile)
        {
            // TRT EP has a constraint here.
            // Users need to provide all the dynamic shape inputs with associated profiles if they want to explicitly
            // specify profiles through provider options.
            for (auto trt_profile : trt_profiles)
            {
                trt_config->addOptimizationProfile(trt_profile);
            }
        }
        else
        {
            return ort_api.CreateStatus(ORT_EP_FAIL,
                                        "No explicit or implicit shapes were provided for dynamic shape inputs.");
        }
    }

    // enable sparse weights
    if (sparsity_enable_)
    {
        trt_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
        std::string message = "[NvTensorRTRTX EP] Sparse weights are allowed";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    // limit auxiliary streams
    if (auxiliary_streams_ >= 0)
    {
        trt_config->setMaxAuxStreams(auxiliary_streams_);
        std::string message = "[NvTensorRTRTX EP] Auxiliary streams are set to " + std::to_string(auxiliary_streams_);
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    if (weight_stripped_engine_enable_)
    {
        // Weightless SDK-support guard, PART 2 (functional floor): kSTRIP_PLAN needs a per-GPU-arch
        // functional TensorRT-RTX floor (e.g. SM120/RTX 5090 -> >= 1.6.1.106) that the compile-time
        // parser-version check (weightless_refit.cc) cannot see. If the build below fails on an
        // unsupported arch, PART 3 (the buildSerializedNetwork failure path) surfaces the floor.
        trt_config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
        std::string message = "[NvTensorRTRTX EP] STRIP_PLAN is enabled";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        trt_config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
        message = "[NvTensorRTRTX EP] REFIT_IDENTICAL is enabled";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    if (weight_streaming_budget_.IsEnabled())
    {
        trt_config->setFlag(nvinfer1::BuilderFlag::kWEIGHT_STREAMING);
        std::string message = "[NvTensorRTRTX EP] WEIGHT_STREAMING is enabled";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    if (ep->profiling_enable_)
    {
        trt_config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    }

    // Build TRT engine (if needed) and load TRT engine if:
    //   (1) Graph has no dynamic shape input
    //   (2) All the dynamic shape inputs have associated explicit profiles specified by user
    //
    // Otherwise engine will be handled at inference time.
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine{nullptr};
    std::unique_ptr<nvinfer1::IRuntimeCache> trt_runtime_cache{nullptr};
    std::unique_ptr<nvinfer1::IRuntimeConfig> trt_runtime_config{nullptr};
    std::filesystem::path runtime_cache_file;

    const char* node_name = nullptr;
    ort_api.Node_GetName(fused_node, &node_name);

    // Generate file name for dumping ep context model
    if (dump_ep_context_model_ && ctx_model_path_.empty())
    {
        ctx_model_path_ = GetCtxModelPath(std::filesystem::path(ToPathString(ep_context_file_path_)),
                                          std::filesystem::path(ToPathString(model_path_)));
    }

    {
        auto lock = GetApiLock();
        // Build engine
        std::chrono::steady_clock::time_point engine_build_start;
        if (detailed_build_log_)
        {
            engine_build_start = std::chrono::steady_clock::now();
        }

        std::unique_ptr<nvinfer1::IHostMemory> serialized_engine{
            trt_builder->buildSerializedNetwork(*trt_network, *trt_config)};
        if (serialized_engine == nullptr)
        {
            std::string message = "[NvTensorRTRTX EP] Failed to create serialized engine for fused node: ";
            if (node_name != nullptr)
            {
                message += node_name;
            }
            // Weightless SDK-support guard, PART 3 (runtime handling): a weight-stripped (kSTRIP_PLAN)
            // build can fail here for two very different reasons -- a genuine per-GPU-arch weight-strip
            // capability gap (a Myelin functional floor, e.g. sm_120 / RTX 50-series -> TensorRT-RTX >=
            // 1.6.1.106) OR an environmental CUDA failure unrelated to weight-strip (e.g. CUDA-in-Graphics
            // / D3D12-Vulkan interop, where cudaMallocAsync can fail). We cannot tell them apart from the
            // TRT error, so report the DETECTED environment (compute capability + runtime version) and
            // only point at the sm_120 floor when actually on sm_120; otherwise point at the environmental
            // path. (getInferLibVersion() exposes only MAJOR.MINOR.PATCH, not the build number, so we
            // report the version and name the floor rather than auto-deciding the exact 1.6.1.106 build.)
            if (weight_stripped_engine_enable_)
            {
                const int v = trt_version_;
                const std::string trt_ver =
                    std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." + std::to_string(v % 100);
                message += ". Weight-stripped (kSTRIP_PLAN) build failed. Detected GPU compute capability sm_" +
                           compute_capability_ + ", TensorRT-RTX runtime " + trt_ver +
                           " (build/patch number not reported by the runtime API).";
                if (compute_capability_ == "120")
                {
                    message += " Weight-stripping on sm_120 (Blackwell / RTX 50-series) requires TensorRT-RTX "
                               ">= 1.6.1.106 -- verify the installed version meets this floor.";
                }
                else
                {
                    message += " If this GPU/SDK is expected to support weight-stripping, the failure may be "
                               "environmental (e.g. CUDA-in-Graphics / D3D12-Vulkan interop, where "
                               "cudaMallocAsync can fail) -- try nv_use_sync_gpu_allocator=1.";
                }
            }
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }

        std::string engine_id;
        // Capture engine header (first 64 bytes) for compatibility validation
        if (serialized_engine->size() < kTensorRTEngineHeaderSize)
        {
            std::string message = "[NvTensorRTRTX EP] Engine header invalid (too small or corrupted): " +
                                  std::to_string(serialized_engine->size()) + " bytes";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }
        else
        {
            std::string engine_header_hex = BinaryToHexString(serialized_engine->data(), kTensorRTEngineHeaderSize);
            if (node_name != nullptr && node_name[0] != '\0')
            {
                engine_id = node_name;
            }
            else
            {
                uint64_t engine_hash[2] = {0, 0};
                MurmurHash3_x64_128(serialized_engine->data(), static_cast<int>(serialized_engine->size()), 0,
                                    &engine_hash);
                engine_id = BinaryToHexString(engine_hash, sizeof(engine_hash));
            }

            engine_headers_[engine_id] = engine_header_hex;
        }

        // In compile-only mode the session will not be used for inference. Skip deserialization
        // and GPU context creation — the serialized engine is already saved as an EP context node.
        if (!compile_only_mode_)
        {
            trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
                runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
            if (trt_engine == nullptr)
            {
                std::string message =
                    "[NvTensorRTRTX EP] NvTensorRTRTX EP failed to deserialize engine for fused node: ";
                if (node_name != nullptr)
                {
                    message += node_name;
                }
                return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
            }
            RETURN_IF_ERROR(
                ApplyWeightStreamingBudget(*trt_engine, weight_streaming_budget_, engine_id, ep->logger_, ep->ort_api));

            trt_runtime_config = std::unique_ptr<nvinfer1::IRuntimeConfig>(trt_engine->createRuntimeConfig());
            if (!trt_runtime_config)
            {
                return ort_api.CreateStatus(
                    ORT_EP_FAIL,
                    "[NvTensorRTRTX EP] createRuntimeConfig returned null; cannot build execution context.");
            }
            if (cuda_graph_enable_)
            {
                trt_runtime_config->setDynamicShapesKernelSpecializationStrategy(
                    nvinfer1::DynamicShapesKernelSpecializationStrategy::kEAGER);
#if TRT_MAJOR_RTX > 1 || (TRT_MAJOR_RTX == 1 && TRT_MINOR_RTX >= 3)
                auto cuda_strategy_flag =
                    trt_runtime_config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
                Ort::ThrowOnError(ort_api.Logger_LogMessage(
                    &ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                    ("[NvTensorRTRTX EP] CUDA graph strategy with RTX Graph capture enabled : " +
                     std::to_string(cuda_strategy_flag))
                        .c_str(),
                    ORT_FILE, __LINE__, __FUNCTION__));
#else
                Ort::ThrowOnError(ort_api.Logger_LogMessage(
                    &ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                    "[NvTensorRTRTX EP] CUDA graph is enabled but RTX Graph capture is not available. "
                    "The current TRT RTX version does not support RTX Graph. "
                    "Please upgrade to TRT RTX >= 1.3 to use RTX Graph capture feature for optimal CUDA graph "
                    "performance.",
                    ORT_FILE, __LINE__, __FUNCTION__));
#endif
            }

            trt_runtime_config->setExecutionContextAllocationStrategy(
                nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
            if (!runtime_cache_.empty())
            {
                auto resolved_runtime_cache_file =
                    ResolveRuntimeCacheFile(runtime_cache_, engine_id, ep->logger_, ep->ort_api);
                if (resolved_runtime_cache_file)
                {
                    runtime_cache_file = std::move(*resolved_runtime_cache_file);
                    trt_runtime_cache =
                        CreateAndAttachRuntimeCache(*trt_runtime_config, runtime_cache_file, ep->logger_, ep->ort_api);
                }
            }
        }

        if (detailed_build_log_)
        {
            auto engine_build_stop = std::chrono::steady_clock::now();
            std::string message = "[NvTensorRTRTX EP] TensorRT engine build for " + std::string(node_name) + " took: " +
                                  std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                     engine_build_stop - engine_build_start)
                                                     .count()) +
                                  "ms";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }

        // dump EP context node model
        std::unique_ptr<EPContextNodeHelper> ep_ctx_node_helper =
            std::make_unique<EPContextNodeHelper>(*ep, graph, fused_node);
        if (dump_ep_context_model_)
        {
            // "ep_cache_context" node attribute should be a relative path to context model directory

            std::filesystem::path cache_path;

            // Customize cache prefix if assigned
            const char* name = nullptr;
            THROW_IF_ERROR(ort_api.Node_GetName(fused_node, &name));
            if (!cache_prefix_.empty())
            {
                // Generate cache suffix in case user would like to customize cache prefix
                cache_path = GetCachePath(cache_path_, cache_prefix_ + "_" + std::string(name) + ".engine");
            }
            else
            {
                cache_path = GetCachePath(cache_path_, std::string(name) + ".engine");
            }
            // NV TRT EP per default generates hardware compatible engines for any RTX device with compute capability >
            // 80
            std::string compute_capability_hw_compat = "80+";

            char* serialized_engine_pointer = reinterpret_cast<char*>(serialized_engine->data());
            size_t serialized_engine_size = serialized_engine->size();

            // Capture the weightless refit table now, while the original ONNX structure
            // (string_buf) and weights (userWeights) built above are still in scope. Only
            // meaningful for weight-stripped (kSTRIP_PLAN) engines; an ordinary engine has no
            // refittable weights to observe. A capture failure is logged and treated as
            // "no weightless table" rather than failing the whole compile -- the EPContext model
            // is still produced and remains usable via the legacy RefitEngineImpl path.
            std::vector<trt_rtx_ep::WeightlessRefitRecord> refit_records;
            if (weight_stripped_engine_enable_)
            {
                auto capture_status = ep->CaptureWeightlessRefitTable(*serialized_engine, string_buf, userWeights,
                                                                      detailed_build_log_, refit_records);
                if (capture_status != nullptr)
                {
                    std::string message = std::string("[NvTensorRTRTX EP] Failed to capture the weightless refit "
                                                      "table; this EPContext model will require the original "
                                                      "ONNX model at load time instead: ") +
                                          std::string(ort_api.GetErrorMessage(capture_status));
                    ort_api.ReleaseStatus(capture_status);
                    Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_,
                                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
                    refit_records.clear();
                }
            }

            auto status = ep_ctx_node_helper->CreateEPContextNode(
                cache_path, serialized_engine_pointer, serialized_engine_size, ep_context_embed_mode_,
                compute_capability_hw_compat, std::filesystem::path(ToPathString(model_path_)), refit_records,
                ep_context_node);
            if (status != nullptr)
            {
                return status;
            }
        }
    }

    // In compile-only mode the engine was built and saved but GPU deserialization was skipped.
    // Return a stub compute info — Compute() will never be called since the session is destroyed
    // immediately after compilation without running any inference.
    if (compile_only_mode_)
    {
        std::string message = "[NvTensorRTRTX EP] Compile-only session: skipping GPU deserialization "
                              "and execution-context creation for fused node '" +
                              std::string(node_name ? node_name : "") + "'; registering stub compute info.";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        auto stub = std::make_unique<TensorRtRtxCompileOnlyNodeComputeInfo>();
        *node_compute_info = stub.release();
        return nullptr;
    }

    // Build context
    // Note: Creating an execution context from an engine is thread safe per TRT doc
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading

    auto trt_context = CreateOwnedExecutionContext(*trt_engine, runtime_cache_file, std::move(trt_runtime_cache),
                                                   std::move(trt_runtime_config), ep->ort_api);
    if (!trt_context)
    {
        std::string message = "[NvTensorRTRTX EP] NvTensorRTRTX EP could not build execution context for fused node: ";
        if (node_name != nullptr)
        {
            message += node_name;
        }
        return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    if (ep->profiling_enable_)
    {
        trt_context->setProfiler(ep->profiler_.get());
        ep->profiler_->SetLayerOnnxMapping(ExtractLayerOnnxMapping(*trt_engine));
    }

    bool is_dynamic_shape_context = false;
    // Create input to index map
    for (int i = 0; i < num_inputs; ++i)
    {
        auto input = trt_network->getInput(i);
        const std::string& input_name = input->getName();
        is_dynamic_shape_context |= checkTrtDimIsDynamic(trt_engine->getTensorShape(input_name.c_str()));
        const auto& iter = input_map.find(input_name);
        if (iter != input_map.end())
        {
            input_indexes[input_name] = iter->second;
        }
    }

    // Create output to index and type maps
    const auto& graph_output = model_proto.graph().output();
    for (int i = 0; i < num_outputs; ++i)
    {
        const std::string& output_name = trt_network->getOutput(i)->getName();
        const auto& iter = output_map.find(output_name);
        if (iter != output_map.end())
        {
            output_indexes[output_name] = iter->second;
        }
        const auto& tensor_type = graph_output[i].type().tensor_type();
        output_types[output_name] = tensor_type.elem_type();
    }

    // Save TRT engine, other TRT objects and input/output info to map
    engines_.emplace(node_name, std::move(trt_engine));
    contexts_.emplace(node_name, std::move(trt_context));
    networks_.emplace(node_name, std::move(trt_network));
    input_info_[node_name].push_back(input_indexes);
    output_info_[node_name].push_back(output_indexes);
    output_info_[node_name].push_back(output_types);
    input_shape_ranges_[node_name] = input_implicit_shape_ranges;
    profiles_.emplace(node_name, std::move(trt_profiles));

    auto compute_state = std::make_unique<TensorrtRtxComputeState>();

    *compute_state = {
        node_name,
        trt_builder,
        &engines_.at(node_name),
        &contexts_.at(node_name),
        &networks_.at(node_name),
        input_info_[node_name],
        output_info_[node_name],
        input_shape_ranges_[node_name],
        &tensorrt_rtx_mu_,
        engine_cache_enable_,
        PathToUTF8String(cache_path_.native()),
        runtime_.get(),
        profiles_[node_name],
        engine_decryption_enable_,
        engine_decryption_,
        engine_encryption_,
        detailed_build_log_,
        sparsity_enable_,
        device_id_,
        auxiliary_streams_,
        cuda_graph_enable_,
        multi_profile_enable_,
        trt_profile_index_,
        is_dynamic_shape_context,
        cache_prefix_,
        "",     // cache_suffix
        {},     // scratch_buffers
        {},     // input_tensors
        {},     // output_tensors
        true,   // is_first_run
        false,  // skip_io_binding_allowed
    };
    ep->compute_states_[node_name] = std::move(compute_state);

    // Update the OrtNodeComputeInfo associated with the graph.
    auto ep_node_compute_info = std::make_unique<TensorRtRtxEpNodeComputeInfo>(*ep);
    *node_compute_info = ep_node_compute_info.release();

    return nullptr;
}

bool TensorrtRtxExecutionProvider::AllNodesAssignedToSpecificEP(const OrtGraph* graph,
                                                                const std::string& provider_type) const
{
    size_t num_nodes = 0;
    THROW_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

    // Get all the nodes from the graph
    std::vector<const OrtNode*> nodes(num_nodes);
    THROW_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    for (const auto node : nodes)
    {
        const char* ep_name;
        THROW_IF_ERROR(ort_api.Node_GetEpName(node, &ep_name));

        if (std::string(ep_name) != provider_type)
        {
            return false;
        }
    }

    return num_nodes != 0;
}

// Detect and remove cycles from supported node list
bool TensorrtRtxExecutionProvider::DetectTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector,
                                                             const Ort::ConstGraph& graph, const HashValue& model_hash,
                                                             bool remove_cycles) const
{
    auto ort_nodes = graph.GetNodes();
    bool trt_cycle = true, cycle_detected = false;

    while (trt_cycle)
    {
        trt_cycle = false;
        std::unordered_map<std::string, size_t> node_to_index_map;
        std::unordered_map<size_t, std::string> index_to_node_map;
        std::unordered_map<std::string, std::unordered_set<std::string>> input_to_nodes_map, node_to_outputs_map;
        std::unordered_set<size_t> non_trt_node_index;

        // Initialize non_trt_node_index with all node indices
        for (size_t i = 0; i < ort_nodes.size(); ++i)
        {
            non_trt_node_index.insert(i);
        }

        size_t id = 0;
        int subgraph_index = 0;

        // Process TensorRT subgraphs
        for (const auto& group : supported_nodes_vector)
        {
            if (!group.first.empty())
            {
                // Create unique name for this TRT kernel
                std::string node_name =
                    "TRTKernel_" + std::to_string(model_hash) + "_" + std::to_string(subgraph_index);

                if (node_to_index_map.find(node_name) == node_to_index_map.end())
                {
                    index_to_node_map[id] = node_name;
                    node_to_index_map[node_name] = id++;
                }

                // Avoid GetSubgraph/GetGraphView here: each Ort::Graph view owns
                // its OrtGraph* and calls ReleaseGraph on destruction, which corrupts
                // the parent graph's internal data when called repeatedly across many
                // subgraphs. Compute external I/O directly from node data instead.
                //
                // All node outputs (including subgraph-internal intermediates) are
                // collected here. This is an over-approximation relative to the old
                // subgraph_view.GetOutputs() which returned only externally-consumed
                // tensors, but it is harmless: internal tensors produced by TRT nodes
                // cannot appear as inputs to other nodes at the same graph level, so
                // they will never be found in input_to_nodes_map and cannot suppress
                // a real cycle edge.
                std::unordered_set<std::string> subgraph_all_outputs;
                for (const auto& index : group.first)
                {
                    if (index < ort_nodes.size())
                    {
                        for (const auto& vi : ort_nodes[index].GetOutputs())
                        {
                            if (vi && !std::string(vi.GetName()).empty())
                                subgraph_all_outputs.insert(vi.GetName());
                        }
                    }
                }
                // External inputs: node inputs not produced within this subgraph.
                // GetImplicitInputs() is intentionally omitted here: implicit inputs
                // come from an outer scope and cannot be produced by a sibling node at
                // this graph level, so they cannot participate in a sibling-level cycle.
                for (const auto& index : group.first)
                {
                    if (index < ort_nodes.size())
                    {
                        for (const auto& vi : ort_nodes[index].GetInputs())
                        {
                            if (vi && subgraph_all_outputs.find(vi.GetName()) == subgraph_all_outputs.end())
                                input_to_nodes_map[vi.GetName()].insert(node_name);
                        }
                    }
                }
                for (const auto& name : subgraph_all_outputs)
                {
                    node_to_outputs_map[node_name].insert(name);
                }

                // Remove TensorRT nodes from non_trt_node_index
                for (const auto& index : group.first)
                {
                    non_trt_node_index.erase(index);
                }
                subgraph_index++;
            }
        }

        // Add non TensorRT nodes to the maps
        for (const auto& index : non_trt_node_index)
        {
            if (index < ort_nodes.size())
            {
                auto node = ort_nodes[index];
                std::string node_name = node.GetName();

                if (node_to_index_map.find(node_name) == node_to_index_map.end())
                {
                    index_to_node_map[id] = node_name;
                    node_to_index_map[node_name] = id++;
                }

                // Process inputs
                for (auto input : node.GetInputs())
                {
                    const OrtValueInfo* in_ptr = (const OrtValueInfo*)(input);
                    if (in_ptr == nullptr)
                        continue;  // optional unconnected input
                    std::string in_name = input.GetName();
                    if (in_name.empty())
                        continue;
                    input_to_nodes_map[in_name].insert(node_name);
                }

                // Process implicit inputs
                for (auto input : node.GetImplicitInputs())
                {
                    const OrtValueInfo* in_ptr = (const OrtValueInfo*)(input);
                    if (in_ptr == nullptr)
                        continue;  // optional unconnected implicit input
                    std::string in_name = input.GetName();
                    if (in_name.empty())
                        continue;
                    input_to_nodes_map[in_name].insert(node_name);
                }

                // Process outputs
                for (auto output : node.GetOutputs())
                {
                    const OrtValueInfo* out_ptr = (const OrtValueInfo*)(output);
                    if (out_ptr == nullptr)
                        continue;  // optional unconnected output (e.g., GRU/LSTM Y sequence output)
                    std::string out_name = output.GetName();
                    if (out_name.empty())
                        continue;
                    node_to_outputs_map[node_name].insert(out_name);
                }
            }
        }

        // Create adjacency list
        size_t graph_size = node_to_index_map.size();
        std::list<size_t>* adjacency_map = new std::list<size_t>[graph_size];

        for (const auto& node : node_to_outputs_map)
        {
            for (auto iter = node.second.begin(); iter != node.second.end(); ++iter)
            {
                const auto& loc = input_to_nodes_map.find(*iter);
                if (loc != input_to_nodes_map.end())
                {
                    size_t parent_node_index = node_to_index_map.find(node.first)->second;
                    for (const auto& child_node : loc->second)
                    {
                        size_t child_node_index = node_to_index_map.find(child_node)->second;
                        adjacency_map[parent_node_index].push_back(child_node_index);
                    }
                }
            }
        }

        // Check cycle in the graph
        bool* visited = new bool[graph_size];
        bool* st = new bool[graph_size];
        for (size_t i = 0; i < graph_size; ++i)
        {
            visited[i] = false;
            st[i] = false;
        }

        std::vector<size_t> cycles;
        bool has_cycle = false;
        for (size_t i = 0; i < graph_size; ++i)
        {
            if (FindCycleHelper(i, adjacency_map, visited, st, cycles))
            {
                has_cycle = true;
                cycle_detected = true;
                break;
            }
        }

        // Remove TensorRT subgraph from the supported node list if it's part of the cycle
        if (has_cycle && remove_cycles)
        {
            for (size_t i = 0; i < cycles.size(); ++i)
            {
                auto loc = index_to_node_map.find(cycles[i]);
                if (loc != index_to_node_map.end() && loc->second.find("TRTKernel") != std::string::npos)
                {
                    // Find which subgraph this corresponds to
                    for (size_t sg_idx = 0; sg_idx < supported_nodes_vector.size(); ++sg_idx)
                    {
                        std::string sg_name = "TRTKernel_" + std::to_string(model_hash) + "_" + std::to_string(sg_idx);
                        if (sg_name == loc->second)
                        {
                            supported_nodes_vector.erase(supported_nodes_vector.begin() + sg_idx);
                            trt_cycle = true;
                            break;
                        }
                    }
                    if (trt_cycle)
                        break;
                }
            }
        }

        delete[] adjacency_map;
        delete[] visited;
        delete[] st;
    }

    return cycle_detected;
}

//
// TensorrtRtxExecutionProvider implementation
//
TensorrtRtxExecutionProvider::TensorrtRtxExecutionProvider(TensorrtRtxExecutionProviderFactory& factory,
                                                           const std::string& name,
                                                           const OrtSessionOptions& session_options,
                                                           const OrtLogger& logger)
    : OrtEp{}
    , ApiPtrs{static_cast<const ApiPtrs&>(factory)}
    , factory_{factory}
    , name_{name}
    , session_options_{session_options}
    , logger_{logger}
{
    // Set OrtEp interface function pointers. ort_version_supported reflects the
    // version negotiated with the host (not compile-time) so a single DLL can
    // serve any ORT host in [kMinSupportedOrtApiVersion, ORT_API_VERSION].
    ort_version_supported = NegotiatedOrtApiVersion();
    GetName = GetNameImpl;
    GetKernelRegistry = GetKernelRegistryImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
    GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;
    OnRunStart = OnRunStartImpl;
    OnRunEnd = OnRunEndImpl;
    // Initialize the execution provider.

    // initializing logging
    auto ort_status = ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                ("Plugin EP has been created with name " + name_).c_str(), ORT_FILE,
                                                __LINE__, __FUNCTION__);
    // ignore status for now
    (void)ort_status;

    // Initialize global API pointers
    g_ort_api = &ort_api;
    g_ep_api = &ep_api;
    g_model_editor_api = &model_editor_api;
    g_logger = &logger_;

    // Register this EP's logger with the TRT singleton so it always points to a live instance.
    GetTensorrtRtxLogger(false).set_ort_logger(&logger_, &ort_api);

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // We extract those EP options to create a new "provider options" key-value map.
    std::string lowercase_ep_name = name_.c_str();
    std::transform(lowercase_ep_name.begin(), lowercase_ep_name.end(), lowercase_ep_name.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    std::string key_prefix = "ep." + lowercase_ep_name + ".";

    // Get all the provider options as session config from sesson
    ProviderOptions provider_options;

    // Get the provider options from all the config entries in session option
    OrtKeyValuePairs* key_value_pairs = nullptr;
    ort_api.GetSessionOptionsConfigEntries(&session_options, &key_value_pairs);

    const char* const* keys = nullptr;
    const char* const* values = nullptr;
    size_t num_entries = 0;
    ort_api.GetKeyValuePairs(key_value_pairs, &keys, &values, &num_entries);

    for (size_t i = 0; i < num_entries; ++i)
    {
        // only gets ep provider options
        if (strncmp(keys[i], key_prefix.c_str(), key_prefix.size()) == 0)
        {
            std::string key_str = keys[i];
            const char* value = values[i];
            provider_options[key_str.substr(key_prefix.size())] = value;
        }
    }

    info_ = TensorrtRtxExecutionProviderInfo::FromProviderOptions(provider_options);

    for (size_t i = 0; i < num_entries; ++i)
    {
        const char* key = keys[i];
        const char* value = values[i];

        if (strncmp(key, kOrtSessionOptionEpContextEnable, strlen(kOrtSessionOptionEpContextEnable)) == 0)
        {
            if (strcmp(value, "1") == 0)
            {
                info_.dump_ep_context_model = true;
            }
            else
            {
                info_.dump_ep_context_model = false;
            }
        }
        else if (strncmp(key, kOrtSessionOptionEpContextFilePath, strlen(kOrtSessionOptionEpContextFilePath)) == 0)
        {

            info_.ep_context_file_path = value;
        }
        // The "session.compile_only" config entry is set internally by OrtCompileAPI::CompileModel()
        // (microsoft/onnxruntime PR #28503). Signals that this session is for compilation only and
        // will never run inference — skip GPU deserialization and execution context creation.
        //
        // The named constant kOrtSessionOptionCompileOnly was added to the public ORT header in
        // ORT_API_VERSION 27 (ORT release 1.27 / main). For older SDKs (1.24/1.25/1.26 ->
        // ORT_API_VERSION 24/25/26) the constant is absent, so we match the literal string
        // instead — the runtime contract is the same. On older ORT versions CompileModel()
        // never sets this flag, so the compile-only branch is reachable only when the host ORT
        // itself contains #28503; the gate here is purely a build-time concession.
#if ORT_API_VERSION >= 27
        else if (strncmp(key, kOrtSessionOptionCompileOnly, strlen(kOrtSessionOptionCompileOnly)) == 0)
#else
        else if (strncmp(key, "session.compile_only", strlen("session.compile_only")) == 0)
#endif
        {
            info_.compile_only_mode = (strcmp(value, "1") == 0);
        }
        else if (strncmp(key, kOrtSessionOptionEpContextEmbedMode, strlen(kOrtSessionOptionEpContextEmbedMode)) == 0)
        {
            auto embed_mode = (value[0] == '\0') ? -1 : std::stoi(value);

            if (embed_mode == -1)
            {
                if (info_.dump_ep_context_model)
                {
                    embed_mode = 0;
                }
                else
                {
                    embed_mode = 1;
                }
            }

            if (0 <= embed_mode && embed_mode < 2)
            {
                info_.ep_context_embed_mode = embed_mode;
            }
            else
            {
                std::string message = "Invalid " + std::string(kOrtSessionOptionEpContextEmbedMode) + " must 0 or 1";
                Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
            }
        }
    }
    ort_api.ReleaseKeyValuePairs(key_value_pairs);

    device_id_ = info_.device_id;

    if (!info_.has_user_compute_stream)
    {
        // If the app is passing in a compute stream, it already has initialized cuda and created a context.
        // Calling cudaSetDevice() will set the default context in the current thread
        // which may not be compatible with the stream created by the app.
        CUDA_CALL_THROW(cudaSetDevice(device_id_));
    }

    cudaDeviceProp prop;
    CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device_id_));
    auto cc = prop.major * 10 + prop.minor;
    if (!(cc == 86 || cc == 89 || cc >= 120))
    {
        std::string message = "[NvTensorRTRTX EP] The execution provider only supports RTX devices with compute "
                              "capabilities 86, 89, 120 and above";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    compute_capability_ = GetComputeCapability(prop);

    // Set cuda_graph_enable_ before stream initialization so we can use the member variable consistently
    cuda_graph_enable_ = info_.cuda_graph_enable;

    if (info_.has_user_compute_stream)
    {
        external_stream_ = true;
        stream_ = static_cast<cudaStream_t>(info_.user_compute_stream);
    }
    else
    {
        external_stream_ = false;
        CUDA_CALL_THROW(cudaStreamCreate(&stream_));
    }
    compute_stream_context_ = GetCudaStreamContextOrThrow(stream_);
    if (external_stream_ && compute_stream_context_ == nullptr)
    {
        THROW("[NvTensorRTRTX EP] user_compute_stream has no associated CUDA context");
    }

    std::string profile_min_shapes, profile_max_shapes, profile_opt_shapes;

    auto enable_engine_cache_for_ep_context_model = [this]()
    {
        if (info_.dump_ep_context_model && info_.ep_context_embed_mode == 0)
        {
            engine_cache_enable_ = true;
        }
    };

    max_partition_iterations_ = info_.max_partition_iterations;
    min_subgraph_size_ = info_.min_subgraph_size;
    max_workspace_size_ = info_.max_workspace_size;
    max_shared_mem_size_ = info_.max_shared_mem_size;
    dump_subgraphs_ = info_.dump_subgraphs;
    weight_stripped_engine_enable_ = info_.weight_stripped_engine_enable;
    if (weight_stripped_engine_enable_)
    {
        const std::string message =
            "[NvTensorRTRTX EP] nv_weight_stripped_engine_enable_experimental is an EXPERIMENTAL feature "
            "(weightless EPContext refit / weight-stripped engine). It is opt-in, not enabled by default, and "
            "may change or be removed in a future release.";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    weight_streaming_budget_ = info_.weight_streaming_budget;
    if (weight_streaming_budget_.IsEnabled() && cuda_graph_enable_)
    {
        const std::string message = "[NvTensorRTRTX EP] Weight streaming is not compatible with CUDA graph replay. "
                                    "Disabling CUDA graph for this session.";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        cuda_graph_enable_ = false;
    }

    // make runtime cache path absolute and create directory if it doesn't exist
    if (!info_.runtime_cache_path.empty())
    {
        // Boundary conversion: provider-option std::string (UTF-8) -> std::filesystem::path (wide on Windows).
        // Must wrap via ToPathString so std::filesystem::path's narrow ctor does not re-decode the bytes via ACP.
        std::filesystem::path abs_path =
            utils::FileSystemUtils::GetAbsolutePath(std::filesystem::path(ToPathString(info_.runtime_cache_path)));
        std::string error_msg;

        if (!utils::FileSystemUtils::CreateDirectoryRecursive(abs_path, error_msg))
        {
            std::string message = "[NvTensorRTRTX EP] The runtime cache directory could not be created at: " +
                                  PathToUTF8String(abs_path.native()) + ". " + error_msg +
                                  ". Runtime cache is disabled.";
            Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
        }
        else
        {
            runtime_cache_ = abs_path;
        }
    }

    onnx_model_folder_path_ = info_.onnx_model_folder_path;
    onnx_model_bytestream_ = info_.onnx_bytestream;
    onnx_model_bytestream_size_ = info_.onnx_bytestream_size;

    if ((onnx_model_bytestream_ != nullptr && onnx_model_bytestream_size_ == 0) ||
        (onnx_model_bytestream_ == nullptr && onnx_model_bytestream_size_ != 0))
    {
        std::string message = "When providing either 'trt_onnx_bytestream_size' or "
                              "'trt_onnx_bytestream' both have to be provided";
        Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
    }
    onnx_external_data_bytestream_ = info_.external_data_bytestream;
    onnx_external_data_bytestream_size_ = info_.external_data_bytestream_size;
    if ((onnx_external_data_bytestream_ != nullptr && onnx_external_data_bytestream_size_ == 0) ||
        (onnx_external_data_bytestream_ == nullptr && onnx_external_data_bytestream_size_ != 0))
    {
        std::string message = "When providing either 'trt_external_data_bytestream_size' or "
                              "'trt_external_data_bytestream' both have to be provided";
        Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
    }
    detailed_build_log_ = info_.detailed_build_log;
    dump_ep_context_model_ = info_.dump_ep_context_model;
    ep_context_file_path_ = info_.ep_context_file_path;
    ep_context_embed_mode_ = info_.ep_context_embed_mode;
    compile_only_mode_ = info_.compile_only_mode;

    enable_engine_cache_for_ep_context_model();
    cache_prefix_ = info_.engine_cache_prefix;

    // use a more global cache if given
    engine_decryption_enable_ = info_.engine_decryption_enable;
    if (engine_decryption_enable_)
    {
        engine_decryption_lib_path_ = info_.engine_decryption_lib_path;
    }
    force_sequential_engine_build_ = info_.force_sequential_engine_build;
    sparsity_enable_ = info_.sparsity_enable;
    auxiliary_streams_ = info_.auxiliary_streams;
    // Caller-provided TensorRT auxiliary streams (user_aux_stream_array). When present, they are bound on
    // the execution context at run time so TensorRT does not create its own context/streams. Require a
    // positive length (nv_length_aux_stream_array) as well, so we never forward the default -1 count to
    // setAuxStreams(); this also keeps the enqueue-time guards safe by construction.
    if (info_.user_aux_stream_array != nullptr && auxiliary_streams_ > 0)
    {
        external_aux_streams_ = true;
        aux_streams_ = reinterpret_cast<cudaStream_t*>(info_.user_aux_stream_array);
    }
    profile_min_shapes = info_.profile_min_shapes;
    profile_max_shapes = info_.profile_max_shapes;
    profile_opt_shapes = info_.profile_opt_shapes;

    /*
     * Parse explicit min/max/opt profile shapes from provider options.
     *
     * The format of min/max/opt profile shapes is defined as below:
     * "input1:dim1xdim2...,input2:dim1xdim2...,...,input1:dim3xdim4...,input2:dim3xdim4...,..."
     *
     * (Note: if multiple shapes with same input name are specified, TRT EP will consider them as multiple profiles.
     *  Please refer to ParserProfileShapes() for more details)
     *
     */
    bool status = true;
    if (status)
    {
        status = ParseProfileShapes(profile_min_shapes, profile_min_shapes_);
        if (!status)
        {
            profile_min_shapes_.clear();
            std::string message = "[NvTensorRTRTX EP] The format of provider option 'trt_profile_min_shapes' is wrong, "
                                  "please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
    }
    if (status)
    {
        status = ParseProfileShapes(profile_max_shapes, profile_max_shapes_);
        if (!status)
        {
            profile_max_shapes_.clear();
            std::string message = "[NvTensorRTRTX EP] The format of provider option 'trt_profile_max_shapes' is wrong, "
                                  "please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
    }
    if (status)
    {
        status = ParseProfileShapes(profile_opt_shapes, profile_opt_shapes_);
        if (!status)
        {
            profile_opt_shapes_.clear();
            std::string message = "[NvTensorRTRTX EP] The format of provider option 'trt_profile_opt_shapes' is wrong, "
                                  "please follow the format of 'input1:dim1xdimd2...,input2:dim1xdim2...,...'";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
    }

    if (status)
    {
        status = ValidateProfileShapes(profile_min_shapes_, profile_max_shapes_, profile_opt_shapes_);
        if (!status)
        {
            std::string message =
                "[NvTensorRTRTX EP] Profile shapes validation failed. Make sure the provider options "
                "'nv_profile_min_shapes', 'nv_profile_max_shapes' and 'nv_profile_opt_shapes' have same input name and "
                "number of profile. "
                "NvTensorRTRTX EP will implicitly create optimization profiles based on input tensor for you.";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            profile_min_shapes_.clear();
            profile_max_shapes_.clear();
            profile_opt_shapes_.clear();
        }
    }

    // cuda_graph_enable_ is set earlier before stream initialization
    multi_profile_enable_ = info_.multi_profile_enable;
    op_types_to_exclude_ = info_.op_types_to_exclude;
    multi_rotary_cache_concat_offset_ = info_.multi_rotary_cache_concat_offset;

    profiling_enable_ = info_.enable_profiling;
    if (profiling_enable_)
    {
        if (cuda_graph_enable_)
        {
            const std::string message = "[NvTensorRTRTX EP] Profiling is not compatible with CUDA graph replay. "
                                        "Disabling CUDA graph for this session.";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            cuda_graph_enable_ = false;
        }
        profiling_output_file_ = info_.profiling_output_file.empty() ? TrtRtxProfiler::GenerateOutputFilePath()
                                                                     : info_.profiling_output_file;
        profiler_ = std::make_unique<TrtRtxProfiler>();
    }

    // Validate setting
    if (max_partition_iterations_ <= 0)
    {
        max_partition_iterations_ = 1000;
    }
    if (min_subgraph_size_ <= 0)
    {
        min_subgraph_size_ = 1;
    }

    // Note: Previously this block auto-created a directory when ep_context_file_path_
    // had no extension. That was wrong: SetEpContextBinaryInformation(dir, stem) produces
    // ep_context_file_path_ = "dir/stem" (no extension), which is a file stem, not a
    // directory. Creating it as a directory caused the engine cache to be placed in a
    // stem-named subfolder instead of the requested dir. Directory semantics are now
    // expressed only via trailing separator (handled in GetPathOrParentPathOfCtxModel).

    // If dump_ep_context_model_ is enabled, TRT EP forces cache_path_ to be the relative path of ep_context_file_path_.
    // For example,
    //    - original cache path = "engine_cache_dir" -> new cache path = "./context_model_dir/engine_cache_dir"
    //    - original cache path = ""                 -> new cache path = "./context_model_dir"
    // The new cache path will be saved as the "ep_cache_context" node attritue of the EP context node.
    // For security reason, it needs to make sure the engine cache is saved inside context model directory.
    if (dump_ep_context_model_)
    {
        // TODO: not sure if this is still needed
        engine_cache_enable_ = true;
        std::string cache_path_utf8 = PathToUTF8String(cache_path_.native());
        if (IsAbsolutePath(cache_path_utf8))
        {
            std::string message =
                "In the case of dumping context model and for security purpose, the trt_engine_cache_path should be "
                "set with a relative path, but it is an absolute path:  " +
                cache_path_utf8;
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
        if (IsRelativePathToParentPath(cache_path_utf8))
        {
            std::string message = "In the case of dumping context model and for security purpose, The "
                                  "trt_engine_cache_path has '..', it's not allowed to point outside the directory.";
            Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }

        // Engine cache relative path to context model directory.
        // It's used when dumping the "ep_cache_context" node attribute (stored as UTF-8 text).
        engine_cache_relative_path_to_context_model_dir_ = cache_path_utf8;

        // Make cache_path_ to be the relative path of ep_context_file_path_
        cache_path_ =
            GetPathOrParentPathOfCtxModel(std::filesystem::path(ToPathString(ep_context_file_path_))) / cache_path_;
    }

    if (engine_decryption_enable_)
    {
        LIBTYPE handle = OPENLIB(engine_decryption_lib_path_.c_str());
        if (handle == nullptr)
        {
            std::string message = "NvTensorRTRTX EP could not open shared library from " + engine_decryption_lib_path_;

            Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
        }
        engine_decryption_ = (int (*)(const char*, char*, size_t*))LIBFUNC(handle, "decrypt");
        engine_encryption_ = (int (*)(const char*, char*, size_t))LIBFUNC(handle, "encrypt");
        if (engine_decryption_ == nullptr)
        {
            CLOSELIB(handle);
            std::string message = "NvTensorRTRTX EP could not find decryption function in shared library from " +
                                  engine_decryption_lib_path_;
            Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
        }

        // Retain the library handle for the EP's lifetime: engine_decryption_ and
        // engine_encryption_ point into it. Released in the destructor.
        engine_decryption_lib_handle_ = static_cast<void*>(handle);
    }

    // cuda graph:
    // cudaStreamSynchronize() is not allowed in cuda graph capture.
    //
    // external stream:
    // If user provides "external" cuda stream, only this cuda stream will be used even if multiple threads are running
    // InferenceSession.Run() concurrently. So, no need to synchronize different streams after enqueueV3.
    if (external_stream_)
    {
        sync_stream_after_enqueue_ = false;
    }

    {
        auto lock = GetApiLock();
        ScopedCudaContext compute_stream_context(compute_stream_context_);
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(GetTensorrtRtxLogger(detailed_build_log_)));

        // Force synchronous GPU allocation if requested: wrap the device's existing BFC arena
        // (device_allocators[device_id], cudaMalloc/cudaFree) in a GpuSyncAllocator and install it,
        // so all GPU memory acquired by the runtime (and the engines/contexts it deserializes) goes
        // through the arena instead of TensorRT RTX's default cudaMallocAsync path. Created here,
        // next to its use; sync_gpu_allocator_ being non-null is the single source of truth for
        // whether the sync path is enabled (it is also installed on the builder in GetBuilder).
        if (runtime_ && info_.use_sync_gpu_allocator)
        {
            OrtAllocator* device_arena = factory_.GetOrCreateDeviceArena(static_cast<uint32_t>(device_id_));
            if (device_arena != nullptr)
            {
                sync_gpu_allocator_ = std::make_unique<trt_rtx_ep::GpuSyncAllocator>(device_arena);
                runtime_->setGpuAllocator(sync_gpu_allocator_.get());
                std::string msg = "[NvTensorRTRTX EP] Using synchronous GPU allocator (GpuSyncAllocator); "
                                  "TensorRT RTX async allocation (cudaMallocAsync) is disabled.";
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                            msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            }
            else
            {
                std::string msg = "[NvTensorRTRTX EP] nv_use_sync_gpu_allocator was requested but the device arena "
                                  "could not be created; falling back to the default allocator.";
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                            msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            }
        }
    }

    trt_version_ = getInferLibVersion();
    CUDA_CALL_THROW(cudaRuntimeGetVersion(&cuda_version_));

    std::string temp_str = "[NvTensorRTRTX EP] TensorRT version is " + std::to_string(trt_version_) +
                           " and CUDA version is " + std::to_string(cuda_version_);
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, temp_str.c_str(),
                                                ORT_FILE, __LINE__, __FUNCTION__));

    // Path-typed members (std::filesystem::path) MUST be rendered via PathToUTF8String(path.native())
    // before stream insertion. The default ostream<<path overload invokes MSVC STL's
    // _Convert_wide_to_narrow, which uses strict CP_ACP encoding and throws std::system_error for
    // any wide char with no CP_ACP mapping (e.g. U+0141 = Ł). On phase-2 Case A that throw was
    // marked EXCEPTION_NONCONTINUABLE downstream and terminated the host process with
    // STATUS_STACK_BUFFER_OVERRUN. PathToUTF8String routes through WideCharToMultiByte(CP_UTF8, ...)
    // which preserves all BMP chars losslessly. Diagnostics-only — does not affect engine I/O.
    std::ostringstream oss;
    oss << "[NvTensorRTRTX EP] Nv provider options: "
        << "device_id: " << device_id_ << ", nv_max_partition_iterations: " << max_partition_iterations_
        << ", nv_min_subgraph_size: " << min_subgraph_size_ << ", nv_max_workspace_size: " << max_workspace_size_
        << ", nv_dump_subgraphs: " << dump_subgraphs_
        << ", nv_weight_stripped_engine_enable_experimental: " << weight_stripped_engine_enable_
        << ", nv_weight_streaming_budget: " << weight_streaming_budget_.requested_value
        << ", nv_onnx_model_folder_path: " << onnx_model_folder_path_
        << ", nv_engine_decryption_enable: " << engine_decryption_enable_
        << ", nv_engine_decryption_lib_path: " << engine_decryption_lib_path_
        << ", nv_force_sequential_engine_build: " << force_sequential_engine_build_
        << ", nv_sparsity_enable: " << sparsity_enable_ << ", nv_auxiliary_streams: " << auxiliary_streams_
        << ", enable_cuda_graph: " << cuda_graph_enable_ << ", nv_dump_ep_context_model: " << dump_ep_context_model_
        << ", nv_ep_context_file_path: " << ep_context_file_path_
        << ", nv_ep_context_embed_mode: " << ep_context_embed_mode_ << ", nv_cache_prefix: " << cache_prefix_
        << ", nv_onnx_model_bytestream_size_: " << onnx_model_bytestream_size_
        << ", nv_onnx_external_bytestream_size_: " << onnx_external_data_bytestream_size_
        << ", nv_op_types_to_exclude: " << op_types_to_exclude_ << ", nv_enable_profiling: " << profiling_enable_
        << (profiling_enable_ ? (std::string(", nv_profiling_output_file: ") + profiling_output_file_) : "")
        << ", nv_runtime_cache_path: " << PathToUTF8String(runtime_cache_.native());
    temp_str = oss.str();
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, temp_str.c_str(),
                                                ORT_FILE, __LINE__, __FUNCTION__));
}

TensorrtRtxExecutionProvider::~TensorrtRtxExecutionProvider()
{
    ScopedCudaContextNoThrow compute_stream_context(compute_stream_context_);

    // Deregister this EP's logger from the TRT singleton before destroying any TRT
    // objects (their destructors may still log). clear_ort_logger is a no-op if a
    // newer EP has already replaced the pointer.
    GetTensorrtRtxLogger(false).clear_ort_logger(&logger_);

    // Explicitly destroy TensorRT objects in the correct order to avoid crashes
    // Order matters: contexts -> engines -> networks -> builders -> runtime

    // 0. Clear compute states first (they may reference contexts)
    compute_states_.clear();
    compute_states_for_ep_context_.clear();

    // 1. Destroy execution contexts first (they depend on engines)
    contexts_.clear();

    // 2. Destroy engines (they depend on runtime)
    engines_.clear();

    // 3. Destroy networks (they depend on builders)
    networks_.clear();

    // 4. Destroy builders
    builders_.clear();

    // 5. Destroy the single builder instance
    builder_.reset();

    // 6. Destroy runtime last
    trt_rtx_runtime_.reset();
    runtime_.reset();

    // 6.5 Release the synchronous GPU allocator only after every TRT object that used it
    //     (contexts, engines, builders, runtime) has been destroyed above.
    sync_gpu_allocator_.reset();

    // 7. Destroy the CUDA stream if we created it
    if (!external_stream_ && stream_ != nullptr)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // 7b. Destroy DDS output allocators BEFORE releasing the allocator they borrow.
    // Each OutputAllocator's destructor calls alloc_->Free(alloc_, ...). This map is
    // otherwise destroyed implicitly AFTER this destructor body (member order), i.e.
    // after step 8 has released alloc_ -> heap use-after-free.
    dds_output_allocator_maps_.clear();

    // 8. Release allocator
    if (alloc_ != nullptr)
    {
        ort_api.ReleaseAllocator(alloc_);
    }

    // 9. Unload the engine decryption/encryption library if one was loaded
    if (engine_decryption_lib_handle_ != nullptr)
    {
        CLOSELIB(engine_decryption_lib_handle_);
        engine_decryption_lib_handle_ = nullptr;
    }
}

const char* ORT_API_CALL TensorrtRtxExecutionProvider::GetNameImpl(const OrtEp* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return nullptr;
    }
    const auto* ep = static_cast<const TensorrtRtxExecutionProvider*>(this_ptr);
    return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL TensorrtRtxExecutionProvider::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr, _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept
{
    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);

    *kernel_registry = nullptr;

    // Get the cached kernel registry from parent factory to avoid recreating the kernel registry for every EP instance.
    RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(kernel_registry));
    return nullptr;
}

const char* ORT_API_CALL
TensorrtRtxExecutionProvider::GetCompiledModelCompatibilityInfoImpl(OrtEp* this_ptr, const OrtGraph* graph) noexcept
{
    if (this_ptr == nullptr)
    {
        return nullptr;
    }
    auto* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);

    ep->compatibility_info_cache_.clear();
    if (graph != nullptr)
    {
        size_t num_nodes = 0;
        OrtStatus* status = ep->ort_api.Graph_GetNumNodes(graph, &num_nodes);
        if (status != nullptr)
        {
            ep->ort_api.ReleaseStatus(status);
        }
        else if (num_nodes > 0)
        {
            std::vector<const OrtNode*> nodes(num_nodes);
            status = ep->ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size());
            if (status != nullptr)
            {
                ep->ort_api.ReleaseStatus(status);
            }
            else
            {
                size_t ep_context_node_count = 0;
                for (const auto* node : nodes)
                {
                    const char* op_type = nullptr;
                    OrtStatus* op_status = ep->ort_api.Node_GetOperatorType(node, &op_type);
                    if (op_status != nullptr)
                    {
                        ep->ort_api.ReleaseStatus(op_status);
                        continue;
                    }
                    if (op_type != nullptr && std::string(op_type) == EPCONTEXT_OP)
                    {
                        ++ep_context_node_count;
                        if (ep_context_node_count > 1)
                        {
                            // TODO: Combine compatibility info for multiple EPContext nodes and
                            //       validate using a worst-case policy instead of returning empty.
                            return ep->compatibility_info_cache_.c_str();
                        }
                    }
                }
            }
        }
    }

    if (!ep->engine_headers_.empty())
    {
        ep->compatibility_info_cache_ = ep->engine_headers_.begin()->second;
    }

    return ep->compatibility_info_cache_.c_str();
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProvider::GetCapabilityImpl(
    OrtEp* this_ptr, const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) noexcept
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] GetCapabilityImpl: this_ptr is null");
    }
    if (graph == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] GetCapabilityImpl: graph is null");
    }
    if (graph_support_info == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] GetCapabilityImpl: graph_support_info is null");
    }

    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);
    const OrtApi& ort_api = ep->ort_api;
    auto ort_graph = Ort::ConstGraph(graph);

    // Extract model path from the graph
    PathString model_path_str = ort_graph.GetModelPath();
    if (!model_path_str.empty())
    {
        std::string model_path_utf8 = PathToUTF8String(model_path_str);
        size_t copy_len = (std::min)(model_path_utf8.size(), sizeof(ep->model_path_) - 1);
        std::memcpy(ep->model_path_, model_path_utf8.c_str(), copy_len);
        ep->model_path_[copy_len] = '\0';
    }
    else
    {
        ep->model_path_[0] = '\0';
    }
    // Early return if the model has unsupported input/output data types
    for (const auto input : ort_graph.GetInputs())
    {
        const auto tp = input.TypeInfo();
        if (tp != nullptr)
        {
            if (tp.GetONNXType() != ONNX_TYPE_TENSOR)
            {
                std::string message = "[NvTensorRTRTX EP] Unsupported ONNX type for input node: " + input.GetName();
                OrtStatus* log_status =
                    ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, message.c_str(),
                                              ORT_FILE, __LINE__, __FUNCTION__);
                if (log_status)
                {
                    ort_api.ReleaseStatus(log_status);
                }
                return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
            }

            auto ts_info = tp.GetTensorTypeAndShapeInfo();
            if (ts_info.GetDimensionsCount() > 0)
            {
                auto data_type = ts_info.GetElementType();
                if (!IsSupportedInputOutputDataType(data_type))
                {
                    std::string message = "[NvTensorRTRTX EP] Unsupported data type " + GetDataTypeName(data_type) +
                                          " for input node: " + input.GetName();
                    OrtStatus* log_status =
                        ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                    if (log_status)
                    {
                        ort_api.ReleaseStatus(log_status);
                    }
                    return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
                }
            }
        }
    }

    for (const auto output : ort_graph.GetOutputs())
    {
        const auto tp = output.TypeInfo();
        if (tp != nullptr)
        {
            if (tp.GetONNXType() != ONNX_TYPE_TENSOR)
            {
                std::string message = "[NvTensorRTRTX EP] Unsupported ONNX type for output node: " + output.GetName();
                OrtStatus* log_status =
                    ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, message.c_str(),
                                              ORT_FILE, __LINE__, __FUNCTION__);
                if (log_status)
                {
                    ort_api.ReleaseStatus(log_status);
                }
                return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
                ;
            }

            auto ts_info = tp.GetTensorTypeAndShapeInfo();
            if (ts_info.GetDimensionsCount() > 0)
            {
                auto data_type = ts_info.GetElementType();
                if (!IsSupportedInputOutputDataType(data_type))
                {
                    std::string message = "[NvTensorRTRTX EP] Unsupported data type " + GetDataTypeName(data_type) +
                                          " for output node: " + output.GetName();
                    OrtStatus* log_status =
                        ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                    if (log_status)
                    {
                        ort_api.ReleaseStatus(log_status);
                    }
                    return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
                }
            }
        }
    }

    size_t number_of_ort_nodes = 0;
    RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &number_of_ort_nodes));

    std::vector<const OrtNode*> nodes(number_of_ort_nodes);
    RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    // generate unique kernel name for TRT graph
    HashValue model_hash = TRTGenerateId(graph, std::to_string(ep->trt_version_), std::to_string(ep->cuda_version_));

    // If there are "EPContext" contrib op nodes, just add them as single-node subgraphs here,
    // since TRT RTX EP can fetch precompiled engine info directly from those nodes
    // without needing to parse the graph or build a TRT engine.
    bool has_context_node = false;

    for (const auto& node : ort_graph.GetNodes())
    {
        // Skip null nodes defensively; ORT graph iterators can yield empty wrappers.
        const OrtNode* node_raw = static_cast<const OrtNode*>(node);
        if (node_raw == nullptr)
        {
            continue;
        }

        if (node.GetOperatorType() == EPCONTEXT_OP)
        {
            // Only claim EPContext nodes that belong to this EP.
            // If the "source" attribute is present and doesn't match our EP name, skip the node.
            //
            // NOTE: The Ort C++ wrapper for Node_GetAttributeByName returns IsOK()==true
            // for attributes that are NOT present on the node, leaving `source_attr` as a
            // null OrtOpAttr*. Calling GetType() on that null pointer causes an access
            // violation. Guard with an explicit validity check to support backward
            // compatibility with legacy EPContext models that don't set a "source" attribute.
            Ort::ConstOpAttr source_attr;
            if (node.GetAttributeByName(SOURCE.c_str(), source_attr).IsOK() &&
                static_cast<const OrtOpAttr*>(source_attr) != nullptr &&
                source_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING)
            {
                std::string source_value;
                if (source_attr.GetValue<std::string>(source_value).IsOK() && source_value != ep->name_)
                {
                    continue;
                }
            }

            // Create fusion options for the EPContext node. Use the cached
            // negotiated version from the OrtEp base (set in the constructor)
            // since GetCapabilityImpl is noexcept and NegotiatedOrtApiVersion()
            // can throw.
            OrtNodeFusionOptions node_fusion_options = {};
            node_fusion_options.ort_version_supported = ep->ort_version_supported;
            // Weightless refit needs the model's weight initializers kept on the fused node so
            // CreateEPContextNode can preserve them on the EPContext node (and, at load, so the
            // replay path can read them by name). Drop them in the ordinary (non-weightless) case.
            node_fusion_options.drop_constant_initializers = !ep->weight_stripped_engine_enable_;

            const OrtNode* node_ptr = node;
            RETURN_IF_ERROR(
                ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, &node_ptr, 1, &node_fusion_options));
            has_context_node = true;
        }
    }

    // return early if context nodes where found
    if (has_context_node)
    {
        return nullptr;
    }

    auto get_exclude_ops_set = [&](std::string node_list_to_exclude) -> std::set<std::string>
    {
        std::set<std::string> set;
        if (!node_list_to_exclude.empty())
        {
            std::stringstream node_list(node_list_to_exclude);
            std::string node;
            while (std::getline(node_list, node, ','))
            {
                set.insert(node);
            }
        }
        return set;
    };

    auto exclude_ops_set = get_exclude_ops_set(ep->op_types_to_exclude_);

    SubGraphCollection_t parser_nodes_vector, supported_nodes_vector;
    bool new_subgraph = true;

    // for regular onnx nodes, get supported node list from TensorRT parser
    std::vector<size_t> nodes_vector(number_of_ort_nodes);
    std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
    /* Iterate all the nodes and exclude the node if:
     *   1. It's a control flow op and its subgraph(s) is not fully TRT eligible.
     *   2. It's a DDS op.
     *   3. It has unsupported data types.
     */
    for (size_t index = 0; index < nodes.size(); index++)
    {
        const OrtNode* ort_node_ptr = nodes[index];
        Ort::ConstNode ort_node{ort_node_ptr};  // Wrap raw pointer in Ort::ConstNode
        bool supported_node = true;
        std::string op_type = ort_node.GetOperatorType();

        /* If current node is control flow op, we take different approach based on following four cases:
         *
         * (1) control flow op is supported by TRT, and its subgraphs are all supported by TRT. Assign this node to TRT.
         * (2) control flow op is supported by TRT, but not all its subgraphs supported by TRT. Don't assign this node
         * to TRT. (3) control flow op is not supported by TRT, but its subgraphs all supported by TRT. Don't assign
         * this node to TRT. (4) control flow op is not supported by TRT, and not all its subgraphs supported by TRT.
         * Don't assign this node to TRT.
         *
         * For cases 2, 3, 4, even though the control flow op is not assigned to TRT, any portion of its subgraphs that
         * can run in TRT will be still fused and assigned to TRT EP.
         */
        if (ep->control_flow_op_set_.find(op_type) != ep->control_flow_op_set_.end())
        {
            auto supported_control_flow_op = [&](Ort::ConstNode node) -> bool
            {
                auto sub_graphs = node.GetSubgraphs();
                if (sub_graphs.size() != 0)
                {
                    for (const auto& attr_subgraph : sub_graphs)
                    {
                        if (attr_subgraph.sub_graph.GetNodes().size() == 0)
                        {
                            continue;
                        }
                        if (!ep->AllNodesAssignedToSpecificEP(attr_subgraph.sub_graph, ep->name_))
                        {
                            return false;
                        }
                        return true;
                    }
                }
                return true;
            };
            supported_node = supported_control_flow_op(ort_node);
        }

        // Exclude any ops, if applicable
        if (exclude_ops_set.find(op_type) != exclude_ops_set.end())
        {
            supported_node = false;
        }

        if (supported_node)
        {
            if (!CheckNodeDataTypes(ort_node))
            {
                supported_node = false;
                std::string message = "[NvTensorRTRTX EP] Node '" + ort_node.GetName() + "' (OpType: " + op_type +
                                      ") excluded due to unsupported data types";
                Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            }
        }

        if (supported_node)
        {
            if (new_subgraph)
            {
                parser_nodes_vector.emplace_back();
                parser_nodes_vector.back().second = false;
                new_subgraph = false;
            }
            parser_nodes_vector.back().first.emplace_back(index);
        }
        else
        {
            new_subgraph = true;
        }
    }

    bool early_termination = false;
    supported_nodes_vector =
        ep->GetSupportedList(parser_nodes_vector, 0, ep->max_partition_iterations_, graph, &early_termination);

    if (early_termination)
    {
        supported_nodes_vector.clear();
        return nullptr;
    }

    // Remove subgraphs if its size is less than the predefined minimal size
    for (auto it = supported_nodes_vector.begin(); it != supported_nodes_vector.end();)
    {
        const size_t subgraph_size = it->first.size();
        if (subgraph_size < ep->min_subgraph_size_)
        {
            it = supported_nodes_vector.erase(it);
        }
        else
        {
            ++it;
        }
    }
    // Detect and remove cycles from supported node list
    ep->DetectTensorRTGraphCycles(supported_nodes_vector, Ort::ConstGraph(graph), model_hash);

    // Consolidate supported node list
    if (supported_nodes_vector.size() > 1)
    {
        nodes_vector.clear();
        for (const auto& group : supported_nodes_vector)
        {
            if (!group.first.empty())
            {
                nodes_vector.insert(nodes_vector.end(), group.first.begin(), group.first.end());
            }
        }
        SubGraphCollection_t consolidated_supported_nodes_vector = {{nodes_vector, true}};
        if (ep->DetectTensorRTGraphCycles(consolidated_supported_nodes_vector, Ort::ConstGraph(graph), model_hash,
                                          false))
        {
            std::string message = "[NvTensorRTRTX EP] TensorRT nodes are not consolidated because graph will have "
                                  "cycles after consolidation";
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        }
        else
        {
            std::string message = "[NvTensorRTRTX EP] TensorRT nodes are consolidated into one subgraph";
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            supported_nodes_vector = consolidated_supported_nodes_vector;
        }
    }

    // Handle the case where the graph is subgraph of control flow op.
    // The purpose is to make control flow op as well as its subgraphs run on TRT.
    // Here we need to check whether subgraph is fully supported by TRT and don't fuse the nodes of the subgraph until
    // control flow op level.
    if (ep->IsSubGraphOfControlFlowOp(graph) && ep->IsSubGraphFullySupported(graph, supported_nodes_vector))
    {

        bool all_subgraphs_are_supported = true;

        // "If" control flow op has two subgraph bodies, "then" body and "else" body respectively.
        // Check its parent node's another subgraph to see whether that subgraph is also fully supported by TRT.
        Ort::ConstNode parent_node = ort_graph.GetParentNode();
        if (parent_node.GetOperatorType() == "If")
        {
            all_subgraphs_are_supported = false;
            SubGraphCollection_t subgraph_supported_nodes_vector;

            std::vector<Ort::AttrNameSubgraph> attr_name_subgraphs = parent_node.GetSubgraphs();
            for (const auto& attr_name_subgraph : attr_name_subgraphs)
            {
                auto subgraph = attr_name_subgraph.sub_graph;
                const OrtGraph* subgraph_raw_pointer = subgraph;
                if (subgraph_raw_pointer != graph)
                {
                    size_t num_subgraph_nodes = 0;
                    RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(subgraph, &num_subgraph_nodes));

                    // Another subgraph of "If" control flow op has no nodes.
                    // In this case, TRT EP should consider this empty subgraph is fully supported by TRT.
                    if (num_subgraph_nodes == 0)
                    {
                        all_subgraphs_are_supported = true;
                        break;
                    }
                    // Another subgraph of "If" control flow op has been parsed by GetCapability before and all
                    // subgraph's nodes assigned to TRT EP.
                    else if (ep->AllNodesAssignedToSpecificEP(subgraph, ep->name_))
                    {
                        all_subgraphs_are_supported = true;
                        break;
                    }
                    // Another subgraph of "If" control flow has been parsed by GetCapability and not all subgraph's
                    // nodes assigned to TRT EP. (Note: GetExecutionProviderType() returns "" meaning node has not yet
                    // been assigned to any EPs)
                    else if (!ep->AllNodesAssignedToSpecificEP(subgraph, ""))
                    {
                        all_subgraphs_are_supported = false;
                        break;
                    }

                    std::vector<size_t> subgraph_nodes_vector(num_subgraph_nodes);
                    std::iota(std::begin(subgraph_nodes_vector), std::end(subgraph_nodes_vector), 0);
                    SubGraphCollection_t parser_subgraph_nodes_vector = {{subgraph_nodes_vector, false}};
                    bool subgraph_early_termination = false;

                    // Another subgraph of "If" control flow has not yet been parsed by GetCapability.
                    subgraph_supported_nodes_vector =
                        ep->GetSupportedList(parser_subgraph_nodes_vector, 0, ep->max_partition_iterations_, subgraph,
                                             &subgraph_early_termination);
                    all_subgraphs_are_supported =
                        ep->IsSubGraphFullySupported(subgraph, subgraph_supported_nodes_vector);
                    break;
                }
            }
        }

        if (all_subgraphs_are_supported)
        {
            // We want the subgraph nodes to be assigned to TRT EP but don't want them to be fused until later at the
            // control flow op level. Simply request the subgraph nodes with a single ComputeCapability for each with no
            // MetaDef (i.e. what the default implementation for IExecutionProvider::GetCapability does).
            for (const auto& group : supported_nodes_vector)
            {
                if (!group.first.empty())
                {
                    for (const auto& index : group.first)
                    {
                        const OrtNode* supported_node = nodes[index];
                        RETURN_IF_ERROR(
                            ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, supported_node));
                    }
                }
            }
            std::string message = "[TensorRT EP] Whole graph will run on TensorRT execution provider";
            Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                            message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

            return nullptr;
        }
    }

    int number_of_trt_nodes = 0;
    for (const auto& group : supported_nodes_vector)
    {
        if (!group.first.empty())
        {
            std::vector<const OrtNode*> supported_nodes;
            supported_nodes.reserve(group.first.size());

            for (const auto& index : group.first)
            {
                const OrtNode* supported_node = nodes[index];

                supported_nodes.push_back(supported_node);
            }

            // Create (optional) fusion options for the supported nodes to fuse.
            // See note at the EPContext call site above: GetCapabilityImpl is
            // noexcept, so use the cached version instead of NegotiatedOrtApiVersion().
            OrtNodeFusionOptions node_fusion_options = {};
            node_fusion_options.ort_version_supported = ep->ort_version_supported;
            // Weightless refit needs the model's weight initializers kept on the fused node so
            // CreateEPContextNode can preserve them on the EPContext node (and, at load, so the
            // replay path can read them by name). Drop them in the ordinary (non-weightless) case.
            node_fusion_options.drop_constant_initializers = !ep->weight_stripped_engine_enable_;

            RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, supported_nodes.data(),
                                                                         supported_nodes.size(), &node_fusion_options));
            number_of_trt_nodes += static_cast<int>(group.first.size());
        }
    }

    const size_t number_of_subgraphs = supported_nodes_vector.size();
    if (number_of_trt_nodes == 0)
    {
        std::string message = "[TensorRT EP] No graph will run on TensorRT execution provider";
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    else if (number_of_trt_nodes == nodes.size())
    {
        std::string message = "[TensorRT EP] Whole graph will run on TensorRT execution provider";
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }
    else
    {
        std::string message =
            "[TensorRT EP] Graph is partitioned and number of subgraphs running on TensorRT execution provider is " +
            std::to_string(number_of_subgraphs);
        Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                        message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProvider::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                                                  _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                                                  _Out_writes_all_(count)
                                                                      OrtNodeComputeInfo** node_compute_infos,
                                                                  _Out_writes_(count)
                                                                      OrtNode** ep_context_nodes) noexcept
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CompileImpl: this_ptr is null");
    }
    if (count > 0)
    {
        if (graphs == nullptr)
        {
            return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                           "[NvTensorRTRTX EP] CompileImpl: graphs array is null");
        }
        if (fused_nodes == nullptr)
        {
            return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                           "[NvTensorRTRTX EP] CompileImpl: fused_nodes array is null");
        }
        if (node_compute_infos == nullptr)
        {
            return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                           "[NvTensorRTRTX EP] CompileImpl: node_compute_infos array is null");
        }
    }

    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);

    const OrtApi& ort_api = ep->ort_api;

    for (size_t fused_node_idx = 0; fused_node_idx < count; fused_node_idx++)
    {
        auto fused_node = fused_nodes[fused_node_idx];
        size_t num_node_inputs = 0;
        RETURN_IF_ERROR(ort_api.Node_GetNumInputs(fused_node, &num_node_inputs));

        std::vector<const OrtValueInfo*> node_inputs(num_node_inputs);
        RETURN_IF_ERROR(ort_api.Node_GetInputs(fused_node, node_inputs.data(), node_inputs.size()));

        std::unordered_map<std::string, size_t> input_map;
        input_map.reserve(num_node_inputs);
        for (size_t i = 0; i < num_node_inputs; i++)
        {
            const OrtValueInfo* value_info = node_inputs[i];
            if (value_info == nullptr)
                continue;
            const char* name = nullptr;
            RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &name));

            input_map.emplace(name, i);
        }

        size_t num_node_outputs = 0;
        RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(fused_node, &num_node_outputs));

        std::vector<const OrtValueInfo*> node_outputs(num_node_outputs);
        RETURN_IF_ERROR(ort_api.Node_GetOutputs(fused_node, node_outputs.data(), node_outputs.size()));

        std::unordered_map<std::string, size_t> output_map;
        output_map.reserve(num_node_outputs);
        for (size_t i = 0; i < num_node_outputs; i++)
        {
            const OrtValueInfo* value_info = node_outputs[i];
            if (value_info == nullptr)
                continue;
            const char* name = nullptr;
            RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &name));

            output_map.emplace(name, i);
        }

        if (EPContextNodeReader::GraphHasCtxNode(graphs[fused_node_idx], ort_api))
        {
            RETURN_IF_ERROR(ep->CreateNodeComputeInfoFromPrecompiledEngine(this_ptr, graphs[fused_node_idx], fused_node,
                                                                           input_map, output_map,
                                                                           &node_compute_infos[fused_node_idx]));
        }
        else
        {
            RETURN_IF_ERROR(ep->CreateNodeComputeInfoFromGraph(this_ptr, graphs[fused_node_idx], fused_node, input_map,
                                                               output_map, &node_compute_infos[fused_node_idx],
                                                               &ep_context_nodes[fused_node_idx]));
        }
    }

    return nullptr;
}

void ORT_API_CALL TensorrtRtxExecutionProvider::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                                            OrtNodeComputeInfo** node_compute_infos,
                                                                            size_t num_node_compute_infos) noexcept
{
    (void)this_ptr;
    // Security check: validate node_compute_infos array is not null before iterating
    if (node_compute_infos == nullptr || num_node_compute_infos == 0)
    {
        return;
    }
    for (size_t i = 0; i < num_node_compute_infos; i++)
    {
        delete node_compute_infos[i];
    }
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProvider::CreateSyncStreamForDeviceImpl(
    _In_ OrtEp* this_ptr, _In_ const OrtMemoryDevice* memory_device, _Outptr_ OrtSyncStreamImpl** stream) noexcept
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] CreateSyncStreamForDeviceImpl: this_ptr is null");
    }
    if (memory_device == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] CreateSyncStreamForDeviceImpl: memory_device is null");
    }
    if (stream == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] CreateSyncStreamForDeviceImpl: stream output is null");
    }

    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);

    // A per-session OrtSyncStreamImpl can be created here if the session options affect the implementation.
    // Logging of any issues should use logger_ which is the session logger.

    // we only create streams for the default device memory.
    if (auto mem_type = ep->factory_.ep_api.MemoryDevice_GetMemoryType(memory_device);
        mem_type != OrtDeviceMemoryType_DEFAULT)
    {
        std::string error = "Invalid OrtMemoryDevice. Expected OrtDeviceMemoryType_DEFAULT(0). Got ";
        error += std::to_string(mem_type);
        return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
    }

    auto device_id = ep->factory_.ep_api.MemoryDevice_GetDeviceId(memory_device);

    std::unique_ptr<TensorrtRtxSyncStreamImpl> sync_stream;
    RETURN_IF_ERROR(TensorrtRtxSyncStreamImpl::Create(ep->factory_, ep, device_id, nullptr, sync_stream));
    *stream = sync_stream.release();

    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProvider::OnRunStartImpl(_In_ OrtEp* this_ptr,
                                                                     _In_ const OrtRunOptions* run_options) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] OnRunStartImpl: this_ptr is null");
    }
    // Note: run_options can be null in some scenarios, so we don't enforce null check on it

    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);

    // TODO: Handle CUDA graph annotation ID if CUDA graph support is needed
    // For now, we only handle multi-profile

    if (ep->multi_profile_enable_)
    {
        // Get profile index from run options
        auto* profile_index_str =
            ep->ort_api.GetRunConfigEntry(run_options, onnxruntime::tensorrt_rtx::run_option_names::kProfileIndex);

        if (profile_index_str != nullptr)
        {
            try
            {
                ep->trt_profile_index_ = std::stoi(profile_index_str);
            }
            catch (...)
            {
                std::string message = "[NvTensorRTRTX EP] Failed to parse profile index: ";
                message += profile_index_str;
                Ort::ThrowOnError(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
            }
        }
    }

    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProvider::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                                                   _In_ const OrtRunOptions* run_options,
                                                                   _In_ bool sync_stream) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] OnRunEndImpl: this_ptr is null");
    }
    // Note: run_options can be null in some scenarios, so we don't enforce null check on it

    TensorrtRtxExecutionProvider* ep = static_cast<TensorrtRtxExecutionProvider*>(this_ptr);

    // Synchronize stream if requested and we're using an external stream
    if (sync_stream && ep->external_stream_)
    {
        cudaError_t err = cudaStreamSynchronize(ep->stream_);
        if (err != cudaSuccess)
        {
            std::string error_msg = "[NvTensorRTRTX EP] Failed to synchronize CUDA stream: ";
            error_msg += cudaGetErrorString(err);
            return ep->ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
    }

    // Check if memory arena shrinkage is requested via RunOptions.
    // The config value is a semicolon-separated list of device strings, e.g. "gpu:0" or "cpu:0;gpu:0".
    if (run_options != nullptr)
    {
        const char* shrinkage_cfg = ep->ort_api.GetRunConfigEntry(
            run_options, onnxruntime::tensorrt_rtx::run_option_names::kMemoryArenaShrinkage);

        if (shrinkage_cfg != nullptr && shrinkage_cfg[0] != '\0')
        {
            std::string target = "gpu:" + std::to_string(ep->device_id_);
            std::string cfg(shrinkage_cfg);
            std::istringstream tokens(cfg);
            std::string token;

            while (std::getline(tokens, token, ';'))
            {
                if (token == target)
                {
                    RETURN_IF_ERROR(ep->factory_.ShrinkCudaMempoolAllocators(static_cast<uint32_t>(ep->device_id_)));
                    break;
                }
            }
        }
    }

    if (ep->profiling_enable_ && ep->profiler_)
    {
        std::lock_guard<std::mutex> lock(ep->tensorrt_rtx_mu_);
        ep->profiler_->FlushToFile(ep->profiling_output_file_);
    }

    return nullptr;
}

OrtStatus* TensorrtRtxExecutionProvider::RefitEngineImpl(
    _In_ const std::filesystem::path& onnx_model_filename, _In_ const std::filesystem::path& onnx_model_folder_path,
    _In_ bool path_check, _In_ const void* onnx_model_bytestream, _In_ size_t onnx_model_bytestream_size,
    _In_ const void* onnx_external_data_bytestream, _In_ size_t onnx_external_data_bytestream_size,
    _In_ nvinfer1::ICudaEngine* trt_engine, _In_ bool detailed_build_log) noexcept
{

    bool refit_from_file = onnx_model_bytestream == nullptr && onnx_model_bytestream_size == 0;
    bool refit_with_external_data = onnx_external_data_bytestream != nullptr && onnx_external_data_bytestream_size != 0;
    bool refit_complete = false;
    std::filesystem::path onnx_model_path = onnx_model_folder_path;
    if (refit_from_file)
    {
        if (!onnx_model_filename.empty())
        {
            onnx_model_path /= onnx_model_filename;
        }
        if (onnx_model_path.empty())
        {
            std::string error_msg = "The ONNX model was not provided as path. Please use provide an ONNX bytestream to "
                                    "enable refitting the weightless engine.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
        else
        {
            // check if file path to ONNX is legal
            std::string onnx_model_path_utf8 = PathToUTF8String(onnx_model_path.native());
            if (path_check && IsAbsolutePath(onnx_model_path_utf8))
            {
                std::string error_msg = "For security purpose, the ONNX model path should be set with a relative path, "
                                        "but it is an absolute path: " +
                                        onnx_model_path_utf8;
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            if (path_check && IsRelativePathToParentPath(onnx_model_path_utf8))
            {
                std::string error_msg = "The ONNX model path has '..'. For security purpose, it's not allowed to point "
                                        "outside the directory.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }

            if (!(std::filesystem::exists(onnx_model_path) && std::filesystem::is_regular_file(onnx_model_path)))
            {
                std::string error_msg = "The ONNX model " + onnx_model_path_utf8 + " does not exist.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
        }
    }

    // weight-stripped engine refit logic
    TensorrtRtxLogger& trt_logger = GetTensorrtRtxLogger(detailed_build_log_);
    auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*trt_engine, trt_logger));
    auto parser_refitter =
        std::unique_ptr<nvonnxparser::IParserRefitter>(nvonnxparser::createParserRefitter(*refitter, trt_logger));

    // New refit APIs
    if (refit_with_external_data)
    {
#if TRT_MAJOR_RTX > 1 || TRT_MINOR_RTX >= 1
        // A valid model bytestream must be passed.
        if (refit_from_file)
        {
            std::string error_msg =
                "NvTensorRTRTX EP's refit with external data must be called with a valid ONNX model bytestream";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        if (!parser_refitter->loadModelProto(onnx_model_bytestream, onnx_model_bytestream_size, nullptr))
        {
            std::string error_msg =
                "NvTensorRTRTX EP's IParserRefitter could not load model from provided onnx_model_bytestream";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        // Extract weight information from the Refitter.
        int required_weights = refitter->getAllWeights(0, nullptr);
        std::vector<char const*> refit_names_prealocated(required_weights);
        refitter->getAllWeights(required_weights, refit_names_prealocated.data());
        // Log: Refitter requires N weights
        std::unordered_set<std::string> refit_names(std::make_move_iterator(refit_names_prealocated.begin()),
                                                    std::make_move_iterator(refit_names_prealocated.end()));

        // Vectors to keep track of data pointers.
        std::vector<std::string> names;
        names.reserve(required_weights);
        std::vector<const char*> bytes;
        bytes.reserve(required_weights);
        std::vector<int64_t> sizes;
        sizes.reserve(required_weights);

        ONNX_NAMESPACE::ModelProto onnx_model;
        google::protobuf::RepeatedPtrField<onnx::TensorProto>* allInitializers_byte_stream = nullptr;

        // Reconstruct onnx model view.
        const auto onnx_model_view = std::string((const char*)onnx_model_bytestream, onnx_model_bytestream_size);
        if (!onnx_model.ParseFromString(onnx_model_view))
        {
            std::string error_msg = "The provided ONNX bytestream to refit could not be parsed.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        // Extract graph and initializer information.
        auto* graph = onnx_model.mutable_graph();
        allInitializers_byte_stream = graph->mutable_initializer();
        // Log: Initializers that were found

        // Loop through all initializers
        int missing_initializer_data = 0;
        for (int initializer_idx = 0; initializer_idx < allInitializers_byte_stream->size(); ++initializer_idx)
        {
            auto& proto = allInitializers_byte_stream->at(initializer_idx);
            auto& proto_name = proto.name();
            if (refit_names.find(proto_name) != refit_names.end())
            {
                if (proto.has_data_location())
                {
                    if (proto.data_location() == onnx::TensorProto_DataLocation_EXTERNAL)
                    {
                        // Default values for reading into external_data blob.
                        int64_t offset = 0;
                        size_t length = 0;
                        auto* external_data = proto.mutable_external_data();
                        const std::string kOffset = "offset", kLength = "length";
                        for (int entry_idx = 0; entry_idx < external_data->size(); ++entry_idx)
                        {
                            auto* current_key = external_data->at(entry_idx).mutable_key();
                            auto* current_value = external_data->at(entry_idx).mutable_value();
                            if (*current_key == kOffset && !current_value->empty())
                            {
                                offset = std::stoll(*current_value);
                            }
                            else if (*current_key == kLength && !current_value->empty())
                            {
                                length = std::stoul(*current_value);
                            }
                        }
                        names.push_back(proto.name());
                        bytes.push_back(static_cast<const char*>(onnx_external_data_bytestream) + offset);
                        sizes.push_back(length);
                    }
                    else
                    {
                        std::string error_msg =
                            "[NvTensorRTRTX EP] Proto: " + proto_name +
                            " expected to have external datalocation, but default datalocation was provided instead.";
                        return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
                    }
                }
                else if (proto.has_raw_data())
                {
                    auto& raw_data = proto.raw_data();
                    names.push_back(proto.name());
                    bytes.push_back(raw_data.c_str());
                    sizes.push_back(raw_data.size());
                }
                else
                {
                    // Log warning: Proto has no raw nor external data
                    ++missing_initializer_data;
                }
            }
            else
            {
                // Log verbose: Initializer was not marked as refittable
            }
        }
        if (missing_initializer_data)
        {
            std::string error_msg = "[NvTensorRTRTX EP] RefitEngine is missing " +
                                    std::to_string(missing_initializer_data) + " initializers.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        // Load extracted initializers into the parser
        if (!names.empty())
        {
            // Log: Number of initializers submitted to refitter
            for (size_t i = 0; i < names.size(); i++)
            {
                bool refloadInit = parser_refitter->loadInitializer(names[i].c_str(), bytes[i], sizes[i]);
                if (!refloadInit)
                {
                    std::string error_msg = "NvTensorRTRTX EP's IParserRefitter could not refit deserialized "
                                            "weight-stripped engine with weights contained in the provided bytestream";
                    return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
                }
            }
        }
        // Perform refit.
        if (!parser_refitter->refitModelProto())
        {
            std::string error_msg = "NvTensorRTRTX EP's IParserRefitter refitModelProto() failed with the provided "
                                    "external data bytestream.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
        refit_complete = true;
#else
        std::string error_msg = "Refit with external data is only supported on TensorRT RTX 1.1.x.x and above.";
        return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
#endif
    }

    // If new refit flow was not completed, then fallback to refit_from_file.
    if (!refit_complete)
    {
        if (refit_from_file)
        {
            // Log: Refitting from file on disk.
            // NOTE: leaf encoding for TRT-RTX 1.5.0.110 refitFromFile is provisionally ACP (.string())
            // pending the probe at Fix/refit_probe_spec.md. The upstream path-typed input is correct
            // regardless of leaf choice; if the probe shows UTF-8, swap .string().c_str() to
            // PathToUTF8String(onnx_model_path.native()).c_str() here as a follow-up.
            if (!parser_refitter->refitFromFile(onnx_model_path.string().c_str()))
            {
                std::string error_msg = "NvTensorRTRTX EP's IParserRefitter could not refit deserialized "
                                        "weight-stripped engine with weights contained in: " +
                                        PathToUTF8String(onnx_model_path.native());
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
        }
        else
        {
            // Log: Refitting from byte array
            if (!parser_refitter->refitFromBytes(onnx_model_bytestream, onnx_model_bytestream_size))
            {
                std::string error_msg = "NvTensorRTRTX EP's IParserRefitter could not refit deserialized "
                                        "weight-stripped engine with weights contained in the provided bytestream";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
        }
    }
    if (refitter->refitCudaEngine())
    {
        // Log: Successfully refitted the weight-stripped engine
    }
    else
    {
        std::string error_msg = "NvTensorRTRTX EP's IRefitter could not refit deserialized weight-stripped engine with "
                                "weights contained in: " +
                                PathToUTF8String(onnx_model_path.native());
        return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    return nullptr;
}

// Weightless capture path — uses the TRT-RTX ONNX parser refit-observer API (IRefitterObserver /
// RefitRecord / setRefitObserver), guaranteed by the enforced TensorRT-RTX >= 1.6 minimum
// (nv_includes.h / cmake). The TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED guard is retained defensively.
#if TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED
namespace
{

//! \brief Deep-copies every RefitRecord emitted during a refit pass into an owned
//!        trt_rtx_ep::WeightlessRefitRecord list. RefitRecord's pointer fields are parser-owned
//!        and only valid for the duration of onRefittableWeight(), hence the copy.
class WeightlessRefitCaptureObserver : public nvonnxparser::IRefitterObserver
{
public:
    std::vector<trt_rtx_ep::WeightlessRefitRecord> records;

    void onRefittableWeight(nvonnxparser::RefitRecord const& src) noexcept override
    {
        // `src` is the parser-owned RefitRecord (camelCase SDK fields); `copy` is our owned
        // trt_rtx_ep::WeightlessRefitRecord (snake_case). Keeping the two variable names distinct avoids
        // the identical field names colliding.
        trt_rtx_ep::WeightlessRefitRecord copy;
        copy.trt_name = src.trtName != nullptr ? src.trtName : "";
        copy.kind = static_cast<int32_t>(src.kind);
        copy.onnx_dtype = src.onnxDtype;
        copy.trt_dtype = static_cast<int32_t>(src.trtDtype);
        copy.count = src.count;
        copy.epsilon = src.epsilon;
        copy.source_onnx_names.reserve(static_cast<size_t>(src.nbSources));
        for (int32_t i = 0; i < src.nbSources; ++i)
        {
            copy.source_onnx_names.emplace_back(src.sourceOnnxNames[i] != nullptr ? src.sourceOnnxNames[i] : "");
        }
        if (src.fixedData != nullptr && src.fixedDataSize > 0)
        {
            const uint8_t* bytes = static_cast<const uint8_t*>(src.fixedData);
            copy.fixed_data.assign(bytes, bytes + src.fixedDataSize);
        }
        records.push_back(std::move(copy));
    }
};

}  // namespace

OrtStatus* TensorrtRtxExecutionProvider::CaptureWeightlessRefitTable(
    const nvinfer1::IHostMemory& serialized_engine, const std::string& serialized_model_proto,
    const std::vector<TensorrtUserWeights>& user_weights, bool detailed_build_log,
    std::vector<trt_rtx_ep::WeightlessRefitRecord>& records) noexcept
{
    records.clear();

    TensorrtRtxLogger& trt_logger = GetTensorrtRtxLogger(detailed_build_log);

    // This capture-only engine and its refitter are discarded once the observer has fired for
    // every weight; they exist solely to drive IParserRefitter's normal refit traversal, not for
    // inference. The `serialized_engine`/execution-context path built by the caller is untouched.
    //
    // Why re-deserialize instead of capturing on the just-built engine: the EP's build path uses
    // IBuilder::buildSerializedNetwork() (see the engine build above), which returns serialized bytes
    // directly and never materializes an nvinfer1::ICudaEngine. IParserRefitter requires an IRefitter,
    // which requires a live ICudaEngine -- so there is no already-built engine object to capture on, and
    // a one-time deserialize here is the only way to obtain a refittable engine. This runs only on
    // weight-stripped (kSTRIP_PLAN) compiles and the engine is freed immediately after capture.
    std::unique_ptr<nvinfer1::ICudaEngine> capture_engine(
        runtime_->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size()));
    if (!capture_engine)
    {
        return ort_api.CreateStatus(
            ORT_EP_FAIL,
            "[NvTensorRTRTX EP] Failed to deserialize a throwaway engine to capture the weightless refit table.");
    }

    auto capture_refitter =
        std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*capture_engine, trt_logger));
    if (!capture_refitter)
    {
        return ort_api.CreateStatus(
            ORT_EP_FAIL,
            "[NvTensorRTRTX EP] createInferRefitter() returned null while capturing the weightless refit table.");
    }

    auto capture_parser_refitter = std::unique_ptr<nvonnxparser::IParserRefitter>(
        nvonnxparser::createParserRefitter(*capture_refitter, trt_logger));
    if (!capture_parser_refitter)
    {
        return ort_api.CreateStatus(
            ORT_EP_FAIL,
            "[NvTensorRTRTX EP] createParserRefitter() returned null while capturing the weightless refit table.");
    }

    WeightlessRefitCaptureObserver observer;
    capture_parser_refitter->setRefitObserver(&observer);

    if (!capture_parser_refitter->loadModelProto(serialized_model_proto.data(), serialized_model_proto.size(), nullptr))
    {
        return ort_api.CreateStatus(ORT_EP_FAIL,
                                    "[NvTensorRTRTX EP] IParserRefitter could not load the model proto while "
                                    "capturing the weightless refit table.");
    }
    for (const auto& weight : user_weights)
    {
        if (!capture_parser_refitter->loadInitializer(weight.Name(), weight.Data(), static_cast<size_t>(weight.Size())))
        {
            std::string error_msg = "[NvTensorRTRTX EP] IParserRefitter could not load initializer '" +
                                    std::string(weight.Name()) + "' while capturing the weightless refit table.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
    }
    if (!capture_parser_refitter->refitModelProto())
    {
        return ort_api.CreateStatus(ORT_EP_FAIL,
                                    "[NvTensorRTRTX EP] IParserRefitter::refitModelProto() failed while capturing "
                                    "the weightless refit table.");
    }

    records = std::move(observer.records);
    return nullptr;
}
#endif  // TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED

// Bytes required to hold `count` elements of a TRT weight dtype, used to validate that a supplied
// source buffer is large enough before it is handed to IRefitter::setNamedWeights (which reads
// `count` elements and cannot detect a short buffer -> out-of-bounds read). Sub-byte types
// (INT4/FP4) pack two elements per byte. Returns 0 for unknown/future types so callers skip the
// check (fail-open) rather than falsely reject a type this build doesn't know how to size.
//
// Malformed counts must fail *closed*, not open: a negative count, or a per-dtype byte total that
// overflows int64_t, returns INT64_MAX (not 0) so the caller's `size < need_bytes` comparison always
// rejects. Returning 0 for these would collide with the "unknown dtype -> skip" sentinel and let a
// corrupt record bypass validation. (Counts large enough to overflow are physically impossible for
// any real buffer, so rejecting them loses nothing.)
static int64_t WeightlessRequiredBytes(nvinfer1::DataType t, int64_t count)
{
    if (count < 0)
        return INT64_MAX;
    // Overflow-safe `count * n`: clamp to INT64_MAX instead of wrapping to a small/negative value or overflowing
    // size_t on a 32-bit host.
    auto mul = [](int64_t c, size_t n) -> int64_t
    {
        size_t byte_size = 0;
        return detail::TryGetWeightlessBufferByteSize(c, n, byte_size) ? static_cast<int64_t>(byte_size) : INT64_MAX;
    };
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return mul(count, 4);
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return mul(count, 2);
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kE8M0:
        return mul(count, 1);
    case nvinfer1::DataType::kINT64:
        return mul(count, 8);
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
        // 4-bit packed, two elements per byte. This form avoids overflowing `count + 1`.
        return mul(count / 2 + count % 2, 1);
    default:
        return 0;  // unknown dtype -> skip the size check
    }
}

// Weightless replay path — the per-kind switch uses nvonnxparser::RefitTransformKind, guaranteed by
// the enforced TensorRT-RTX >= 1.6 minimum (nv_includes.h / cmake). The
// TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED guard is retained defensively.
#if TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED
OrtStatus* TensorrtRtxExecutionProvider::WeightlessRefitEngineImpl(
    const std::vector<trt_rtx_ep::WeightlessRefitRecord>& records,
    const std::unordered_map<std::string, std::pair<const void*, size_t>>& weight_data_by_name,
    nvinfer1::ICudaEngine* trt_engine, bool detailed_build_log) noexcept
try
{
    TensorrtRtxLogger& trt_logger = GetTensorrtRtxLogger(detailed_build_log);
    auto refitter = std::unique_ptr<nvinfer1::IRefitter>(nvinfer1::createInferRefitter(*trt_engine, trt_logger));

    // nvinfer1::Weights only references memory, it does not copy it -- any transformed (cast or
    // batchnorm-folded) buffer must stay alive until refitCudaEngine() below runs.
    std::vector<std::vector<float>> owned_float_buffers;

    auto find_source = [&](const std::string& name, const void** data, size_t* size) -> bool
    {
        auto it = weight_data_by_name.find(name);
        if (it == weight_data_by_name.end())
        {
            return false;
        }
        *data = it->second.first;
        *size = it->second.second;
        return true;
    };

    for (const auto& record : records)
    {
        nvinfer1::Weights weights{};
        weights.type = static_cast<nvinfer1::DataType>(record.trt_dtype);
        weights.count = record.count;

        // Reject a malformed (negative) element count up front, before any per-kind byte math or vector
        // sizing below: record.count comes from the deserialized table (untrusted) and feeds
        // static_cast<size_t>(record.count) vector allocations and setNamedWeights, where a negative
        // value becomes an enormous size_t -> bad_alloc / out-of-bounds. (Overflow on large positive
        // counts is handled inside WeightlessRequiredBytes.)
        if (record.count < 0)
        {
            std::string error_msg = "[NvTensorRTRTX EP] Weightless refit record for '" + record.trt_name +
                                    "' has a negative element count (" + std::to_string(record.count) + ").";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        const auto kind = static_cast<nvonnxparser::RefitTransformKind>(record.kind);
        switch (kind)
        {
        case nvonnxparser::RefitTransformKind::kIDENTITY:
        {
            if (record.source_onnx_names.empty())
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit record for '" + record.trt_name +
                                        "' (kIDENTITY) has no source name.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            const void* data = nullptr;
            size_t size = 0;
            if (!find_source(record.source_onnx_names[0], &data, &size))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: source weight '" +
                                        record.source_onnx_names[0] + "' for '" + record.trt_name +
                                        "' was not supplied to the EPContext node.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            // Validate the supplied buffer is large enough for `count` elements of the target dtype
            // (consistency with kDOUBLE_TO_FLOAT below; setNamedWeights would otherwise OOB-read a
            // short/mismatched buffer). Skipped only for unknown dtypes (need_bytes == 0).
            const int64_t need_bytes = WeightlessRequiredBytes(weights.type, record.count);
            if (need_bytes > 0 && size < static_cast<size_t>(need_bytes))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: source weight '" +
                                        record.source_onnx_names[0] + "' for '" + record.trt_name + "' is " +
                                        std::to_string(size) + " bytes, smaller than the " +
                                        std::to_string(need_bytes) + " bytes needed for " +
                                        std::to_string(record.count) + " elements of the target dtype.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            weights.values = data;
            break;
        }
        case nvonnxparser::RefitTransformKind::kDOUBLE_TO_FLOAT:
        {
            if (record.source_onnx_names.empty())
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit record for '" + record.trt_name +
                                        "' (kDOUBLE_TO_FLOAT) has no source name.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            const void* data = nullptr;
            size_t size = 0;
            if (!find_source(record.source_onnx_names[0], &data, &size))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: source weight '" +
                                        record.source_onnx_names[0] + "' for '" + record.trt_name +
                                        "' was not supplied to the EPContext node.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            const int64_t required_bytes = WeightlessRequiredBytes(nvinfer1::DataType::kINT64, record.count);
            if (required_bytes == INT64_MAX ||
                static_cast<uint64_t>(record.count) > static_cast<uint64_t>(std::vector<float>{}.max_size()))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: kDOUBLE_TO_FLOAT element count for '" +
                                        record.trt_name + "' is too large to materialize safely.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            if (size < static_cast<size_t>(required_bytes))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: source weight '" +
                                        record.source_onnx_names[0] +
                                        "' is smaller than expected for kDOUBLE_TO_FLOAT.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            const double* src = static_cast<const double*>(data);
            std::vector<float> casted(static_cast<size_t>(record.count));
            for (int64_t i = 0; i < record.count; ++i)
            {
                casted[static_cast<size_t>(i)] = static_cast<float>(src[i]);
            }
            owned_float_buffers.push_back(std::move(casted));
            weights.values = owned_float_buffers.back().data();
            break;
        }
        case nvonnxparser::RefitTransformKind::kBATCH_NORM_FOLD_SCALE:
        case nvonnxparser::RefitTransformKind::kBATCH_NORM_FOLD_BIAS:
        {
            if (record.source_onnx_names.size() < 4)
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit record for '" + record.trt_name +
                                        "' (batchnorm fold) needs 4 sources (scale, bias, mean, variance), got " +
                                        std::to_string(record.source_onnx_names.size()) + ".";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            const void* scale_data = nullptr;
            size_t scale_size = 0;
            const void* bias_data = nullptr;
            size_t bias_size = 0;
            const void* mean_data = nullptr;
            size_t mean_size = 0;
            const void* var_data = nullptr;
            size_t var_size = 0;
            if (!find_source(record.source_onnx_names[0], &scale_data, &scale_size) ||
                !find_source(record.source_onnx_names[1], &bias_data, &bias_size) ||
                !find_source(record.source_onnx_names[2], &mean_data, &mean_size) ||
                !find_source(record.source_onnx_names[3], &var_data, &var_size))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: one or more BatchNormalization "
                                        "source weights for '" +
                                        record.trt_name + "' were not supplied to the EPContext node.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            // The fold below reads all four sources as float32 and writes a float32 result, so reject
            // non-float32 BatchNorm sources (e.g. fp16/double) that the `const float*` casts would
            // otherwise misinterpret into wrong weights. (ONNX TensorProto::FLOAT == 1; for the
            // batchnorm-fold kinds the parser reports source and result ONNX types as identical.)
            // Full fp16/double support would require converting sources to float before folding.
            if (record.onnx_dtype != 1 /* onnx float32 */)
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: BatchNormalization fold for '" +
                                        record.trt_name +
                                        "' has non-float32 sources (onnx_dtype=" + std::to_string(record.onnx_dtype) +
                                        "); only float32 is currently supported.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            // All four sources are read as `count` floats in the fold loop below; validate each is
            // large enough to avoid an out-of-bounds read on a short/mismatched buffer. Use the
            // overflow-safe WeightlessRequiredBytes (kFLOAT) for consistency with the kIDENTITY / constant
            // paths rather than static_cast<size_t>(record.count) * sizeof(float), which could wrap on a
            // 32-bit size_t (or overflow) for a corrupt oversized count and falsely pass these checks.
            const int64_t bn_need = WeightlessRequiredBytes(nvinfer1::DataType::kFLOAT, record.count);
            if (bn_need > 0 &&
                (scale_size < static_cast<uint64_t>(bn_need) || bias_size < static_cast<uint64_t>(bn_need) ||
                 mean_size < static_cast<uint64_t>(bn_need) || var_size < static_cast<uint64_t>(bn_need)))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: a BatchNormalization source "
                                        "weight for '" +
                                        record.trt_name + "' is smaller than the " + std::to_string(bn_need) +
                                        " bytes needed for " + std::to_string(record.count) + " float elements.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            const float* scale = static_cast<const float*>(scale_data);
            const float* bias = static_cast<const float*>(bias_data);
            const float* mean = static_cast<const float*>(mean_data);
            const float* variance = static_cast<const float*>(var_data);
            std::vector<float> combined_scale(static_cast<size_t>(record.count));
            for (int64_t i = 0; i < record.count; ++i)
            {
                combined_scale[static_cast<size_t>(i)] = scale[i] / std::sqrt(variance[i] + record.epsilon);
            }
            if (kind == nvonnxparser::RefitTransformKind::kBATCH_NORM_FOLD_SCALE)
            {
                owned_float_buffers.push_back(std::move(combined_scale));
                weights.values = owned_float_buffers.back().data();
            }
            else
            {
                std::vector<float> combined_bias(static_cast<size_t>(record.count));
                for (int64_t i = 0; i < record.count; ++i)
                {
                    combined_bias[static_cast<size_t>(i)] = bias[i] - mean[i] * combined_scale[static_cast<size_t>(i)];
                }
                owned_float_buffers.push_back(std::move(combined_bias));
                weights.values = owned_float_buffers.back().data();
            }
            break;
        }
        case nvonnxparser::RefitTransformKind::kCONSTANT_NODE:
        case nvonnxparser::RefitTransformKind::kCONSTANT_OF_SHAPE:
        {
            if (record.fixed_data.empty())
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit record for '" + record.trt_name +
                                        "' is missing its persisted fixed_data bytes.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            // Symmetric size check: the persisted bytes must cover `count` elements of the target dtype.
            const int64_t const_need = WeightlessRequiredBytes(weights.type, record.count);
            if (const_need > 0 && record.fixed_data.size() < static_cast<size_t>(const_need))
            {
                std::string error_msg = "[NvTensorRTRTX EP] Weightless refit record for '" + record.trt_name +
                                        "' has " + std::to_string(record.fixed_data.size()) +
                                        " persisted bytes, smaller than the " + std::to_string(const_need) +
                                        " needed for " + std::to_string(record.count) + " elements.";
                return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
            }
            weights.values = record.fixed_data.data();
            break;
        }
        default:
        {
            std::string error_msg = "[NvTensorRTRTX EP] Weightless refit: unknown RefitTransformKind (" +
                                    std::to_string(record.kind) + ") for '" + record.trt_name +
                                    "'. This EP build's refit-table format may be older/newer than the TRT-RTX "
                                    "SDK it was captured with.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
        }

        if (!refitter->setNamedWeights(record.trt_name.c_str(), weights))
        {
            std::string error_msg =
                "[NvTensorRTRTX EP] IRefitter::setNamedWeights failed for '" + record.trt_name + "'.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
    }

    if (!refitter->refitCudaEngine())
    {
        return ort_api.CreateStatus(
            ORT_EP_FAIL,
            "[NvTensorRTRTX EP] IRefitter could not refit the weight-stripped engine from the weightless refit table.");
    }

    return nullptr;
}
catch (const std::exception&)
{
    return ort_api.CreateStatus(
        ORT_EP_FAIL, "[NvTensorRTRTX EP] Weightless refit failed while materializing transformed weight buffers.");
}
catch (...)
{
    return ort_api.CreateStatus(
        ORT_EP_FAIL,
        "[NvTensorRTRTX EP] Weightless refit failed with an unknown error while materializing transformed weights.");
}
#endif  // TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED

//
// CUDA Graph related functions
//
bool TensorrtRtxExecutionProvider::IsGraphCaptureAllowed(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    // Graph capture needs the async mempool; the synchronous BFC fallback uses
    // cudaMalloc, which isn't capturable. So when the pool is unavailable (probed
    // out or latched off), run eagerly. (Also blocks new captures after a latch;
    // ORT-level capture isn't wired into compute yet, so none can outlive a latch.)
    if (!factory_.IsAsyncMempoolEnabledForDevice(static_cast<uint32_t>(device_id_)))
    {
        return false;
    }

    if (!IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id))
    {
        return false;
    }

    // Safe access to map - return false if key doesn't exist yet
    auto it = graph_id_to_run_count_.find(cuda_graph_annotation_id);
    if (it == graph_id_to_run_count_.end())
    {
        return false;  // Entry doesn't exist yet, not ready for capture
    }

    bool allowed = it->second >= min_num_runs_before_cuda_graph_capture_;
    return allowed;
}

bool TensorrtRtxExecutionProvider::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    return cuda_graph_.IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id);
}

CudaGraphAnnotation_t TensorrtRtxExecutionProvider::GetCudaGraphAnnotationId(const OrtRunOptions* run_options) const
{
    // Get CUDA graph annotation from run options using C API
    const char* graph_annotation_str =
        ort_api.GetRunConfigEntry(run_options, onnxruntime::tensorrt_rtx::run_option_names::kCudaGraphAnnotation);
    CudaGraphAnnotation_t cuda_graph_annotation_id = kCudaGraphAnnotationDefault;

    // Parse the annotation ID from string
    if (graph_annotation_str != nullptr && graph_annotation_str[0] != '\0')
    {
        try
        {
            cuda_graph_annotation_id = static_cast<CudaGraphAnnotation_t>(std::stoi(graph_annotation_str));
        }
        catch (const std::exception&)
        {
            cuda_graph_annotation_id = kCudaGraphAnnotationDefault;
        }
    }
    return cuda_graph_annotation_id;
}

void TensorrtRtxExecutionProvider::SetCurrentGraphAnnotationId(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    current_graph_annotation_id_ = cuda_graph_annotation_id;
}

CudaGraphAnnotation_t TensorrtRtxExecutionProvider::GetCurrentGraphAnnotationId() const
{
    return current_graph_annotation_id_;
}

void TensorrtRtxExecutionProvider::CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    cuda_graph_.Reset();
    cuda_graph_.CaptureBegin(cuda_graph_annotation_id);
}

void TensorrtRtxExecutionProvider::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    cuda_graph_.CaptureEnd(cuda_graph_annotation_id);
}

bool TensorrtRtxExecutionProvider::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    return cuda_graph_.IsGraphCaptured(cuda_graph_annotation_id);
}

OrtStatus* TensorrtRtxExecutionProvider::ReplayGraph(CudaGraphAnnotation_t cuda_graph_annotation_id,
                                                     bool sync_status_flag)
{
    return cuda_graph_.Replay(cuda_graph_annotation_id, sync_status_flag);
}

void TensorrtRtxExecutionProvider::IncrementRegularRunCountBeforeGraphCapture(
    CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    graph_id_to_run_count_[cuda_graph_annotation_id]++;
}

void TensorrtRtxExecutionProvider::DeleteCapturedGraph(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    graph_id_to_run_count_.erase(cuda_graph_annotation_id);
    cuda_graph_.Reset();
}

void TensorrtRtxExecutionProvider::ResetWarmupRuns(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    if (graph_id_to_run_count_.find(cuda_graph_annotation_id) == graph_id_to_run_count_.end())
    {
        return;
    }
    graph_id_to_run_count_[cuda_graph_annotation_id] = 0;
}

void TensorrtRtxExecutionProvider::HandleCudaGraphStart(cudaStream_t stream, bool require_io_binding,
                                                        CudaGraphAnnotation_t cuda_graph_annotation_id,
                                                        bool& graph_replay_on_this_run, bool& should_start_capture)
{
    graph_replay_on_this_run = false;
    should_start_capture = false;

    // Case 1: CUDA Graph capture is enabled AND IO binding is required.
    // In this case, we force graph re-capture by resetting warmup runs.
    // If a graph for this annotation ID already exists, delete it before proceeding.
    if (require_io_binding && cuda_graph_enable_)
    {
        ResetWarmupRuns(cuda_graph_annotation_id);

        if (IsGraphCaptured(cuda_graph_annotation_id))
        {
            DeleteCapturedGraph(cuda_graph_annotation_id);
        }
        // Case 2: CUDA Graph capture is enabled AND IO binding is NOT required
    }
    else if (cuda_graph_enable_ && !require_io_binding)
    {
        // If the graph is not yet captured, increment the regular run counter
        if (cuda_graph_annotation_id != kCudaGraphAnnotationSkip && !IsGraphCaptured(cuda_graph_annotation_id))
        {
            IncrementRegularRunCountBeforeGraphCapture(cuda_graph_annotation_id);
        }

        // If capture is allowed and graph not already captured,
        // set the stream and begin capture
        if (!IsGraphCaptured(cuda_graph_annotation_id) && IsGraphCaptureAllowed(cuda_graph_annotation_id))
        {
            SetCudaGraphStream(stream);
            CaptureBegin(cuda_graph_annotation_id);
            should_start_capture = true;
        }

        // If a graph is already captured for this ID, mark it for replay in this run.
        if (IsGraphCaptured(cuda_graph_annotation_id))
        {
            graph_replay_on_this_run = true;
        }
    }
}

//
// TRTRtxEpNodeComputeInfo implementation
//
TensorRtRtxEpNodeComputeInfo::TensorRtRtxEpNodeComputeInfo(TensorrtRtxExecutionProvider& ep)
    : ep(ep)
{
    ort_version_supported = NegotiatedOrtApiVersion();
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
}

OrtStatus* TensorRtRtxEpNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                         OrtNodeComputeContext* compute_context, void** compute_state)
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateStateImpl: this_ptr is null");
    }
    if (compute_context == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] CreateStateImpl: compute_context is null");
    }
    if (compute_state == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] CreateStateImpl: compute_state output is null");
    }

    auto* node_compute_info = static_cast<TensorRtRtxEpNodeComputeInfo*>(this_ptr);
    TensorrtRtxExecutionProvider& ep = node_compute_info->ep;

    std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);

    auto trt_state = ep.compute_states_[fused_node_name].get();
    if (trt_state == nullptr)
    {
        std::string message = "[NvTensorRTRTX EP] Compute state not found for fused node: " + fused_node_name;
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    *compute_state = trt_state;

    return nullptr;
}

//
// Get the shape of "shape tensor" input
//
template <typename T>
void GetShapeOfShapeTensor(Ort::ConstValue& input_tensor, void* shape_values, int shape_size, cudaStream_t stream)
{
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(shape_values, input_tensor.GetTensorData<T>(), shape_size * sizeof(T),
                                         cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
}

#define CASE_GET_INPUT_TENSOR(DATA_TYPE, SrcT)                                        \
    case DATA_TYPE:                                                                   \
    {                                                                                 \
        auto input_tensor_ptr = input_tensor.GetTensorData<SrcT>();                   \
        if (input_tensor_ptr != nullptr && elem_cnt > 0)                              \
        {                                                                             \
            data = const_cast<SrcT*>(input_tensor_ptr);                               \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, 1)); \
            data = scratch_buffers.back().get();                                      \
        }                                                                             \
        break;                                                                        \
    }

#define CASE_GET_OUTPUT_TENSOR(DATA_TYPE, SrcT)                                       \
    case DATA_TYPE:                                                                   \
    {                                                                                 \
        auto output_tensor_ptr = output_tensor.GetTensorMutableData<SrcT>();          \
        data_ptr = output_tensor_ptr;                                                 \
        if (output_tensor_ptr != nullptr && elem_cnt > 0)                             \
        {                                                                             \
            buffer = output_tensor_ptr;                                               \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            scratch_buffers.push_back(MakeUniquePtrFromOrtAllocator<void>(alloc, 1)); \
            buffer = scratch_buffers.back().get();                                    \
        }                                                                             \
        break;                                                                        \
    }

#define CASE_COPY_TENSOR(DATA_TYPE, DstT)                                                                            \
    case DATA_TYPE:                                                                                                  \
    {                                                                                                                \
        auto output_tensor_ptr = output_tensor.GetTensorMutableData<DstT>();                                         \
        if (output_tensor_ptr != nullptr && elem_cnt > 0)                                                            \
        {                                                                                                            \
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_tensor_ptr, allocator->getBuffer(), elem_cnt * sizeof(DstT), \
                                                 cudaMemcpyDeviceToDevice, stream));                                 \
        }                                                                                                            \
        break;                                                                                                       \
    }

//!
//! \brief Resolves a pointer to its device address for TensorRT binding.
//!
//! \details This helper uses cudaHostGetDevicePointer to obtain
//!          the mapped device pointer for registered host accessible memory.
//!          For device memory, returns the input unchanged.
//!
static bool IsHostAccessibleGpuTensor(const OrtValue* tensor)
{
    if (tensor == nullptr || g_ep_api == nullptr)
    {
        return false;
    }

    const OrtMemoryDevice* memory_device = g_ep_api->Value_GetMemoryDevice(tensor);
    if (memory_device == nullptr)
    {
        return false;
    }

    return g_ep_api->MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_GPU &&
           g_ep_api->MemoryDevice_GetMemoryType(memory_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE;
}

static OrtStatusPtr ResolveDevicePointer(void* ptr, bool is_host_accessible_gpu_tensor, const char* tensor_name,
                                         void** resolved_ptr)
{
    *resolved_ptr = ptr;
    if (ptr == nullptr)
    {
        return nullptr;
    }

    if (is_host_accessible_gpu_tensor)
    {
        void* device_ptr = nullptr;
        const cudaError_t err = cudaHostGetDevicePointer(&device_ptr, ptr, 0);
        if (err != cudaSuccess)
        {
            const char* cuda_error = cudaGetErrorString(err);
            (void)cudaGetLastError();
            std::string error_msg = std::string("NvTensorRTRTX EP failed to resolve host-accessible ") + " tensor '" +
                                    tensor_name + "' to a device pointer via cudaHostGetDevicePointer(): " + cuda_error;
            return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
        }

        if (device_ptr == nullptr)
        {
            std::string error_msg = std::string("NvTensorRTRTX EP failed to resolve host-accessible ") + " tensor '" +
                                    tensor_name + "' to a device pointer: cudaHostGetDevicePointer() returned null.";
            return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
        }

        *resolved_ptr = device_ptr;
    }

    return nullptr;
}

OrtStatusPtr BindContextInput(Ort::KernelContext& ctx, nvinfer1::ICudaEngine* trt_engine,
                              nvinfer1::IExecutionContext* trt_context, const char* input_name, size_t input_index,
                              std::unordered_map<std::string, std::vector<int32_t>>& shape_tensor_values,
                              std::unordered_map<std::string, std::vector<int64_t>>& shape_tensor_values_int64,
                              std::vector<AllocatorUniquePtr<void>>& scratch_buffers, OrtAllocator* alloc,
                              cudaStream_t stream, bool& skip_input_binding_allowed)
{

    auto input_tensor = ctx.GetInput(input_index);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto tensor_shapes = tensor_info.GetShape();
    const auto tensor_type = tensor_info.GetElementType();

    /*
     * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
     * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
     *
     * Examples:<br>
     * [] = 1<br>
     * [1,3,4] = 12<br>
     * [2,0,4] = 0<br>
     * [-1,3,4] = -1<br>
     */
    const auto elem_cnt = tensor_info.GetElementCount();

    if (trt_engine->isShapeInferenceIO(input_name))
    {
        // Bind "shape tensor" input buffer
        skip_input_binding_allowed = false;  // Shape tensor input binding cannot be skipped
        // The shape of the "shape tensor" is either zero dimension (scalar) or 1-dimension
        int shape_size = trt_engine->getTensorShape(input_name).nbDims == 0 ? 1 : static_cast<int>(tensor_shapes[0]);

        switch (tensor_type)
        {

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        {
            // get shape tensor value if not present
            if (shape_tensor_values.find(input_name) == shape_tensor_values.end())
            {
                auto input = std::make_unique<int32_t[]>(shape_size);
                GetShapeOfShapeTensor<int32_t>(input_tensor, input.get(), shape_size, stream);
                shape_tensor_values[input_name].resize(shape_size);
                for (int i = 0; i < shape_size; ++i)
                {
                    shape_tensor_values[input_name][i] = input[i];
                }
            }

            if (!trt_context->setTensorAddress(input_name, &shape_tensor_values[input_name][0]))
            {
                std::string error_input_name = input_name;
                std::string error_msg = "NvTensorRTRTX EP failed to call "
                                        "nvinfer1::IExecutionContext::setTensorAddress() for shape input '" +
                                        error_input_name + "'";
                return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        {
            // get shape tensor value if not present
            if (shape_tensor_values_int64.find(input_name) == shape_tensor_values_int64.end())
            {
                auto input = std::make_unique<int64_t[]>(shape_size);
                GetShapeOfShapeTensor<int64_t>(input_tensor, input.get(), shape_size, stream);
                shape_tensor_values_int64[input_name].resize(shape_size);
                for (int i = 0; i < shape_size; ++i)
                {
                    shape_tensor_values_int64[input_name][i] = input[i];
                }
            }

            if (!trt_context->setTensorAddress(input_name, &shape_tensor_values_int64[input_name][0]))
            {
                std::string error_input_name = input_name;
                std::string error_msg = "NvTensorRTRTX EP failed to call "
                                        "nvinfer1::IExecutionContext::setTensorAddress() for shape input '" +
                                        error_input_name + "'";
                return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
            }
            break;
        }
        default:
        {
            std::string error_input_name = input_name;
            std::string error_msg =
                "The data type of shape tensor should be INT32 or INT64. Please check the data type of " +
                error_input_name;
            return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
        }
        }
    }
    else
    {

        // Set shape for input tensor which is execution tensor
        nvinfer1::Dims dims = trt_context->getTensorShape(input_name);
        int nb_dims = dims.nbDims;
        for (int j = 0, end = nb_dims; j < end; ++j)
        {
            dims.d[j] = static_cast<int32_t>(tensor_shapes[j]);
        }
        if (!trt_context->setInputShape(input_name, dims))
        {
            std::string error_input_name = input_name;
            std::string error_msg =
                "NvTensorRTRTX EP failed to call nvinfer1::IExecutionContext::setInputShape() for input '" +
                error_input_name + "'";
            return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
        }

        // Bind "execution tensor" input buffer
        //
        // Note: If an engine binding is an empty tensor, it still needs a non-null memory address, and different
        // tensors should have different addresses.
        //       Therefore, in the case of empty tensor, TRT EP always allocates a dummy byte.
        //       https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#empty-tensors
        void* data = nullptr;
        const bool is_host_accessible_gpu_tensor = IsHostAccessibleGpuTensor(input_tensor);
        switch (tensor_type)
        {
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, uint16_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, uint8_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN, uint8_t)
            CASE_GET_INPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1, uint8_t)
        default:
        {
            std::string error_msg =
                "NvTensorRTRTX EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.";
            return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
        }
        }
        void* resolved_data = nullptr;
        if (auto status = ResolveDevicePointer(data, is_host_accessible_gpu_tensor, input_name, &resolved_data))
        {
            return status;
        }
        trt_context->setTensorAddress(input_name, resolved_data);
    }

    return nullptr;
}

//
// Bind Nv execution context output.
//
// Please note that the "data-dependent shape" output needs corresponding allocator provided.
//
//
// @param ctx - ORT kernel context
// @param trt_context - A pointer to Nv execution context object
// @param output_name - Output tensor name
// @param output_index - The index of the output to the ORT kernel context
// @param output_type - Data type of the output
// @param i - Output iteration index
// @param output_tensors - Output iteration index to output's ORT value
// @param dds_output_allocator_map - DDS output to its allocator
// @param scratch_buffer - The allocation buffer created by TRT EP
// @param allocator - ORT allocator
// @param buffers - It holds all the output values which are binding to TRT's execution context

OrtStatusPtr BindContextOutput(Ort::KernelContext& ctx, nvinfer1::IExecutionContext* trt_context,
                               const char* output_name, size_t output_index, size_t output_type,
                               DDSOutputAllocatorMap& dds_output_allocator_map,
                               std::vector<AllocatorUniquePtr<void>>& scratch_buffers, OrtAllocator* alloc,
                               nvinfer1::Dims& dims, void*& data_ptr, cudaStream_t stream,
                               bool& skip_output_binding_allowed)
{

    // get Output shape
    dims = trt_context->getTensorShape(output_name);
    int nb_dims = dims.nbDims;
    bool is_DDS = false;
    for (int j = 0, end = nb_dims; j < end; ++j)
    {
        if (dims.d[j] == -1)
        {
            is_DDS = true;
            break;
        }
    }

    auto known_DDS = dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end();

    // If the output tensor has data-dependent shape, TRT EP will provide an IOutputAllocator for enqueueV3 to
    // dynamically allocate memory buffer.
    // Once enqueueV3 returns, TRT EP will then bind the output allocation to ORT kernel context output.
    // (Please note that we take strategy A mentioned in
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dynamic-shaped-output,
    //  which we defer allocation until the size is known and don't call IExecution::setTensorAddress)
    //
    // Otherwise, if the shape of the output tensor is known prior to the runtime, ORT will pre-allocate memory buffer
    // for the output tensor for enqueueV3.
    if (is_DDS || known_DDS)
    {
        if (!known_DDS)
        {
            auto allocatorPtr = std::make_unique<OutputAllocator>(alloc);
            trt_context->setOutputAllocator(output_name, allocatorPtr.get());
            dds_output_allocator_map[output_name] = std::move(allocatorPtr);
            dims.nbDims = -1;    // Set to -1 to indicate that the shape is not known at this point.
            data_ptr = nullptr;  // Set data_ptr to nullptr for DDS output binding.
        }
    }
    else
    {
        auto output_tensor = ctx.GetOutput(output_index, dims.d, nb_dims);
        const auto elem_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        const bool is_host_accessible_gpu_tensor = IsHostAccessibleGpuTensor(output_tensor);
        void* buffer = nullptr;

        switch (output_type)
        {
            // below macros set data_ptr and skip_output_binding_allowed variables
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, uint16_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, uint8_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN, uint8_t)
            CASE_GET_OUTPUT_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1, uint8_t)
        default:
        {
            std::string error_msg =
                "NvTensorRTRTX EP output tensor data type: " + std::to_string(output_type) + " not supported.";
            return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
        }
        }
        void* resolved_buffer = nullptr;
        if (auto status = ResolveDevicePointer(buffer, is_host_accessible_gpu_tensor, output_name, &resolved_buffer))
        {
            return status;
        }
        trt_context->setTensorAddress(output_name, resolved_buffer);
    }

    return nullptr;
}
/**
 * @brief Determines whether I/O binding is required for TensorRT execution.
 *
 * This function optimizes TensorRT inference performance by determining when tensor
 * input/output binding operations can be skipped. Binding is an expensive operation
 * that involves setting up tensor pointers in the TensorRT execution context, so
 * avoiding unnecessary rebinding can significantly improve inference throughput.
 *
 * The function implements a three-tier decision logic:
 * 1. First run: Always requires binding to establish initial tensor mappings
 * 2. Subsequent runs with optimization allowed: Only rebind if tensors have changed
 * 3. Subsequent runs without optimization: Always rebind for safety
 *
 * @tparam TRTState The TensorRT state type (TensorrtFuncState or TensorrtShortFuncState)
 * @param trt_state Pointer to the TensorRT execution state containing tensor cache
 *                  and configuration flags
 * @param ctx ONNX Runtime kernel context providing access to current input tensors
 *
 * @return true if I/O binding is required (tensors changed or safety conditions apply),
 *         false if binding can be safely skipped (optimization enabled and tensors unchanged)
 *
 * @note This function modifies trt_state by:
 *       - Setting is_first_run to false after first execution
 *       - Caching current tensor parameters in input_tensors vector
 *       - Updating cached tensors when changes are detected
 *
 * @warning The skip_io_binding_allowed flag must be carefully managed as incorrect
 *          usage can lead to inference with stale tensor bindings and incorrect results.
 */
template <class TRTState>
static bool IsIOBindingRequired(TRTState* const trt_state, OrtKernelContext* ctx)
{

    auto ort_ctx = Ort::KernelContext(ctx);
    // Check if input tensors have changed since the last run
    // If so, we need to bind input tensors again
    bool require_io_binding = false;

    if (trt_state->is_first_run)
    {
        // If this is the first run, we always bind input tensors
        require_io_binding = true;
        auto input_tensor_count = ort_ctx.GetInputCount();
        auto output_tensor_count = ort_ctx.GetOutputCount();
        trt_state->input_tensors.resize(input_tensor_count);
        trt_state->output_tensors.resize(output_tensor_count);
        for (size_t input_index = 0; input_index < input_tensor_count; ++input_index)
        {
            const auto& input_tensor = ort_ctx.GetInput(input_index);
            const auto& tensor_info = input_tensor.GetTensorTypeAndShapeInfo();

            trt_state->input_tensors[input_index] =
                TensorParams{input_tensor.GetTensorRawData(), tensor_info.GetShape()};
        }
        trt_state->is_first_run = false;
    }
    else if (trt_state->skip_io_binding_allowed)
    {
        // If skip_io_binding_allowed is true, we can skip binding if input tensors are the same as before
        auto input_tensor_count = ort_ctx.GetInputCount();
        for (size_t input_index = 0; input_index < input_tensor_count; ++input_index)
        {
            const auto& input_tensor = ort_ctx.GetInput(input_index);
            const auto& tensor_info = input_tensor.GetTensorTypeAndShapeInfo();

            TensorParams ip_tensor{input_tensor.GetTensorRawData(), tensor_info.GetShape()};

            if (ip_tensor != trt_state->input_tensors[input_index])
            {
                require_io_binding = true;
                trt_state->input_tensors[input_index] = ip_tensor;
            }
        }
    }
    else
    {
        // If this is not the first run and skip_io_binding_allowed is false, we need to bind input tensors
        require_io_binding = true;
    }

    if (!require_io_binding)
    {
        // no need to bind inputs, but check outputs as well
        auto output_tensor_count = ort_ctx.GetOutputCount();

        for (size_t output_index = 0; output_index < output_tensor_count; ++output_index)
        {
            const auto& prev_output_tensor = trt_state->output_tensors[output_index];

            if (prev_output_tensor.dims.nbDims != -1)
            {
                const auto& new_output_tensor =
                    ort_ctx.GetOutput(output_index, prev_output_tensor.dims.d, prev_output_tensor.dims.nbDims);

                // different output tensor data means we need to bind outputs again
                if (prev_output_tensor.data != new_output_tensor.GetTensorRawData())
                {
                    require_io_binding = true;
                    break;
                }
            }
        }
    }

    return require_io_binding;
}

OrtStatusPtr BindKernelOutput(Ort::KernelContext& ctx, const OrtMemoryInfo* /*mem_info*/,
                              DDSOutputAllocatorMap& allocator_map, char const* output_name, size_t output_index,
                              size_t output_type, cudaStream_t stream)
{

    auto allocator = allocator_map[output_name].get();
    auto& shape = allocator->getOutputShape();
    auto output_tensor = ctx.GetOutput(output_index, shape);

    /*
     * Return the number of elements specified by the tensor shape (all dimensions multiplied by each other).
     * For 0 dimensions, 1 is returned. If any dimension is less than 0, the result is always -1.
     *
     * Examples:<br>
     * [] = 1<br>
     * [1,3,4] = 12<br>
     * [2,0,4] = 0<br>
     * [-1,3,4] = -1<br>
     */
    auto elem_cnt = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    /*
     * Copy output data from allocation buffer to ORT kernel context output location or
     * cast (int32 or float) -> (int64 or double) to ORT kernel context output location.
     *
     * Note:
     * 1. If the output tensor is empty tensor (i.e. any of the dimension is 0) which means element count is 0,
     *    TRT EP does not perform cuda memory copy nor cuda cast to prevent overwriting other location that might belong
     * to other tensors.
     * 2. The cudaMemcpyAsync() and cuda::Impl_Cast() (implemented as _UnaryElementWise() in cuda ep) are all async, but
     * we don't need to explicitly call cudaStreamSynchronize() after those APIs due to CUDA EP and TRT EP uses same
     * stream, and within the same stream, operations are guaranteed to be executed in order.
     */
    switch (output_type)
    {
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, float)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, uint16_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, uint16_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, bool)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, uint8_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, int8_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, uint8_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, int32_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, int64_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN, uint8_t)
        CASE_COPY_TENSOR(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1, uint8_t)
    default:
    {
        std::string error_msg =
            "NvTensorRTRTX EP output tensor data type: " + std::to_string(output_type) + " not supported.";
        return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
    }
    }

    return nullptr;
}

OrtStatus* TensorRtRtxEpNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                                     OrtKernelContext* kernel_context)
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] ComputeImpl: this_ptr is null");
    }
    if (compute_state == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] ComputeImpl: compute_state is null");
    }
    if (kernel_context == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] ComputeImpl: kernel_context is null");
    }

    auto* node_compute_info = static_cast<TensorRtRtxEpNodeComputeInfo*>(this_ptr);
    TensorrtRtxExecutionProvider& ep = node_compute_info->ep;

    auto compute_state_ptr = static_cast<TensorrtRtxComputeState*>(compute_state);
    auto ort_ctx = Ort::KernelContext(kernel_context);
    // The whole compute_function should be considered the critical section where multiple threads may update kernel
    // function state, access one builder, create/serialize/save engine, save profile and serialize/save timing cache.
    // Therefore, those operations should be synchronized across different threads when ORT is using multithreading.
    // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    std::lock_guard<std::mutex> lock(*(compute_state_ptr->tensorrt_mu_ptr));
    const std::unordered_map<std::string, size_t>& input_indexes = (compute_state_ptr->input_info)[0];
    const std::unordered_map<std::string, size_t>& output_indexes = (compute_state_ptr->output_info)[0];
    const std::unordered_map<std::string, size_t>& output_types = (compute_state_ptr->output_info)[1];
    const auto& fused_node_name = compute_state_ptr->fused_node_name;

    std::unordered_map<std::string, std::vector<int32_t>>
        shape_tensor_values;  // This map holds "shape tensor -> shape values" for the shape tensor input across this
                              // inference run
    std::unordered_map<std::string, std::vector<int64_t>>
        shape_tensor_values_int64;  // same as above but for int64 shape tensor input
    auto& dds_output_allocator_map = ep.dds_output_allocator_maps_[fused_node_name];
    auto trt_engine = compute_state_ptr->engine->get();
    auto trt_context = compute_state_ptr->context->get();
    std::unordered_set<std::string> input_names;

    // Get default OrtMemoryInfo from factory
    const OrtMemoryInfo* mem_info = nullptr;
    if (ep.factory_.device_memory_infos.find(compute_state_ptr->device_id) != ep.factory_.device_memory_infos.end())
    {
        mem_info = ep.factory_.device_memory_infos[compute_state_ptr->device_id].get();
    }

    // Get allocator from OrtKernelContext
    if (ep.alloc_ == nullptr)
    {
        Ort::ThrowOnError(ep.ort_api.KernelContext_GetAllocator(kernel_context, mem_info, &ep.alloc_));
    }
    OrtAllocator* alloc = ep.alloc_;

    cudaStream_t stream;
    if (ep.stream_ != nullptr)
    {
        // Use our existing stream (either user's or our early-created for CUDA graph)
        stream = ep.stream_;
    }
    else
    {
        // Get the compute stream from the kernel context
        void* cuda_stream;
        Ort::ThrowOnError(ep.ort_api.KernelContext_GetGPUComputeStream(kernel_context, &cuda_stream));
        stream = static_cast<cudaStream_t>(cuda_stream);
        ep.stream_ = stream;
    }
    ScopedCudaContext compute_stream_context(ep.compute_stream_context_);

    if (compute_state_ptr->multi_profile_enable == true)
    {
        if (!trt_context->setOptimizationProfileAsync(compute_state_ptr->trt_profile_index_, stream))
            return ep.ort_api.CreateStatus(
                ORT_EP_FAIL, "NvTensorRTRTX EP select an optimization profile for the current context failed");
    }

    // Check before using trt_engine
    if (trt_engine == nullptr)
    {
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, "No engine is found.");
    }

    bool require_io_binding = IsIOBindingRequired(compute_state_ptr, kernel_context);

    // Get Inputs and Outputs binding names
    int total_binding_count = trt_engine->getNbIOTensors();
    std::vector<char const*> input_binding_names, output_binding_names;
    for (int i = 0, end = total_binding_count; i < end; ++i)
    {
        auto const& name = trt_engine->getIOTensorName(i);
        auto const& mode = trt_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_binding_names.push_back(name);
        }
        else
        {
            output_binding_names.push_back(name);
        }
    }

    /*
     * Set input shapes and bind input buffers
     */
    auto& scratch_buffers = compute_state_ptr->scratch_buffers;
    if (require_io_binding)
    {
        scratch_buffers.clear();
        bool skip_input_binding_allowed = true;
        for (size_t i = 0, end = input_binding_names.size(); i < end; ++i)
        {
            char const* input_name = input_binding_names[i];

            size_t input_index = 0;
            const auto iter = input_indexes.find(input_name);
            if (iter != input_indexes.end())
            {
                input_index = iter->second;
            }

            auto status =
                BindContextInput(ort_ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values,
                                 shape_tensor_values_int64, scratch_buffers, alloc, stream, skip_input_binding_allowed);
            if (status != nullptr)
            {
                return status;
            }
        }
        compute_state_ptr->skip_io_binding_allowed = skip_input_binding_allowed;
    }

    /*
     * Set output shapes and bind output buffers
     */
    if (require_io_binding)
    {
        for (size_t i = 0, end = output_binding_names.size(); i < end; ++i)
        {
            char const* output_name = output_binding_names[i];

            size_t output_index = 0;
            const auto& index_iter = output_indexes.find(output_name);
            if (index_iter != output_indexes.end())
            {
                output_index = index_iter->second;
            }

            size_t output_type = 0;
            const auto type_iter = output_types.find(output_name);
            if (type_iter != output_types.end())
            {
                output_type = type_iter->second;
            }

            nvinfer1::Dims dims;
            void* data_ptr = nullptr;
            bool skip_output_binding_allowed = true;

            auto status = BindContextOutput(ort_ctx, trt_context, output_name, output_index, output_type,
                                            dds_output_allocator_map, scratch_buffers, alloc, dims, data_ptr, stream,
                                            skip_output_binding_allowed);
            if (status != nullptr)
            {
                return status;
            }

            compute_state_ptr->output_tensors[output_index] = TensorParams{data_ptr, dims};
        }
    }

    // Set execution context memory
    AllocatorUniquePtr<void> context_memory;

    {
        size_t mem_size = trt_engine->getDeviceMemorySizeV2();
        if (compute_state_ptr->is_dynamic_shape)
        {
            mem_size = trt_context->updateDeviceMemorySizeForShapes();
        }
        if (mem_size > 0)
        {
            const uint32_t device_id = static_cast<uint32_t>(compute_state_ptr->device_id);
            // When the synchronous GPU allocator is enabled, skip the async mempool
            // (cudaMallocFromPoolAsync) so execution context memory is also allocated
            // synchronously through the cudaMalloc-backed OrtAllocator path below.
            CudaMempoolAllocator* mempool =
                ep.IsSyncGpuAllocatorEnabled() ? nullptr : ep.factory_.GetActiveMempoolForDevice(device_id);
            if (mempool != nullptr)
            {
                context_memory = MakeUniquePtrFromCudaMempool<void>(mempool, mem_size, stream);
                if (!context_memory)
                {
                    // Runtime async OOM (VA fragmented after the create probe).
                    // Latch the device to the sync arena and fall back for this run.
                    // (Once ORT-level graph capture is wired into compute, a captured
                    // graph holding a pool pointer must be evicted when the latch trips.)
                    ep.factory_.NoteAsyncMempoolFailure(device_id);
                    context_memory = MakeUniquePtrFromOrtAllocator<void>(alloc, mem_size, stream);
                }
            }
            else
            {
                context_memory = MakeUniquePtrFromOrtAllocator<void>(alloc, mem_size, stream);
            }
            if (!context_memory)
            {
                std::string error_msg = "Failed to allocate device memory of size " + std::to_string(mem_size) +
                                        " for TensorRT execution context.";
                return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
            }
            trt_context->setDeviceMemoryV2(context_memory.get(), mem_size);
        }
    }

    // Bind caller-provided auxiliary streams so TensorRT runs auxiliary work on them rather than
    // creating its own context/streams (required for correct CIG graphics interop).
    if (ep.external_aux_streams_ && ep.aux_streams_ != nullptr)
    {
        trt_context->setAuxStreams(ep.aux_streams_, static_cast<int32_t>(ep.auxiliary_streams_));
    }

    if (ep.profiling_enable_ && ep.profiler_)
    {
        ep.profiler_->BeginSection(fused_node_name);
    }

    if (!trt_context->enqueueV3(stream))
    {
        std::string error_msg = "NvTensorRTRTX EP execution context enqueue failed.";
        return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
    }

    /*
     * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this
     * function concurrently, TRT EP needs to carefully take care of concurrency here, if not, following concurrent
     * issue might happen:
     *
     * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per
     * stream. In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling
     * InferenceSession::Run() concurrently, the trt execution context instance is shared by all the threads and each
     * thread aquires different stream from ORT. So TRT EP will end up having one trt execution context using multiple
     * streams which is not suggested. But, since the whole compute_func() is protected by the lock and if
     * cudaStreamSynchronize() is enforced here, one trt execution context per stream is guaranteed.
     *
     * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all
     * operations to prevent the concurrent issue mentioned above. However, if cuda graph is enabled, TRT EP won't call
     * cudaStreamSynchronize() since it's not allowed during graph capture.
     */

    if (ep.sync_stream_after_enqueue_)
    {
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    }

    // Assign TRT output back to ORT output
    // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
    // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i)
    {
        char const* output_name = output_binding_names[i];

        size_t output_type = 0;
        const auto& iter = output_types.find(output_name);
        if (iter != output_types.end())
        {
            output_type = iter->second;
        }

        if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end())
        {
            size_t output_index = 0;
            const auto& index_iter = output_indexes.find(output_name);
            if (index_iter != output_indexes.end())
            {
                output_index = index_iter->second;
            }
            auto status = BindKernelOutput(ort_ctx, mem_info, dds_output_allocator_map, output_name, output_index,
                                           output_type, stream);
            if (status != nullptr)
            {
                return status;
            }
        }
    }
    return nullptr;
}

void TensorRtRtxEpNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state)
{
    (void)this_ptr;
    // Security check: validate compute_state is not null before dereferencing
    if (compute_state == nullptr)
    {
        return;
    }
    TensorrtRtxComputeState& trt_ep_compute_state = *reinterpret_cast<TensorrtRtxComputeState*>(compute_state);
    (void)trt_ep_compute_state;
    // Do nothing for here.
}

//
// TensorRtRtxCompileOnlyNodeComputeInfo implementation
//
TensorRtRtxCompileOnlyNodeComputeInfo::TensorRtRtxCompileOnlyNodeComputeInfo()
{
    ort_version_supported = NegotiatedOrtApiVersion();
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
}

OrtStatus* ORT_API_CALL TensorRtRtxCompileOnlyNodeComputeInfo::CreateStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/, OrtNodeComputeContext* /*compute_context*/, void** compute_state)
{
    if (compute_state != nullptr)
    {
        *compute_state = nullptr;
    }
    return nullptr;
}

OrtStatus* ORT_API_CALL TensorRtRtxCompileOnlyNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                                                           void* /*compute_state*/,
                                                                           OrtKernelContext* /*kernel_context*/)
{
    return g_ort_api->CreateStatus(ORT_NOT_IMPLEMENTED,
                                   "[NvTensorRTRTX EP] inference is not available in a compile-only session");
}

void ORT_API_CALL TensorRtRtxCompileOnlyNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                                                          void* /*compute_state*/)
{
}

TensorRtRtxEpContextNodeComputeInfo::TensorRtRtxEpContextNodeComputeInfo(TensorrtRtxExecutionProvider& ep)
    : ep(ep)
{
    ort_version_supported = NegotiatedOrtApiVersion();
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
}

OrtStatus* TensorRtRtxEpContextNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                                OrtNodeComputeContext* compute_context,
                                                                void** compute_state)
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] EPContext CreateStateImpl: this_ptr is null");
    }
    if (compute_context == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] EPContext CreateStateImpl: compute_context is null");
    }
    if (compute_state == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] EPContext CreateStateImpl: compute_state output is null");
    }

    auto* node_compute_info = static_cast<TensorRtRtxEpContextNodeComputeInfo*>(this_ptr);
    TensorrtRtxExecutionProvider& ep = node_compute_info->ep;

    std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto compute_state_ptr = ep.compute_states_for_ep_context_[fused_node_name].get();
    if (compute_state_ptr == nullptr)
    {
        std::string message = "[NvTensorRTRTX EP] Compute state not found for fused node: " + fused_node_name;
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    TensorrtRtxEpContextNodeComputeState& trt_ep_compute_state = *compute_state_ptr;
    *compute_state = &trt_ep_compute_state;
    return nullptr;
}

OrtStatus* TensorRtRtxEpContextNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                                            OrtKernelContext* kernel_context)
{
    // Security check: validate input parameters are not null
    if (this_ptr == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] EPContext ComputeImpl: this_ptr is null");
    }
    if (compute_state == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] EPContext ComputeImpl: compute_state is null");
    }
    if (kernel_context == nullptr)
    {
        return g_ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                       "[NvTensorRTRTX EP] EPContext ComputeImpl: kernel_context is null");
    }

    auto* node_compute_info = static_cast<TensorRtRtxEpContextNodeComputeInfo*>(this_ptr);
    TensorrtRtxExecutionProvider& ep = node_compute_info->ep;

    TensorrtRtxEpContextNodeComputeState* trt_state =
        reinterpret_cast<TensorrtRtxEpContextNodeComputeState*>(compute_state);

    auto ort_ctx = Ort::KernelContext(kernel_context);

    // The whole compute_function should be considered the critical section.
    // More details here, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#threading
    std::lock_guard<std::mutex> lock(*(trt_state->tensorrt_mu_ptr));
    const std::unordered_map<std::string, size_t>& input_indexes = (trt_state->input_info)[0];
    const std::unordered_map<std::string, size_t>& output_indexes = (trt_state->output_info)[0];
    const std::unordered_map<std::string, size_t>& output_types = (trt_state->output_info)[1];
    const auto& fused_node_name = trt_state->fused_node_name;
    auto& dds_output_allocator_map = ep.dds_output_allocator_maps_[fused_node_name];
    auto trt_engine = trt_state->engine->get();
    auto trt_context = trt_state->context->get();
    std::unordered_map<std::string, std::vector<int32_t>>
        shape_tensor_values;  // This map holds "shape tensor -> shape values" for the shape tensor input across this
                              // inference run
    std::unordered_map<std::string, std::vector<int64_t>>
        shape_tensor_values_int64;  // same as above but for int64 shape tensor input

    // Get default OrtMemoryInfo from factory
    const OrtMemoryInfo* mem_info = nullptr;
    if (ep.factory_.device_memory_infos.find(trt_state->device_id) != ep.factory_.device_memory_infos.end())
    {
        mem_info = ep.factory_.device_memory_infos[trt_state->device_id].get();
    }

    // Get allocator from OrtKernelContext
    if (ep.alloc_ == nullptr)
    {
        Ort::ThrowOnError(ep.ort_api.KernelContext_GetAllocator(kernel_context, mem_info, &ep.alloc_));
    }
    OrtAllocator* alloc = ep.alloc_;

    cudaStream_t stream;
    if (ep.stream_ != nullptr)
    {
        // Use our existing stream (either user's or our early-created for CUDA graph)
        stream = ep.stream_;
    }
    else
    {
        // Get the compute stream from the kernel context
        void* cuda_stream;
        Ort::ThrowOnError(ep.ort_api.KernelContext_GetGPUComputeStream(kernel_context, &cuda_stream));
        stream = static_cast<cudaStream_t>(cuda_stream);
        ep.stream_ = stream;
    }
    ScopedCudaContext compute_stream_context(ep.compute_stream_context_);

    // Check before using trt_engine
    if (trt_engine == nullptr)
    {
        return ep.ort_api.CreateStatus(ORT_EP_FAIL, "No engine is found.");
    }

    bool require_io_binding = IsIOBindingRequired(trt_state, kernel_context);

    // Get Inputs and Outputs binding names
    int total_binding_count = trt_engine->getNbIOTensors();
    std::vector<char const*> input_binding_names, output_binding_names;
    for (int i = 0, end = total_binding_count; i < end; ++i)
    {
        auto const& name = trt_engine->getIOTensorName(i);
        auto const& mode = trt_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_binding_names.push_back(name);
        }
        else
        {
            output_binding_names.push_back(name);
        }
    }

    /*
     * Set input shapes and bind input buffers
     */
    auto& scratch_buffers = trt_state->scratch_buffers;
    if (require_io_binding)
    {
        scratch_buffers.clear();
        bool skip_input_binding_allowed = true;
        for (size_t i = 0, end = input_binding_names.size(); i < end; ++i)
        {
            char const* input_name = input_binding_names[i];

            size_t input_index = 0;
            const auto iter = input_indexes.find(input_name);
            if (iter != input_indexes.end())
            {
                input_index = iter->second;
            }

            auto status =
                BindContextInput(ort_ctx, trt_engine, trt_context, input_name, input_index, shape_tensor_values,
                                 shape_tensor_values_int64, scratch_buffers, alloc, stream, skip_input_binding_allowed);
            if (status != nullptr)
            {
                return status;
            }
        }
        trt_state->skip_io_binding_allowed = skip_input_binding_allowed;
    }

    /*
     * Set output shapes and bind output buffers
     */
    if (require_io_binding)
    {
        for (size_t i = 0, end = output_binding_names.size(); i < end; ++i)
        {
            char const* output_name = output_binding_names[i];

            size_t output_index = 0;
            const auto& index_iter = output_indexes.find(output_name);
            if (index_iter != output_indexes.end())
            {
                output_index = index_iter->second;
            }

            size_t output_type = 0;
            const auto type_iter = output_types.find(output_name);
            if (type_iter != output_types.end())
            {
                output_type = type_iter->second;
            }

            nvinfer1::Dims dims;
            void* data_ptr = nullptr;
            bool skip_output_binding_allowed = true;

            auto status = BindContextOutput(ort_ctx, trt_context, output_name, output_index, output_type,
                                            dds_output_allocator_map, scratch_buffers, alloc, dims, data_ptr, stream,
                                            skip_output_binding_allowed);
            if (status != nullptr)
            {
                return status;
            }

            trt_state->output_tensors[output_index] = TensorParams{data_ptr, dims};
        }
    }

    // Set execution context memory
    AllocatorUniquePtr<void> context_memory;

    {
        size_t mem_size = trt_engine->getDeviceMemorySizeV2();
        if (trt_state->is_dynamic_shape)
        {
            mem_size = trt_context->updateDeviceMemorySizeForShapes();
        }
        if (mem_size > 0)
        {
            const uint32_t device_id = static_cast<uint32_t>(trt_state->device_id);
            // When the synchronous GPU allocator is enabled, skip the async mempool
            // (cudaMallocFromPoolAsync) so execution context memory is also allocated
            // synchronously through the cudaMalloc-backed OrtAllocator path below.
            CudaMempoolAllocator* mempool =
                ep.IsSyncGpuAllocatorEnabled() ? nullptr : ep.factory_.GetActiveMempoolForDevice(device_id);
            if (mempool != nullptr)
            {
                context_memory = MakeUniquePtrFromCudaMempool<void>(mempool, mem_size, stream);
                if (!context_memory)
                {
                    // Runtime async OOM (VA fragmented after the create probe).
                    // Latch the device to the sync arena and fall back for this run.
                    // (Once ORT-level graph capture is wired into compute, a captured
                    // graph holding a pool pointer must be evicted when the latch trips.)
                    ep.factory_.NoteAsyncMempoolFailure(device_id);
                    context_memory = MakeUniquePtrFromOrtAllocator<void>(alloc, mem_size, stream);
                }
            }
            else
            {
                context_memory = MakeUniquePtrFromOrtAllocator<void>(alloc, mem_size, stream);
            }
            if (!context_memory)
            {
                std::string error_msg = "Failed to allocate device memory of size " + std::to_string(mem_size) +
                                        " for TensorRT execution context.";
                return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
            }
            trt_context->setDeviceMemoryV2(context_memory.get(), mem_size);
        }
    }

    // Start CUDA graph capture with the correct stream
    // Note: We need to set the stream and start capture here because this is where we have access to the actual compute
    // stream Get the graph annotation ID that was stored during OnRunStart
    // CudaGraphAnnotation_t cuda_graph_annotation_id = GetPerThreadContext().GetCurrentGraphAnnotationId();
    // bool graph_replay_on_this_run = false;
    // bool should_start_capture = false;

    // HandleCudaGraphStart(stream, require_io_binding, cuda_graph_annotation_id,
    //                      graph_replay_on_this_run, should_start_capture);

    if (ep.profiling_enable_ && ep.profiler_)
    {
        ep.profiler_->BeginSection(fused_node_name);
    }

    // if (!graph_replay_on_this_run) {
    // Bind caller-provided auxiliary streams so TensorRT runs auxiliary work on them rather than
    // creating its own context/streams (required for correct CIG graphics interop).
    if (ep.external_aux_streams_ && ep.aux_streams_ != nullptr)
    {
        trt_context->setAuxStreams(ep.aux_streams_, static_cast<int32_t>(ep.auxiliary_streams_));
    }

    if (!trt_context->enqueueV3(stream))
    {
        std::string error_msg = "NvTensorRTRTX EP execution context enqueue failed.";
        return g_ort_api->CreateStatus(OrtErrorCode::ORT_EP_FAIL, error_msg.c_str());
    }
    //} else {
    //  ORT_RETURN_IF_ERROR(GetPerThreadContext().ReplayGraph(cuda_graph_annotation_id, sync_stream_after_enqueue_));
    //}

    /*
     * Given that InferenceSession::Run() is guaranteed to be thread-safe meaning multiple threads can call this
     * function concurrently, TRT EP needs to carefully take care of concurrency here, if not, following concurrent
     * issue might happen:
     *
     * It's suggested that to perform inference concurrently in multiple streams, use one trt execution context per
     * stream. In the design of TRT EP (Not apply per-thread context implementation) and if multiple threads are calling
     * InferenceSession::Run() concurrently, the trt execution context instance is shared by all the threads and each
     * thread aquires different stream from ORT. So TRT EP will end up having one trt execution context using multiple
     * streams which is not suggested. But, since the whole compute_func() is protected by the lock and if
     * cudaStreamSynchronize() is enforced here, one trt execution context per stream is guaranteed.
     *
     * Therefore, TRT EP needs to call cudaStreamSynchronize() which means to wait until stream has completed all
     * operations to prevent the concurrent issue mentioned above. However, if cuda graph is enabled, TRT EP won't call
     * cudaStreamSynchronize() since it's not allowed during graph capture.
     */

    // if (cuda_graph_enable_ && should_start_capture) {
    //   GetPerThreadContext().CaptureEnd(cuda_graph_annotation_id);
    //   ORT_RETURN_IF_ERROR(GetPerThreadContext().ReplayGraph(cuda_graph_annotation_id, sync_stream_after_enqueue_));
    // }

    if (ep.sync_stream_after_enqueue_)
    {
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    }

    // Assign TRT output back to ORT output
    // (1) Bind TRT DDS output to ORT kernel context output. (It needs to wait until enqueueV3 is finished)
    // (2) Cast TRT INT32 output to ORT INT64 output or TRT double output to float output
    for (size_t i = 0, end = output_binding_names.size(); i < end; ++i)
    {
        char const* output_name = output_binding_names[i];

        size_t output_type = 0;
        const auto& iter = output_types.find(output_name);
        if (iter != output_types.end())
        {
            output_type = iter->second;
        }

        if (dds_output_allocator_map.find(output_name) != dds_output_allocator_map.end())
        {
            size_t output_index = 0;
            const auto& index_iter = output_indexes.find(output_name);
            if (index_iter != output_indexes.end())
            {
                output_index = index_iter->second;
            }
            auto status = BindKernelOutput(ort_ctx, mem_info, dds_output_allocator_map, output_name, output_index,
                                           output_type, stream);
            if (status != nullptr)
            {
                return status;
            }
        }
    }
    return nullptr;
}

void TensorRtRtxEpContextNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state)
{
    (void)this_ptr;
    // Security check: validate compute_state is not null before dereferencing
    if (compute_state == nullptr)
    {
        return;
    }
    TensorrtRtxEpContextNodeComputeState& trt_ep_compute_state =
        *reinterpret_cast<TensorrtRtxEpContextNodeComputeState*>(compute_state);
    (void)trt_ep_compute_state;
    // Do nothing for here.
}

}  // namespace trt_rtx_ep
