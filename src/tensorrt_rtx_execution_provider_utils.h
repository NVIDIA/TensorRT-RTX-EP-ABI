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

#include "utils/path_string.h"
#include "utils/murmurhash3.h"
#include "cuda_mempool_arena.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_rtx_ep
{

extern const OrtApi* g_ort_api;
extern const OrtEpApi* g_ep_api;
extern const OrtModelEditorApi* g_model_editor_api;
extern const OrtLogger* g_logger;

//!
//! \brief Get compute capability from CUDA device properties.
//!
inline std::string GetComputeCapability(const cudaDeviceProp& prop)
{
    const std::string compute_capability = std::to_string(prop.major * 10 + prop.minor);
    return compute_capability;
}

template <typename T>
AllocatorUniquePtr<T> MakeUniquePtrFromOrtAllocator(OrtAllocator* ort_allocator, size_t count_or_bytes,
                                                    bool use_reserve = false)
{
    size_t alloc_size = count_or_bytes;

    T* p = nullptr;
    if (use_reserve)
    {
        p = static_cast<T*>(ort_allocator->Reserve(ort_allocator, alloc_size));
    }
    else
    {
        p = static_cast<T*>(ort_allocator->Alloc(ort_allocator, alloc_size));
    }

    return AllocatorUniquePtr<T>{p, [ort_allocator](T* p)
                                 {
                                     ort_allocator->Free(ort_allocator, p);
                                 }};                                 
}

//! \brief Stream-ordered allocation via CudaMempoolAllocator, wrapped in a unique_ptr.
//!
//! Uses cudaMallocFromPoolAsync on the given stream, avoiding the default-stream
//! synchronisation that OrtAllocator::Alloc would cause.
template <typename T>
AllocatorUniquePtr<T> MakeUniquePtrFromCudaMempool(CudaMempoolAllocator* allocator, size_t count_or_bytes,
                                                   cudaStream_t stream)
{
    T* p = static_cast<T*>(allocator->AllocOnCudaStream(count_or_bytes, stream));
    return AllocatorUniquePtr<T>{p, [allocator](T* p)
                                 {
                                     allocator->FreeOnCudaStream(p);
                                 }};
}

//!
//! \brief Make input-name and shape as a pair.
//!
//! This helper function is being used by ParseProfileShapes().
//!
//! For example:
//! The input string is "input_id:32x1",
//! after the string is being parsed, the pair object is returned as below.
//! pair("input_id", [32, 1])
//!
//! \return True if string can be successfully parsed or false if string has wrong format.
//!
inline bool MakeInputNameShapePair(std::string pair_string, std::pair<std::string, std::vector<int64_t>>& pair)
{
    if (pair_string.empty())
    {
        return true;
    }

    std::stringstream input_string_stream(pair_string);
    char first_delim = ':';
    char second_delim = 'x';
    std::string input_name;
    std::string shape;
    std::getline(input_string_stream, input_name, first_delim);
    std::getline(input_string_stream, shape, first_delim);

    std::vector<int64_t> shapes;
    std::stringstream shape_string_stream(shape);
    std::string value;
    while (std::getline(shape_string_stream, value, second_delim))
    {
        shapes.push_back(std::stoi(value));
    }

    // wrong input string
    if (input_name.empty() || shapes.empty())
    {
        return false;
    }

    pair.first = input_name;
    pair.second = shapes;

    return true;
}

//!
//! \brief Parse explicit profile min/max/opt shapes from Nv EP provider options.
//!
//! For example:
//! The provider option is --trt_profile_min_shapes="input_id:32x1,attention_mask:32x1,input_id:32x41,attention_mask:32x41",
//! after string is being parsed, the profile shapes has two profiles and is being represented as below.
//! {"input_id": [[32, 1], [32, 41]], "attention_mask": [[32, 1], [32, 41]]}
//!
//! \return True if string can be successfully parsed or false if string has wrong format.
//!
inline bool ParseProfileShapes(std::string profile_shapes_string,
                               std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_shapes)
{
    if (profile_shapes_string.empty())
    {
        return true;
    }

    std::stringstream input_string_stream(profile_shapes_string);
    char delim = ',';
    std::string input_name_with_shape;  // input_name:shape, ex: "input_id:32x1"
    while (std::getline(input_string_stream, input_name_with_shape, delim))
    {
        std::pair<std::string, std::vector<int64_t>> pair;
        if (!MakeInputNameShapePair(input_name_with_shape, pair))
        {
            return false;
        }

        std::string input_name = pair.first;
        if (profile_shapes.find(input_name) == profile_shapes.end())
        {
            std::vector<std::vector<int64_t>> profile_shape_vector;
            profile_shapes[input_name] = profile_shape_vector;
        }
        profile_shapes[input_name].push_back(pair.second);

        std::string shape_string = "";
        for (auto v : pair.second)
        {
            shape_string += std::to_string(v);
            shape_string += ", ";
        }
    }

    return true;
}

inline bool ValidateProfileShapes(std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_min_shapes,
                                  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_max_shapes,
                                  std::unordered_map<std::string, std::vector<std::vector<int64_t>>>& profile_opt_shapes)
{
    if (profile_min_shapes.empty() && profile_max_shapes.empty() && profile_opt_shapes.empty())
    {
        return true;
    }

    if ((profile_min_shapes.size() != profile_max_shapes.size()) ||
        (profile_min_shapes.size() != profile_opt_shapes.size()) ||
        (profile_max_shapes.size() != profile_opt_shapes.size()))
    {
        return false;
    }

    std::unordered_map<std::string, std::vector<std::vector<int64_t>>>::iterator it;
    for (it = profile_min_shapes.begin(); it != profile_min_shapes.end(); it++)
    {
        auto input_name = it->first;
        auto num_profile = it->second.size();

        // input_name must also be in max/opt profile
        if ((profile_max_shapes.find(input_name) == profile_max_shapes.end()) ||
            (profile_opt_shapes.find(input_name) == profile_opt_shapes.end()))
        {
            return false;
        }

        // number of profiles should be the same
        if ((num_profile != profile_max_shapes[input_name].size()) ||
            (num_profile != profile_opt_shapes[input_name].size()))
        {
            return false;
        }
    }

    return true;
}

//!
//! \brief Get cache by name.
//!
inline std::string GetCachePath(const std::string& root, const std::string& name)
{
    if (root.empty())
    {
        return name;
    }
    else
    {
        std::filesystem::path path = root;
        path.append(name);
        return path.string();
    }
}

//!
//! \brief Helper class to generate engine id via model name/model content/env metadata.
//!
//! \details The TensorRT RTX Execution Provider is used in multiple sessions and the underlying infrastructure caches
//! compiled kernels, so the name must be unique and deterministic across models and sessions.
//!
inline HashValue TRTGenerateId(const OrtGraph* graph, const std::string& trt_version, const std::string& cuda_version)
{
    HashValue model_hash = 0;
    auto ort_graph = Ort::ConstGraph(graph);

    // find the top level graph
    auto parent_node = ort_graph.GetParentNode();
    while (parent_node && parent_node.GetGraph() != nullptr)
    {
        ort_graph = parent_node.GetGraph();
        parent_node = ort_graph.GetParentNode();
    }

    auto main_graph = ort_graph;
    uint32_t hash[4] = {0, 0, 0, 0};
    auto hash_str = [&hash](const std::string& str)
    {
        MurmurHash3_x86_128(str.data(), static_cast<int32_t>(str.size()), hash[0], &hash);
    };

    PathString model_path = main_graph.GetModelPath();

    if (!model_path.empty())
    {
        // Extract filename from the full path
        size_t last_separator = model_path.find_last_of(PathChar('/'));
        size_t last_backslash = model_path.find_last_of(PathChar('\\'));
        size_t separator_pos = PathString::npos;

        if (last_separator != PathString::npos && last_backslash != PathString::npos)
        {
            separator_pos = (std::max)(last_separator, last_backslash);
        }
        else if (last_separator != PathString::npos)
        {
            separator_pos = last_separator;
        }
        else if (last_backslash != PathString::npos)
        {
            separator_pos = last_backslash;
        }

        PathString filename;
        if (separator_pos != PathString::npos)
        {
            filename = model_path.substr(separator_pos + 1);
        }
        else
        {
            filename = model_path;
        }

        std::string model_name = PathToUTF8String(filename);

        Ort::ThrowOnError(g_ort_api->Logger_LogMessage(g_logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       ("Model name is " + model_name).c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        // Hash the model name directly - the hash function handles variable-length input
        hash_str(model_name);
    }
    else
    {
        Ort::ThrowOnError(g_ort_api->Logger_LogMessage(g_logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Model path is empty", ORT_FILE, __LINE__, __FUNCTION__));
    }

    // fingerprint current graph by hashing graph inputs
    for (auto input : main_graph.GetInputs())
    {
        hash_str(input.GetName());
    }

    // hashing initializers
    for (auto initializer : main_graph.GetInitializers())
    {
        hash_str(initializer.GetName());
    }

    // hashing outputs of each node
    for (auto node : main_graph.GetNodes())
    {
        for (auto output : node.GetOutputs())
        {
            hash_str(output.GetName());
        }
    }

#if defined(__linux__)
    hash_str("LINUX");
#elif defined(_WIN32)
    hash_str("WINDOWS");
#endif

#if defined(ORT_VERSION)
    hash_str(ORT_VERSION);
#endif

#if defined(CUDA_VERSION)
    hash_str(cuda_version);
#endif

#if defined(NV_TENSORRT_MAJOR) && defined(NV_TENSORRT_MINOR)
    hash_str(trt_version);
#endif

    model_hash = hash[0] | (uint64_t(hash[1]) << 32);

    return model_hash;
}

inline std::vector<std::string> split(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

inline std::string join(const std::vector<std::string>& vec, const std::string& delimiter)
{
    std::string result;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        result += vec[i];
        if (i < vec.size() - 1)
        {
            result += delimiter;
        }
    }
    return result;
}

//!
//! \brief Parse engine cache name suffix when user customizes prefix for engine cache name.
//!
//! For example:
//! When default subgraph name is "NvExecutionProvider_TRTKernel_graph_torch-jit-export_2068723788287043730_189_189_fp16"
//! This func will generate the suffix "2068723788287043730_189_fp16"
//!
inline std::string GetCacheSuffix(const std::string& fused_node_name, const std::string& trt_node_name_with_precision)
{
    std::vector<std::string> split_fused_node_name = split(fused_node_name, '_');
    if (split_fused_node_name.size() >= 3)
    {
        // Get index of model hash from fused_node_name
        std::string model_hash = split_fused_node_name[split_fused_node_name.size() - 3];
        size_t index = fused_node_name.find(model_hash);
        // Parse suffix from trt_node_name_with_precision, as it has additional precision info
        std::vector<std::string> suffix_group = split(trt_node_name_with_precision.substr(index), '_');
        if (suffix_group.size() > 2)
        {
            suffix_group.erase(suffix_group.begin() + 2);
        }
        return join(suffix_group, "_");
    }
    return "";
}

//!
//! \brief Checks if there is an element with value `-1` in nvinfer1::Dims.
//!
static bool checkTrtDimIsDynamic(nvinfer1::Dims dims)
{
    for (int j = 0, end = dims.nbDims; j < end; ++j)
    {
        if (dims.d[j] == -1)
        {
            return true;
        }
    }
    return false;
}

//!
//! \brief Checks if an nvinfer1::ITensor signals a dynamic shape.
//!
//! Either due to dynamic shapes or due to it being a shape tensor.
//!
static bool checkTrtTensorIsDynamic(nvinfer1::ITensor* tensor)
{
    if (tensor->isShapeTensor())
    {
        return true;
    }
    else
    {
        // Execution tensor
        return checkTrtDimIsDynamic(tensor->getDimensions());
    }
}

}  // namespace trt_rtx_ep
