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

#include "onnx_ctx_model_helper.h"

#include "utils/ep_utils.h"
#include "utils/path_string.h"

#include "onnx/onnx_pb.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace trt_rtx_ep
{

/*
 * Convert binary data to hex string
 */
std::string BinaryToHexString(const void* data, size_t size)
{
    static const char hex_chars[] = "0123456789abcdef";
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    std::string result;
    result.reserve(size * 2);

    for (size_t i = 0; i < size; ++i)
    {
        result.push_back(hex_chars[(bytes[i] >> 4) & 0xF]);
        result.push_back(hex_chars[bytes[i] & 0xF]);
    }
    return result;
}

/*
 * Convert hex string back to binary
 */
std::vector<uint8_t> HexStringToBinary(const std::string& hex)
{
    if (hex.size() % 2 != 0)
    {
        THROW("Hex string must have even length");
    }

    std::vector<uint8_t> result;
    result.reserve(hex.size() / 2);

    auto nibble = [](char c, uint8_t& value) -> bool
    {
        if (c >= '0' && c <= '9')
        {
            value = static_cast<uint8_t>(c - '0');
            return true;
        }
        if (c >= 'a' && c <= 'f')
        {
            value = static_cast<uint8_t>(c - 'a' + 10);
            return true;
        }
        if (c >= 'A' && c <= 'F')
        {
            value = static_cast<uint8_t>(c - 'A' + 10);
            return true;
        }
        return false;
    };

    for (size_t i = 0; i < hex.size(); i += 2)
    {
        uint8_t high = 0;
        uint8_t low = 0;
        if (!nibble(hex[i], high) || !nibble(hex[i + 1], low))
        {
            THROW("Hex string contains non-hex characters");
        }
        result.push_back(static_cast<uint8_t>((high << 4) | low));
    }
    return result;
}

// Forward declaration
std::string GetWeightRefittedEnginePath(std::string stripped_engine_cache);

bool IsAbsolutePath(const std::string& path_string)
{
    // Use std::filesystem::path::is_absolute() for consistent cross-platform behavior
    PathString ort_path_string = ToPathString(path_string);
    auto path = std::filesystem::path(ort_path_string.c_str());
    return path.is_absolute();
}

//!
//! \brief Check if path is like "../file_path"
//!
//! Returns true if the path attempts to traverse to a parent directory.
//! Uses path normalization and component inspection to avoid false positives
//! from filenames containing ".." (e.g., "file..name.txt").
//!
bool IsRelativePathToParentPath(const std::string& path_string)
{
    if (path_string.empty())
    {
        return false;
    }

#if defined(_WIN32)
    PathString ort_path_string = ToPathString(path_string);
    std::filesystem::path path(ort_path_string.c_str());
#else
    std::filesystem::path path(path_string);
#endif

    // Normalize the path to resolve "." and ".." where possible.
    // After normalization, ".." components only remain if they escape the base directory.
    std::filesystem::path normalized_path = path.lexically_normal();

    // Iterate through path components to check if any is ".."
    for (const auto& component : normalized_path)
    {
        if (component == "..")
        {
            return true;
        }
    }

    return false;
}

//!
//! \brief Return the directory where the ep context model locates.
//!
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::string& ep_context_file_path)
{
    if (ep_context_file_path.empty())
    {
        return std::filesystem::path();
    }
    std::filesystem::path ctx_path(ep_context_file_path);
    if (std::filesystem::is_directory(ep_context_file_path))
    {
        return ctx_path;
    }
    else
    {
        return ctx_path.parent_path();
    }
}

bool IsWeightStrippedEngineCache(std::filesystem::path& engine_cache_path)
{
    // The weight-stripped engine cache has the naming of xxx.stripped.engine
    return engine_cache_path.stem().extension().string() == ".stripped";
}

//!
//! \brief Create an EPContext OrtNode from a fused_node.
//!
OrtStatus* EPContextNodeHelper::CreateEPContextNode(const std::string& engine_cache_path,
                                                    char* engine_data,
                                                    size_t size,
                                                    const int64_t embed_mode,
                                                    const std::string& compute_capability,
                                                    const std::string& onnx_model_path,
                                                    OrtNode** ep_context_node)
{
    // Helper to collect input or output names from an array of OrtValueInfo instances.
    // Only includes actual graph inputs, excluding initializers (constant weights).
    auto collect_input_output_names = [&](const std::vector<const OrtValueInfo*>& value_infos,
                                          std::vector<const char*>& result) -> OrtStatus*
    {
        size_t num_values = value_infos.size();
        std::vector<const char*> value_names;

        for (size_t i = 0; i < num_values; ++i)
        {
            const OrtValueInfo* value_info = value_infos[i];

            // Check if this is an initializer (constant weight) - if so, skip it
            const OrtValue* initializer_value = nullptr;
            OrtStatus* status = ort_api.ValueInfo_GetInitializerValue(value_info, &initializer_value);
            if (status != nullptr)
            {
                ort_api.ReleaseStatus(status);
                // Error getting initializer value, treat as non-initializer
            }

            // Skip initializers - only include actual graph inputs
            if (initializer_value != nullptr)
            {
                continue;
            }

            // Check if this is a required graph input
            bool is_required_graph_input = true;
            // RETURN_IF_ERROR(ort_api.ValueInfo_IsRequiredGraphInput(value_info, &is_required_graph_input));

            if (is_required_graph_input)
            {
                const char* value_name = nullptr;
                RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &value_name));
                value_names.push_back(value_name);
            }
        }

        result = std::move(value_names);
        return nullptr;
    };

    const char* fused_node_name = nullptr;

    RETURN_IF_ERROR(ort_api.Node_GetName(fused_node_, &fused_node_name));

    size_t num_fused_node_inputs = 0;
    size_t num_fused_node_outputs = 0;
    RETURN_IF_ERROR(ort_api.Node_GetNumInputs(fused_node_, &num_fused_node_inputs));
    RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(fused_node_, &num_fused_node_outputs));

    std::vector<const OrtValueInfo*> fused_node_inputs(num_fused_node_inputs);
    std::vector<const OrtValueInfo*> fused_node_outputs(num_fused_node_outputs);
    RETURN_IF_ERROR(ort_api.Node_GetInputs(fused_node_, fused_node_inputs.data(), fused_node_inputs.size()));
    RETURN_IF_ERROR(ort_api.Node_GetOutputs(fused_node_, fused_node_outputs.data(), fused_node_outputs.size()));

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    RETURN_IF_ERROR(collect_input_output_names(fused_node_inputs, /*out*/ input_names));
    RETURN_IF_ERROR(collect_input_output_names(fused_node_outputs, /*out*/ output_names));

    // Create node attributes. The CreateNode() function copies the attributes, so we have to release them.
    std::array<OrtOpAttr*, 9> attributes = {};
    DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

    RETURN_IF_ERROR(ort_api.CreateOpAttr(EMBED_MODE.c_str(), &embed_mode, sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[0]));

    std::string engine_data_str = "";
    if (embed_mode)
    {
        if (size > 0)
        {
            engine_data_str.assign(engine_data, size);
        }
        RETURN_IF_ERROR(
            ort_api.CreateOpAttr(EP_CACHE_CONTEXT.c_str(), engine_data_str.c_str(), static_cast<int>(engine_data_str.size()), ORT_OP_ATTR_STRING, &attributes[1]));
    }
    else
    {
        std::fstream engine_cache_file(engine_cache_path, std::ios::binary | std::ios::out);
        if (!engine_cache_file.is_open())
        {
            return ort_api.CreateStatus(ORT_FAIL,
                                        ("Failed to open engine cache file for writing: " + engine_cache_path).c_str());
        }
        engine_cache_file.write(engine_data, size);
        if (engine_cache_file.fail())
        {
            engine_cache_file.close();
            return ort_api.CreateStatus(ORT_FAIL,
                                        ("Failed to write engine data to cache file: " + engine_cache_path).c_str());
        }
        engine_cache_file.close();

        std::string attr_path = engine_cache_path;
        if (IsAbsolutePath(attr_path))
        {
            std::filesystem::path ctx_model_dir = GetPathOrParentPathOfCtxModel(ep_.GetEpContextFilePath());
            if (!ctx_model_dir.empty())
            {
                std::error_code ec;
                std::filesystem::path rel_path =
                    std::filesystem::relative(std::filesystem::path(engine_cache_path), ctx_model_dir, ec);
                if (!ec && !rel_path.empty())
                {
                    attr_path = rel_path.string();
                }
            }
        }
        if (IsAbsolutePath(attr_path) || IsRelativePathToParentPath(attr_path))
        {
            attr_path = std::filesystem::path(engine_cache_path).filename().string();
        }
        RETURN_IF_ERROR(ort_api.CreateOpAttr(EP_CACHE_CONTEXT.c_str(), attr_path.c_str(),
                                             static_cast<int>(attr_path.size()), ORT_OP_ATTR_STRING, &attributes[1]));
    }

    std::string onnx_model_filename = std::filesystem::path(onnx_model_path).filename().string();
    RETURN_IF_ERROR(ort_api.CreateOpAttr(COMPUTE_CAPABILITY.c_str(), compute_capability.c_str(), static_cast<int>(compute_capability.size()), ORT_OP_ATTR_STRING, &attributes[2]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr(ONNX_MODEL_FILENAME.c_str(), onnx_model_filename.c_str(), static_cast<int>(onnx_model_filename.size()),
                                         ORT_OP_ATTR_STRING, &attributes[3]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr(PARTITION_NAME.c_str(), fused_node_name, static_cast<int>(strlen(fused_node_name)), ORT_OP_ATTR_STRING, &attributes[4]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr(SDK_VERSION.c_str(), "1.0", 3, ORT_OP_ATTR_STRING, &attributes[5]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_context_op_domain", EPCONTEXT_OP_DOMAIN.c_str(), static_cast<int>(EPCONTEXT_OP_DOMAIN.size()), ORT_OP_ATTR_STRING, &attributes[6]));

    int64_t main_context = 0;
    RETURN_IF_ERROR(ort_api.CreateOpAttr(MAIN_CONTEXT.c_str(), &main_context, sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[7]));

    const std::string& ep_source = ep_.name_;
    RETURN_IF_ERROR(ort_api.CreateOpAttr(SOURCE.c_str(), ep_source.c_str(), static_cast<int>(ep_source.size()), ORT_OP_ATTR_STRING, &attributes[8]));

    RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name, input_names.data(),
                                                input_names.size(), output_names.data(), output_names.size(),
                                                attributes.data(), attributes.size(), ep_context_node));

    return nullptr;
}

//!
//! \brief Check whether the graph has the EP context node.
//!
//! The node can contain the precompiled engine info for TRT EP to directly load the engine.
//!
//! \note Please see more details about "EPContext" contrib op in contrib_defs.cc
//!
bool EPContextNodeReader::GraphHasCtxNode(const OrtGraph* graph, const OrtApi& ort_api)
{
    size_t num_nodes = 0;
    RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));

    std::vector<const OrtNode*> nodes(num_nodes);
    RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    for (size_t i = 0; i < num_nodes; ++i)
    {
        auto node = nodes[i];

        const char* op_type = nullptr;
        RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node, &op_type));
        if (node != nullptr && std::string(op_type) == "EPContext")
        {
            return true;
        }
    }
    return false;
}

//!
//! \brief The sanity check for EP context contrib op.
//!
bool EPContextNodeReader::ValidateEPCtxNode(const OrtGraph* graph) const
{
    size_t num_nodes = 0;
    THROW_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));
    ENFORCE(num_nodes == 1);

    std::vector<const OrtNode*> nodes(num_nodes);
    RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

    const char* op_type = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetOperatorType(nodes[0], &op_type));
    ENFORCE(std::string(op_type) == "EPContext");

    // TODO: (Umang) Check TRT RTX API support for the EPContext node
    return true;
}

OrtStatus* EPContextNodeReader::GetEpContextFromGraph(const OrtGraph& graph)
{
    if (!ValidateEPCtxNode(&graph))
    {
        return ort_api.CreateStatus(ORT_EP_FAIL, "It's not a valid EPContext node");
    }

    size_t num_nodes = 0;
    RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(&graph, &num_nodes));

    auto ort_graph = Ort::ConstGraph(&graph);
    std::vector<Ort::ConstNode> nodes(num_nodes);
    nodes = ort_graph.GetNodes();

    // ValidateEPCtxNode() already checked ENFORCE(num_nodes == 1)
    auto& node = nodes[0];
    Ort::ConstOpAttr node_attr;

    // Get "embed_mode" attribute
    RETURN_IF_ORT_STATUS_ERROR(node.GetAttributeByName("embed_mode", node_attr));
    ENFORCE(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_INT);

    int64_t embed_mode = 0;
    RETURN_IF_ORT_STATUS_ERROR(node_attr.GetValue(embed_mode));

    // Get "partition_name" (fused node name at build time) for runtime cache path consistency across sessions
    partition_name_.clear();
    {
        Ort::ConstOpAttr part_attr;
        if (node.GetAttributeByName(PARTITION_NAME.c_str(), part_attr).IsOK() &&
            part_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING)
        {
            (void)part_attr.GetValue<std::string>(partition_name_);
        }
    }

    // Only make path checks if model not provided as byte buffer
    bool make_secure_path_checks = !ort_graph.GetModelPath().empty();

    if (embed_mode)
    {
        // Get engine from byte stream.
        RETURN_IF_ORT_STATUS_ERROR(node.GetAttributeByName("ep_cache_context", node_attr));
        ENFORCE(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING);
        std::string context_binary;
        RETURN_IF_ORT_STATUS_ERROR(node_attr.GetValue<std::string>(context_binary));

        *(trt_rtx_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_rtx_runtime_->deserializeCudaEngine(const_cast<char*>(context_binary.c_str()),
                                                                                                           static_cast<size_t>(context_binary.length())));

        std::string message = "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep context node and deserialized it";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        if (!(*trt_rtx_engine_))
        {
            return ort_api.CreateStatus(ORT_EP_FAIL, "TensorRT EP could not deserialize engine from binary data");
        }

        if (weight_stripped_engine_refit_)
        {
            RETURN_IF_ORT_STATUS_ERROR(node.GetAttributeByName("onnx_model_filename", node_attr));
            std::string onnx_model_filename;
            RETURN_IF_ORT_STATUS_ERROR(node_attr.GetValue<std::string>(onnx_model_filename));
            auto status = ep_.RefitEngineImpl(onnx_model_filename,
                                              onnx_model_folder_path_,
                                              make_secure_path_checks,
                                              onnx_model_bytestream_,
                                              onnx_model_bytestream_size_,
                                              onnx_external_data_bytestream_,
                                              onnx_external_data_bytestream_size_,
                                              (*trt_rtx_engine_).get(),
                                              detailed_build_log_);
            if (status != nullptr)
            {
                return status;
            }
        }
    }
    else
    {
        // Get engine from cache file.
        RETURN_IF_ORT_STATUS_ERROR(node.GetAttributeByName("ep_cache_context", node_attr));
        std::string cache_path;
        RETURN_IF_ORT_STATUS_ERROR(node_attr.GetValue<std::string>(cache_path));

        // For security purpose, in the case of running context model, TRT EP won't allow
        // engine cache path to be the relative path like "../file_path" or the absolute path.
        // It only allows the engine cache to be in the same directory or sub directory of the context model.
        if (IsAbsolutePath(cache_path))
        {
            std::string message = "For security purpose, the ep_cache_context attribute should be set with a relative path, but it is an absolute path:  " + cache_path;
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }
        if (IsRelativePathToParentPath(cache_path))
        {
            std::string message = "The file path in ep_cache_context attribute has '..'. For security purpose, it's not allowed to point outside the directory.";
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }

        // The engine cache and context model (current model) should be in the same directory
        std::filesystem::path ctx_model_dir(GetPathOrParentPathOfCtxModel(ep_context_model_path_));
        auto engine_cache_path = ctx_model_dir.append(cache_path);

        std::string message = "[TensorRT EP] GetEpContextFromGraph engine_cache_path: " + engine_cache_path.string();
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

        // If it's a weight-stripped engine cache, it needs to be refitted even though the refit flag is not enabled
        if (!weight_stripped_engine_refit_)
        {
            weight_stripped_engine_refit_ = IsWeightStrippedEngineCache(engine_cache_path);
        }

        // If the serialized refitted engine is present, use it directly without refitting the engine again
        if (weight_stripped_engine_refit_)
        {
            const std::filesystem::path refitted_engine_cache_path = GetWeightRefittedEnginePath(engine_cache_path.string());
            if (std::filesystem::exists(refitted_engine_cache_path))
            {
                std::string refit_message = "[TensorRT EP] " + refitted_engine_cache_path.string() + " exists.";
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            refit_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
                engine_cache_path = refitted_engine_cache_path.string();
                weight_stripped_engine_refit_ = false;
            }
        }

        if (!std::filesystem::exists(engine_cache_path))
        {
            std::string error_msg =
                "TensorRT EP can't find engine cache: " + engine_cache_path.string() +
                ". Please make sure engine cache is in the same directory or sub-directory of context model.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        std::ifstream engine_file(engine_cache_path.string(), std::ios::binary | std::ios::in);
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);
        *(trt_rtx_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_rtx_runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        if (!(*trt_rtx_engine_))
        {
            std::string error_msg = "TensorRT EP could not deserialize engine from cache: " + engine_cache_path.string();
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        message = "[TensorRT EP] DeSerialized " + engine_cache_path.string();
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

        if (weight_stripped_engine_refit_)
        {
            RETURN_IF_ORT_STATUS_ERROR(node.GetAttributeByName("onnx_model_filename", node_attr));
            std::string onnx_model_filename;
            RETURN_IF_ORT_STATUS_ERROR(node_attr.GetValue<std::string>(onnx_model_filename));
            auto status = ep_.RefitEngineImpl(onnx_model_filename,
                                              onnx_model_folder_path_,
                                              make_secure_path_checks,
                                              onnx_model_bytestream_,
                                              onnx_model_bytestream_size_,
                                              onnx_external_data_bytestream_,
                                              onnx_external_data_bytestream_size_,
                                              (*trt_rtx_engine_).get(),
                                              detailed_build_log_);
            if (status != nullptr)
            {
                return status;
            }
        }
    }
    return nullptr;
}

//!
//! \brief Get the weight-refitted engine cache path from a weight-stripped engine cache path.
//!
//! Weight-stipped engine:
//! An engine with weights stripped and its size is smaller than a regualr engine.
//! The cache name of weight-stripped engine is TensorrtExecutionProvider_TRTKernel_XXXXX.stripped.engine
//!
//! Weight-refitted engine:
//! An engine that its weights have been refitted and it's simply a regular engine.
//! The cache name of weight-refitted engine is TensorrtExecutionProvider_TRTKernel_XXXXX.engine
//!
std::string GetWeightRefittedEnginePath(std::string stripped_engine_cache)
{
    std::filesystem::path stripped_engine_cache_path(stripped_engine_cache);
    std::string refitted_engine_cache_path = stripped_engine_cache_path.stem().stem().string() + ".engine";
    return refitted_engine_cache_path;
}

//!
//! \brief Get "EP context" model path.
//!
//! Function logic:
//! If ep_context_file_path is provided,
//!     - If ep_context_file_path is a file, return "ep_context_file_path".
//!     - If ep_context_file_path is a directory, return "ep_context_file_path/original_model_name_ctx.onnx".
//! If ep_context_file_path is not provided,
//!     - Return "original_model_name_ctx.onnx".
//!
//! TRT EP has rules about context model path and engine cache path (see tensorrt_execution_provider.cc):
//! - If dump_ep_context_model_ and engine_cache_enabled_ is enabled, TRT EP will dump context model and save engine cache
//!   to the same directory provided by ep_context_file_path_. (i.e. engine_cache_path_ = ep_context_file_path_)
//!
//! Example 1:
//! ep_context_file_path = "/home/user/ep_context_model_directory"
//! original_model_path = "model.onnx"
//! => return "/home/user/ep_context_model_folder/model_ctx.onnx"
//!
//! Example 2:
//! ep_context_file_path = "my_ctx_model.onnx"
//! original_model_path = "model.onnx"
//! => return "my_ctx_model.onnx"
//!
//! Example 3:
//! ep_context_file_path = "/home/user2/ep_context_model_directory/my_ctx_model.onnx"
//! original_model_path = "model.onnx"
//! => return "/home/user2/ep_context_model_directory/my_ctx_model.onnx"
//!
std::string GetCtxModelPath(const std::string& ep_context_file_path,
                            const std::string& original_model_path)
{
    std::string ctx_model_path;

    if (!ep_context_file_path.empty() && !std::filesystem::is_directory(ep_context_file_path))
    {
        ctx_model_path = ep_context_file_path;
    }
    else
    {
        std::filesystem::path model_path = original_model_path;
        std::filesystem::path model_name_stem = model_path.stem();  // model_name.onnx -> model_name
        std::string ctx_model_name = model_name_stem.string() + "_ctx.onnx";

        if (std::filesystem::is_directory(ep_context_file_path))
        {
            std::filesystem::path model_directory = ep_context_file_path;
            ctx_model_path = model_directory.append(ctx_model_name).string();
        }
        else
        {
            ctx_model_path = ctx_model_name;
        }
    }
    return ctx_model_path;
}

}  // namespace trt_rtx_ep
