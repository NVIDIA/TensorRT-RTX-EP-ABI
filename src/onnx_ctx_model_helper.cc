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

#include "tensorrt_rtx_execution_provider_utils.h"

#include "utils/ep_utils.h"
#include "utils/filesystem_utils.h"
#include "utils/path_string.h"
#include "utils/path_validation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <filesystem>
#include <iostream>
#include <limits>
#include <unordered_set>

#include "direct_cuda_stream_reader.h"
#include "file_stream_writer.h"
#include "onnx/onnx_pb.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trt_rtx_ep
{
namespace
{
// Keep asynchronous transfers while avoiding a fixed 64 MiB pinned-host
// allocation for every cached-engine load.
constexpr std::size_t kEngineStreamReaderSlotBytes = 64ULL * 1024 * 1024;
constexpr int kEngineStreamReaderRingDepth = 2;
constexpr uintmax_t kEngineStreamingThresholdBytes = 256ULL * 1024 * 1024;

nvinfer1::ICudaEngine* DeserializeMappedEngine(nvinfer1::IRuntime& runtime, const std::filesystem::path& path)
{
#if defined(_WIN32)
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("CreateFileW failed");
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0)
    {
        CloseHandle(file);
        throw std::runtime_error("GetFileSizeEx failed");
    }
    const HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    const void* data = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!data)
    {
        if (mapping)
            CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("MapViewOfFile failed");
    }
    const auto cleanup = [&]
    {
        UnmapViewOfFile(data);
        CloseHandle(mapping);
        CloseHandle(file);
    };
    try
    {
        auto* engine = runtime.deserializeCudaEngine(data, static_cast<size_t>(size.QuadPart));
        cleanup();
        return engine;
    }
    catch (...)
    {
        cleanup();
        throw;
    }
#else
    const int file = ::open(path.c_str(), O_RDONLY);
    if (file < 0)
        throw std::runtime_error("open failed");
    struct stat status{};
    if (::fstat(file, &status) != 0 || status.st_size < 0)
    {
        ::close(file);
        throw std::runtime_error("fstat failed");
    }
    void* data = ::mmap(nullptr, static_cast<size_t>(status.st_size), PROT_READ, MAP_PRIVATE, file, 0);
    ::close(file);
    if (data == MAP_FAILED)
        throw std::runtime_error("mmap failed");
    try
    {
        auto* engine = runtime.deserializeCudaEngine(data, static_cast<size_t>(status.st_size));
        ::munmap(data, static_cast<size_t>(status.st_size));
        return engine;
    }
    catch (...)
    {
        ::munmap(data, static_cast<size_t>(status.st_size));
        throw;
    }
#endif
}
}  // namespace

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
std::filesystem::path GetWeightRefittedEnginePath(const std::filesystem::path& stripped_engine_cache_path);

//!
//! \brief Return the directory where the ep context model locates.
//!
//! Semantics:
//!  - If ep_context_file_path ends with a path separator (e.g., "/tmp/out/"),
//!    it is treated as a directory and returned as-is.
//!  - Otherwise ep_context_file_path is treated as a file (or file stem), and
//!    its parent directory is returned. This matches both the SetOutputModelPath
//!    case (e.g., "dir/model.onnx" -> "dir") and the SetEpContextBinaryInformation
//!    case (e.g., "dir/stem" -> "dir"). We deliberately do NOT rely on
//!    std::filesystem::is_directory() here because a same-named directory may
//!    pre-exist on disk from earlier runs, which would otherwise flip the
//!    interpretation in a state-dependent way.
//!
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::filesystem::path& ep_context_file_path)
{
    if (ep_context_file_path.empty())
    {
        return std::filesystem::path();
    }
    const auto& native = ep_context_file_path.native();
    const PathChar back = native.back();
    // A trailing path separator alone implies directory semantics. We deliberately
    // do not probe the filesystem here so that interpretation is stable regardless
    // of whether a same-named directory happens to exist from a previous run.
    const bool has_trailing_sep = (back == PathChar('/') || back == PathChar('\\'));
    if (has_trailing_sep)
    {
        return ep_context_file_path;
    }
    return ep_context_file_path.parent_path();
}

bool IsWeightStrippedEngineCache(std::filesystem::path& engine_cache_path)
{
    // The weight-stripped engine cache has the naming of xxx.stripped.engine
    return engine_cache_path.stem().extension().string() == ".stripped";
}

EPContextNodeHelper::EPContextNodeHelper(TensorrtRtxExecutionProvider& ep, const OrtGraph* graph,
                                         const OrtNode* fused_node)
    : ApiPtrs{static_cast<const ApiPtrs&>(ep)}
    , ep_(ep)
    , graph_(graph)
    , fused_node_(fused_node)
{
}

EPContextNodeHelper::~EPContextNodeHelper()
{
    engine_stream_writer_.reset();
    if (!preserve_serialized_engine_on_destroy_ && !serialized_engine_path_.empty())
    {
        std::error_code error;
        std::filesystem::remove(serialized_engine_path_, error);
    }
}

OrtStatus* EPContextNodeHelper::InitializeEngineOutput(const std::filesystem::path& cache_root,
                                                       const std::string& cache_prefix, bool create_ep_context_node,
                                                       int64_t embed_mode)
{
    embed_mode_ = embed_mode;

    const char* fused_node_name = nullptr;
    RETURN_IF_ERROR(ort_api.Node_GetName(fused_node_, &fused_node_name));
    if (fused_node_name == nullptr || fused_node_name[0] == '\0')
        return ort_api.CreateStatus(ORT_EP_FAIL, "GraphPartitioner produced an unnamed fused node.");

    const std::string engine_cache_name = cache_prefix.empty()
                                              ? std::string(fused_node_name) + ".engine"
                                              : cache_prefix + "_" + std::string(fused_node_name) + ".engine";
    engine_cache_path_ = GetCachePath(cache_root, engine_cache_name);

    const bool can_stream_to_final_cache =
        create_ep_context_node && embed_mode_ == 0 && !ep_.HasEpContextDataWriteCallback();
    if (can_stream_to_final_cache)
    {
        serialized_engine_path_ = engine_cache_path_;
        engine_stream_writer_ = std::make_unique<FileStreamWriter>(serialized_engine_path_);
        return nullptr;
    }

    std::error_code error;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path(error);
    std::string reason;
    if (error)
    {
        reason = "the temporary directory is unavailable: " + error.message();
    }
    else
    {
        static std::atomic<uint64_t> sequence{0};
        std::filesystem::path temp_path =
            temp_dir / ("trt_rtx_ep_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                        "_" + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".engine");
        auto writer = std::make_unique<FileStreamWriter>(temp_path);
        if (writer->IsOpen())
        {
            serialized_engine_path_ = std::move(temp_path);
            engine_stream_writer_ = std::move(writer);
            return nullptr;
        }
        reason = PathToUTF8String(temp_path.native()) + " cannot be created";
    }

    const std::string message = "[NvTensorRTRTX EP] Building the serialized engine in memory because " + reason;
    OrtStatus* log_status = ort_api.Logger_LogMessage(&ep_.logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                      message.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
    if (log_status != nullptr)
    {
        ort_api.ReleaseStatus(log_status);
    }
    return nullptr;
}

bool EPContextNodeHelper::StreamsEngineOutput() const noexcept
{
    return engine_stream_writer_ != nullptr;
}

nvinfer1::IStreamWriter& EPContextNodeHelper::GetEngineStreamWriter()
{
    return *engine_stream_writer_;
}

void EPContextNodeHelper::SetSerializedEngine(std::unique_ptr<nvinfer1::IHostMemory> serialized_engine) noexcept
{
    serialized_engine_ = std::move(serialized_engine);
}

OrtStatus* EPContextNodeHelper::FinalizeEngineOutput()
{
    if (serialized_engine_)
    {
        if (serialized_engine_->size() == 0)
            return ort_api.CreateStatus(ORT_EP_FAIL, "Serialized engine was empty.");
        serialized_engine_size_ = serialized_engine_->size();
        return nullptr;
    }

    if (!engine_stream_writer_ || !engine_stream_writer_->Close())
    {
        std::string message = "Failed to close serialized engine stream";
        if (engine_stream_writer_ && !engine_stream_writer_->GetLastError().empty())
            message += ": " + engine_stream_writer_->GetLastError();
        return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    std::error_code error;
    const uintmax_t size = std::filesystem::file_size(serialized_engine_path_, error);
    if (error || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return ort_api.CreateStatus(ORT_EP_FAIL, "Serialized engine stream was empty or could not be read.");

    serialized_engine_size_ = static_cast<size_t>(size);
    return nullptr;
}

size_t EPContextNodeHelper::GetEngineSize() const noexcept
{
    return serialized_engine_size_;
}

const std::vector<char>& EPContextNodeHelper::GetEngineHeader()
{
    if (engine_header_.empty())
    {
        if (serialized_engine_)
        {
            const auto* data = static_cast<const char*>(serialized_engine_->data());
            const size_t header_size = (std::min)(serialized_engine_->size(), kTensorRTEngineHeaderSize);
            engine_header_.assign(data, data + header_size);
        }
        else if (!engine_data_.empty())
        {
            const size_t header_size = (std::min)(engine_data_.size(), kTensorRTEngineHeaderSize);
            engine_header_.assign(engine_data_.begin(), engine_data_.begin() + header_size);
        }
        else
        {
            engine_header_ = utils::ReadFile(serialized_engine_path_, ort_api, kTensorRTEngineHeaderSize);
        }
    }
    return engine_header_;
}

std::span<const char> EPContextNodeHelper::GetEngineData()
{
    if (serialized_engine_)
        return {static_cast<const char*>(serialized_engine_->data()), serialized_engine_->size()};
    if (engine_data_.empty())
        engine_data_ = utils::ReadFile(serialized_engine_path_, ort_api);
    return engine_data_;
}

//!
//! \brief Create an EPContext OrtNode from a fused_node.
//!
OrtStatus* EPContextNodeHelper::CreateEPContextNode(const std::string& compute_capability,
                                                    const std::filesystem::path& onnx_model_path,
                                                    std::vector<WeightlessRefitRecord>& refit_records,
                                                    bool persist_refit_table, bool refit_table_filtered_against_engine,
                                                    OrtNode** ep_context_node)
{
    // Initializers referenced by a captured weightless refit record must be kept as named
    // EPContext node inputs (instead of dropped like ordinary initializers) so ORT preserves them
    // as real initializers on the exported EPContext model and the replay path can resolve them by
    // name at load time. Everything else about the input/output list is unchanged.
    const std::vector<std::string> refit_source_names = CollectWeightlessRefitSourceNames(refit_records);
    const std::unordered_set<std::string> keep_initializer_names(refit_source_names.begin(), refit_source_names.end());

    // Helper to collect input or output names from an array of OrtValueInfo instances.
    // Only includes actual graph inputs, excluding initializers (constant weights) other than
    // the weightless-refit source names carved out above.
    auto collect_input_output_names = [&](const std::vector<const OrtValueInfo*>& value_infos,
                                          const std::unordered_set<std::string>& keep_names,
                                          std::vector<const char*>& result) -> OrtStatus*
    {
        size_t num_values = value_infos.size();
        std::vector<const char*> value_names;

        for (size_t i = 0; i < num_values; ++i)
        {
            const OrtValueInfo* value_info = value_infos[i];

            const char* value_name = nullptr;
            RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &value_name));

            // Defensive: ORT should only pair a null name with a non-OK status, but guard anyway --
            // keep_names.find(value_name) below (and the push_back) would otherwise construct a
            // std::string from nullptr, which is undefined behavior.
            if (value_name == nullptr)
            {
                continue;
            }

            // Check if this is an initializer (constant weight) - if so, skip it unless it's a
            // weightless-refit source that must be preserved.
            const OrtValue* initializer_value = nullptr;
            OrtStatus* status = ort_api.ValueInfo_GetInitializerValue(value_info, &initializer_value);
            if (status != nullptr)
            {
                ort_api.ReleaseStatus(status);
                // Error getting initializer value, treat as non-initializer
            }

            if (initializer_value != nullptr && keep_names.find(value_name) == keep_names.end())
            {
                continue;
            }

            // Check if this is a required graph input
            bool is_required_graph_input = true;
            // RETURN_IF_ERROR(ort_api.ValueInfo_IsRequiredGraphInput(value_info, &is_required_graph_input));

            if (is_required_graph_input)
            {
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

    RETURN_IF_ERROR(collect_input_output_names(fused_node_inputs, keep_initializer_names, /*out*/ input_names));
    RETURN_IF_ERROR(collect_input_output_names(fused_node_outputs, {}, /*out*/ output_names));

    // NOTE: a by-name keep-probe that appended every
    // CollectWeightlessRefitSourceNames entry to input_names here was tried and REVERTED. It fails at
    // compile ("<name> must be either specified in graph inputs or graph initializers") because ORT's
    // CleanUnusedInitializers prunes the internal refit-source initializers (e.g. *_pre_quant_scale,
    // cos_cache) BEFORE this node is created, so there is nothing to reference. The EP also has no
    // mutable output OrtGraph in the OrtEp::Compile ABI (it only returns an OrtNode), so
    // AddInitializerToGraph cannot re-inject them here either. The real fix needs either
    // prune-suppression or an ORT-side change.

    // Fail-fast (weightless refit): every refit source must have survived as a named EPContext node
    // input (via the keep_initializer_names carve-out above, which only picks it up if the keep-fix
    // drop_constant_initializers=false actually kept it in fused_node_inputs). If ORT's
    // CleanUnusedInitializers pruned one before this node was created (see the NOTE above), it will NOT
    // be an initializer on the exported model, so the serialized ep_refit_table would reference a name
    // the replay path can never resolve -> a silently unusable weight-stripped model.
    //
    // When refit_table_filtered_against_engine is true, the caller already filtered refit_records
    // against the live built engine's IRefitter, so every remaining record is expected to be genuinely
    // needed: a missing source here is a real bug. Reject at export time naming the offending source
    // instead of shipping a broken model.
    //
    // When it is false (compile-only mode: no live engine was available to filter against), refit_records
    // is still the parser's unfiltered superset, so a missing source may simply belong to a weight the
    // builder folded away -- not a bug. Drop just the affected record(s) instead of aborting the whole
    // export; WeightlessRefitEngineImpl's load-time coverage check remains the fail-closed backstop for a
    // genuine gap.
    if (!refit_source_names.empty())
    {
        const std::unordered_set<std::string> preserved_input_names(input_names.begin(), input_names.end());
        std::unordered_set<std::string> missing_sources;
        for (const std::string& src : refit_source_names)
        {
            if (preserved_input_names.find(src) == preserved_input_names.end())
            {
                if (refit_table_filtered_against_engine)
                {
                    const std::string msg = "Weightless refit source '" + src +
                                            "' was not preserved as an EPContext node input (pruned before "
                                            "node creation); the ep_refit_table would be unusable at load. "
                                            "Aborting EPContext export.";
                    return ort_api.CreateStatus(ORT_FAIL, msg.c_str());
                }
                missing_sources.insert(src);
            }
        }

        if (!missing_sources.empty())
        {
            const size_t num_records_before = refit_records.size();
            refit_records.erase(std::remove_if(refit_records.begin(), refit_records.end(),
                                               [&](const WeightlessRefitRecord& record)
                                               {
                                                   for (const auto& name : record.source_onnx_names)
                                                   {
                                                       if (missing_sources.find(name) != missing_sources.end())
                                                       {
                                                           return true;
                                                       }
                                                   }
                                                   return false;
                                               }),
                                refit_records.end());
            const std::string msg =
                "[NvTensorRTRTX EP] " + std::to_string(num_records_before - refit_records.size()) +
                " weightless refit record(s) referenced a source pruned before EPContext node creation "
                "(likely a weight the builder folded away; the parse-time superset could not be filtered "
                "against a live engine in compile-only mode) and were dropped from the persisted "
                "ep_refit_table.";
            OrtStatus* log_status = ort_api.Logger_LogMessage(&ep_.logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                                              msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
            if (log_status != nullptr)
            {
                ort_api.ReleaseStatus(log_status);
            }
        }
    }

    // Serialize the captured weightless refit table (if any) up front so we know whether the
    // node needs 9 or 10 attributes before allocating the attributes vector below.
    std::string refit_table_bytes;
    if (persist_refit_table || !refit_records.empty())
    {
        refit_table_bytes = SerializeWeightlessRefitTable(refit_records);
        if (refit_table_bytes.empty())
        {
            return ort_api.CreateStatus(ORT_FAIL,
                                        "Failed to serialize weightless refit table (too large or too many records).");
        }
    }
    const bool has_refit_table = !refit_table_bytes.empty();

    // Create node attributes. The CreateNode() function copies the attributes, so we have to release them.
    std::vector<OrtOpAttr*> attributes(has_refit_table ? 10 : 9, nullptr);
    DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

    RETURN_IF_ERROR(
        ort_api.CreateOpAttr(EMBED_MODE.c_str(), &embed_mode_, sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[0]));

    std::string engine_data_str = "";
    if (embed_mode_)
    {
        const auto& engine_data = GetEngineData();
        const size_t size = engine_data.size();
        // OrtApi::CreateOpAttr's len parameter is a 32-bit int, so an engine at or above
        // INT_MAX bytes (~2GB) cannot be embedded as a single attribute without silently
        // truncating. Reject explicitly instead of corrupting the engine cache.
        if (size > static_cast<size_t>(INT_MAX))
        {
            return ort_api.CreateStatus(
                ORT_FAIL, ("Engine size (" + std::to_string(size) + " bytes) exceeds the " + std::to_string(INT_MAX) +
                           "-byte (~2GB) limit for embed_mode=1 (ORT's CreateOpAttr uses a 32-bit "
                           "length). Use embed_mode=0 to write this engine to an external cache "
                           "file instead.")
                              .c_str());
        }
        if (size > 0)
        {
            engine_data_str.assign(engine_data.data(), size);
        }
        RETURN_IF_ERROR(ort_api.CreateOpAttr(EP_CACHE_CONTEXT.c_str(), engine_data_str.c_str(),
                                             static_cast<int>(engine_data_str.size()), ORT_OP_ATTR_STRING,
                                             &attributes[1]));
    }
    else
    {
        std::filesystem::path attr_path = engine_cache_path_;
        if (IsAbsolutePath(PathToUTF8String(attr_path.native())))
        {
            std::filesystem::path ctx_model_dir =
                GetPathOrParentPathOfCtxModel(std::filesystem::path(ToPathString(ep_.GetEpContextFilePath())));
            if (!ctx_model_dir.empty())
            {
                std::error_code ec;
                std::filesystem::path rel_path = std::filesystem::relative(engine_cache_path_, ctx_model_dir, ec);
                if (!ec && !rel_path.empty())
                {
                    attr_path = rel_path;
                }
            }
        }
        {
            std::string attr_path_utf8 = PathToUTF8String(attr_path.native());
            if (IsAbsolutePath(attr_path_utf8) || IsRelativePathToParentPath(attr_path_utf8))
            {
                attr_path = engine_cache_path_.filename();
            }
        }
        // The EP_CACHE_CONTEXT attribute is stored as text inside the ONNX EPContext node and
        // read back via GetAttributeByName<std::string>; encode as UTF-8 for portable round-trip.
        std::string attr_path_utf8 = PathToUTF8String(attr_path.native());
        if (ep_.HasEpContextDataWriteCallback())
        {
            const auto& engine_data = GetEngineData();
            RETURN_IF_ERROR(ep_.WriteEpContextData(attr_path_utf8.c_str(), engine_data.data(), engine_data.size()));
        }
        RETURN_IF_ERROR(ort_api.CreateOpAttr(EP_CACHE_CONTEXT.c_str(), attr_path_utf8.c_str(),
                                             static_cast<int>(attr_path_utf8.size()), ORT_OP_ATTR_STRING,
                                             &attributes[1]));
    }

    std::string onnx_model_filename = PathToUTF8String(onnx_model_path.filename().native());
    RETURN_IF_ERROR(ort_api.CreateOpAttr(COMPUTE_CAPABILITY.c_str(), compute_capability.c_str(),
                                         static_cast<int>(compute_capability.size()), ORT_OP_ATTR_STRING,
                                         &attributes[2]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr(ONNX_MODEL_FILENAME.c_str(), onnx_model_filename.c_str(),
                                         static_cast<int>(onnx_model_filename.size()), ORT_OP_ATTR_STRING,
                                         &attributes[3]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr(PARTITION_NAME.c_str(), fused_node_name,
                                         static_cast<int>(strlen(fused_node_name)), ORT_OP_ATTR_STRING,
                                         &attributes[4]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr(SDK_VERSION.c_str(), "1.0", 3, ORT_OP_ATTR_STRING, &attributes[5]));
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_context_op_domain", EPCONTEXT_OP_DOMAIN.c_str(),
                                         static_cast<int>(EPCONTEXT_OP_DOMAIN.size()), ORT_OP_ATTR_STRING,
                                         &attributes[6]));

    int64_t main_context = 0;
    RETURN_IF_ERROR(
        ort_api.CreateOpAttr(MAIN_CONTEXT.c_str(), &main_context, sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[7]));

    const std::string& ep_source = ep_.name_;
    RETURN_IF_ERROR(ort_api.CreateOpAttr(SOURCE.c_str(), ep_source.c_str(), static_cast<int>(ep_source.size()),
                                         ORT_OP_ATTR_STRING, &attributes[8]));

    if (has_refit_table)
    {
        RETURN_IF_ERROR(ort_api.CreateOpAttr(EP_REFIT_TABLE.c_str(), refit_table_bytes.c_str(),
                                             static_cast<int>(refit_table_bytes.size()), ORT_OP_ATTR_STRING,
                                             &attributes[9]));
    }

    RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name, input_names.data(),
                                                input_names.size(), output_names.data(), output_names.size(),
                                                attributes.data(), attributes.size(), ep_context_node));

    // The direct-to-cache path is now safe to retain: the EPContext node was created successfully and references
    // this engine. Until this point the destructor removes it so partial build output cannot be mistaken for a
    // usable cache on a later compile/load.
    if (serialized_engine_path_ == engine_cache_path_)
        preserve_serialized_engine_on_destroy_ = true;

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

OrtStatus* EPContextNodeReader::TryWeightlessRefit(const Ort::ConstNode& node, bool& handled)
{
    handled = false;

    Ort::ConstOpAttr refit_table_attr;
    if (!node.GetAttributeByName(EP_REFIT_TABLE.c_str(), refit_table_attr).IsOK() ||
        static_cast<const OrtOpAttr*>(refit_table_attr) == nullptr)
    {
        // No weightless refit table on this node -- caller falls back to the legacy,
        // original-ONNX-requiring RefitEngineImpl path unchanged.
        return nullptr;
    }
    ENFORCE(refit_table_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING);

    std::string refit_table_bytes;
    RETURN_IF_ORT_STATUS_ERROR(refit_table_attr.GetValue<std::string>(refit_table_bytes));

    std::vector<WeightlessRefitRecord> records;
    if (!DeserializeWeightlessRefitTable(refit_table_bytes, records))
    {
        return ort_api.CreateStatus(
            ORT_EP_FAIL, "[TensorRT EP] Failed to parse ep_refit_table attribute (corrupt or version mismatch).");
    }

    // Weightless refit sources are preserved as named initializer inputs on the EPContext node
    // itself (see CreateEPContextNode), so their data is available directly from this node's
    // inputs -- no original ONNX model needed at all.
    std::unordered_map<std::string, std::pair<const void*, size_t>> weight_data_by_name;
    for (const auto& value_info : node.GetInputs())
    {
        Ort::ConstValue initializer_value{nullptr};
        OrtStatus* status = value_info.GetInitializer(initializer_value);
        if (status != nullptr)
        {
            ort_api.ReleaseStatus(status);
            continue;
        }
        if (!initializer_value || !initializer_value.IsTensor())
        {
            continue;
        }
        weight_data_by_name.emplace(value_info.GetName(),
                                    std::make_pair(initializer_value.GetTensorRawData(),
                                                   static_cast<size_t>(initializer_value.GetTensorSizeInBytes())));
    }

    RETURN_IF_ERROR(
        ep_.WeightlessRefitEngineImpl(records, weight_data_by_name, (*trt_rtx_engine_).get(), detailed_build_log_));
    handled = true;
    return nullptr;
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

    // Get "partition_name" (fused node name at build time) for runtime cache path consistency across sessions.
    //
    // NOTE: The Ort C++ wrapper Node_GetAttributeByName returns IsOK()==true for
    // attributes that are NOT present, leaving the attr as a null OrtOpAttr*.
    // Calling GetType() on that null pointer causes an access violation. Guard here.
    partition_name_.clear();
    {
        Ort::ConstOpAttr part_attr;
        if (node.GetAttributeByName(PARTITION_NAME.c_str(), part_attr).IsOK() &&
            static_cast<const OrtOpAttr*>(part_attr) != nullptr &&
            part_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING)
        {
            (void)part_attr.GetValue<std::string>(partition_name_);
        }
    }

    if (embed_mode)
    {
        // Get engine from byte stream.
        RETURN_IF_ORT_STATUS_ERROR(node.GetAttributeByName("ep_cache_context", node_attr));
        ENFORCE(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING);
        std::string context_binary;
        RETURN_IF_ORT_STATUS_ERROR(node_attr.GetValue<std::string>(context_binary));

        {
            auto api_lock = ep_.GetApiLock();
            *(trt_rtx_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_rtx_runtime_->deserializeCudaEngine(
                const_cast<char*>(context_binary.c_str()), static_cast<size_t>(context_binary.length())));
        }

        std::string message = "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep "
                              "context node and deserialized it";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        if (!(*trt_rtx_engine_))
        {
            return ort_api.CreateStatus(ORT_EP_FAIL, "TensorRT EP could not deserialize engine from binary data");
        }

        bool weightless_refit_handled = false;
        RETURN_IF_ERROR(TryWeightlessRefit(node, weightless_refit_handled));

        if (!weightless_refit_handled && weight_stripped_engine_refit_)
        {
            return ort_api.CreateStatus(ORT_EP_FAIL,
                                        "[NvTensorRTRTX EP] weight-stripped engine has no ep_refit_table attribute. "
                                        "Recompile the EPContext model with the weightless option enabled against "
                                        "TensorRT-RTX 1.7 or later.");
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
        const PathClassification cache_path_check = ClassifyPath(cache_path);
        if (cache_path_check.is_absolute)
        {
            std::string message = "For security purpose, the ep_cache_context attribute should be set with a relative "
                                  "path, but it is an absolute path:  " +
                                  cache_path;
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }
        if (cache_path_check.has_parent_traversal)
        {
            std::string message = "The file path in ep_cache_context attribute has '..'. For security purpose, it's "
                                  "not allowed to point outside the directory.";
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }
        if (cache_path_check.has_root_component)
        {
            std::string message = "For security purpose, the ep_cache_context attribute must be a plain relative "
                                  "path with no drive letter or leading path separator, but it is: " +
                                  cache_path;
            return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
        }

        // The engine cache and context model (current model) should be in the same directory.
        // When the context model is loaded from a buffer, ep_context_model_path_ is empty,
        // so there is no model-file location to anchor the relative ep_cache_context path.
        // Fall back to the ep_context_file_path session option (set via
        // kOrtSessionOptionEpContextFilePath), which the caller is expected to provide to
        // locate external engine binaries in that scenario (matches the QNN EP pattern).
        const std::string& effective_ctx_path =
            !ep_context_model_path_.empty() ? ep_context_model_path_ : ep_.GetEpContextFilePath();
        std::filesystem::path ctx_model_dir =
            GetPathOrParentPathOfCtxModel(std::filesystem::path(ToPathString(effective_ctx_path)));
        std::filesystem::path engine_cache_path = ctx_model_dir;
        engine_cache_path /= std::filesystem::path(ToPathString(cache_path));

        std::string message =
            "[TensorRT EP] GetEpContextFromGraph engine_cache_path: " + PathToUTF8String(engine_cache_path.native());
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

        // If it's a weight-stripped engine cache, it needs to be refitted even though the refit flag is not enabled
        if (!weight_stripped_engine_refit_)
        {
            weight_stripped_engine_refit_ = IsWeightStrippedEngineCache(engine_cache_path);
        }

        // If the serialized refitted engine is present, use it directly without refitting the engine again.
        // GetWeightRefittedEnginePath returns a basename (e.g. "TRTKernel_XXXXX.engine"); root it under
        // engine_cache_path.parent_path() so the existence probe is relative to the cache dir, not CWD.
        // Set when an already-refitted full engine is selected from cache below. The engine we load is
        // then complete (not weight-stripped), so neither the weightless ep_refit_table nor the legacy
        // refit path must run on it again.
        bool already_refitted_cache_used = false;
        // Only the .stripped.engine convention identifies a separate refitted cache. CompileModel
        // can write stripped engines with ordinary .engine names, which must not select themselves.
        if (weight_stripped_engine_refit_ && IsWeightStrippedEngineCache(engine_cache_path))
        {
            const std::filesystem::path refit_name = GetWeightRefittedEnginePath(engine_cache_path).filename();
            const std::filesystem::path refitted_engine_cache_path = engine_cache_path.parent_path() / refit_name;
            if (std::filesystem::exists(refitted_engine_cache_path))
            {
                std::string refit_message =
                    "[TensorRT EP] " + PathToUTF8String(refitted_engine_cache_path.native()) + " exists.";
                Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                            refit_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
                engine_cache_path = refitted_engine_cache_path;
                weight_stripped_engine_refit_ = false;
                already_refitted_cache_used = true;
            }
        }

        // The callback supplies a complete engine buffer, so it intentionally bypasses
        // the file probe and DirectCudaStreamReader path below. The callback API has no
        // range/streaming contract; using it therefore trades streaming for custom I/O.
        OrtAllocator* callback_allocator = nullptr;
        void* callback_engine_data = nullptr;
        size_t callback_engine_size = 0;
        const bool use_ep_context_read_callback = ep_.HasEpContextDataReadCallback();
        if (use_ep_context_read_callback)
        {
            RETURN_IF_ERROR(ort_api.GetAllocatorWithDefaultOptions(&callback_allocator));
            OrtStatus* read_status = ep_.ReadEpContextData(cache_path.c_str(), callback_allocator,
                                                           &callback_engine_data, &callback_engine_size);
            if (read_status != nullptr)
            {
                if (callback_engine_data != nullptr)
                    callback_allocator->Free(callback_allocator, callback_engine_data);
                return read_status;
            }
            if (callback_engine_data == nullptr || callback_engine_size == 0)
            {
                if (callback_engine_data != nullptr)
                    callback_allocator->Free(callback_allocator, callback_engine_data);
                return ort_api.CreateStatus(ORT_EP_FAIL, "EPContext read callback returned an empty engine.");
            }
        }

        if (!use_ep_context_read_callback && !std::filesystem::exists(engine_cache_path))
        {
            std::string error_msg =
                "TensorRT EP can't find engine cache: " + PathToUTF8String(engine_cache_path.native()) +
                ". Please make sure engine cache is in the same directory or sub-directory of context model.";
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        std::error_code file_size_error;
        const uintmax_t engine_size =
            use_ep_context_read_callback ? 0 : std::filesystem::file_size(engine_cache_path, file_size_error);
        const bool use_streaming_reader =
            !use_ep_context_read_callback && !file_size_error && engine_size >= kEngineStreamingThresholdBytes;
        std::unique_ptr<DirectCudaStreamReader> stream_reader;
        const bool defer_weights = use_streaming_reader && deferred_weights_loading_ && !weight_stripped_engine_refit_;
        try
        {
            auto api_lock = ep_.GetApiLock();
            if (use_ep_context_read_callback)
            {
                *(trt_rtx_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(
                    trt_rtx_runtime_->deserializeCudaEngine(callback_engine_data, callback_engine_size));
                callback_allocator->Free(callback_allocator, callback_engine_data);
                callback_engine_data = nullptr;
            }
            else if (use_streaming_reader)
            {
                stream_reader = std::make_unique<DirectCudaStreamReader>(
                    engine_cache_path, kEngineStreamReaderSlotBytes, kEngineStreamReaderRingDepth);
                const bool previous_deferred_setting = trt_rtx_runtime_->getDeferredWeightsLoading();
                trt_rtx_runtime_->setDeferredWeightsLoading(defer_weights);
                try
                {
                    *(trt_rtx_engine_) =
                        std::unique_ptr<nvinfer1::ICudaEngine>(trt_rtx_runtime_->deserializeCudaEngine(*stream_reader));
                }
                catch (...)
                {
                    trt_rtx_runtime_->setDeferredWeightsLoading(previous_deferred_setting);
                    throw;
                }
                trt_rtx_runtime_->setDeferredWeightsLoading(previous_deferred_setting);
            }
            else
            {
                *(trt_rtx_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(
                    DeserializeMappedEngine(*trt_rtx_runtime_, engine_cache_path));
            }
        }
        catch (const std::exception& error)
        {
            if (callback_engine_data != nullptr)
                callback_allocator->Free(callback_allocator, callback_engine_data);
            const std::string error_msg = "TensorRT EP failed to deserialize engine cache " +
                                          PathToUTF8String(engine_cache_path.native()) + ": " + error.what();
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }
        if (!(*trt_rtx_engine_))
        {
            std::string error_msg =
                "TensorRT EP could not deserialize engine from cache: " + PathToUTF8String(engine_cache_path.native());
            if (stream_reader && !stream_reader->GetLastError().empty())
                error_msg += ": " + stream_reader->GetLastError();
            return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
        }

        if (stream_reader && defer_weights && !(*trt_rtx_engine_)->areWeightsLoaded())
            engine_stream_reader_ = std::move(stream_reader);

        message = "[TensorRT EP] DeSerialized " + PathToUTF8String(engine_cache_path.native());
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

        // Skip the refit-table application entirely when an already-refitted engine was loaded from cache
        // above (fast path): that engine is already complete, so pre-mark it handled instead of calling
        // TryWeightlessRefit, which would redundantly (and possibly incorrectly) reapply the ep_refit_table.
        bool weightless_refit_handled = already_refitted_cache_used;
        if (!weightless_refit_handled)
        {
            RETURN_IF_ERROR(TryWeightlessRefit(node, weightless_refit_handled));
        }

        if (!weightless_refit_handled && weight_stripped_engine_refit_)
        {
            return ort_api.CreateStatus(ORT_EP_FAIL,
                                        "[NvTensorRTRTX EP] weight-stripped engine has no ep_refit_table attribute. "
                                        "Recompile the EPContext model with the weightless option enabled against "
                                        "TensorRT-RTX 1.7 or later.");
        }
    }
    return nullptr;
}

OrtStatus* EPContextNodeReader::LoadDeferredWeights(cudaStream_t stream)
{
    if (!engine_stream_reader_)
        return nullptr;
    if (!trt_rtx_engine_ || !(*trt_rtx_engine_))
        return ort_api.CreateStatus(ORT_EP_FAIL, "TensorRT EP cannot load deferred weights without an engine");

    if (!(*trt_rtx_engine_)->loadWeightsAsync(*engine_stream_reader_, stream))
    {
        auto* reader = static_cast<DirectCudaStreamReader*>(engine_stream_reader_.get());
        std::string error_msg = "TensorRT EP failed to enqueue deferred engine weights from disk";
        if (!reader->GetLastError().empty())
            error_msg += ": " + reader->GetLastError();
        engine_stream_reader_.reset();
        return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    // The stream used by ORT for the first inference is not known during compilation and may differ from `stream`.
    // Finish the upload here before releasing the pinned staging buffers so every later execution stream is safe.
    const cudaError_t error = cudaStreamSynchronize(stream);
    engine_stream_reader_.reset();
    if (error != cudaSuccess)
    {
        const std::string error_msg = std::string("TensorRT EP failed to synchronize deferred engine weight upload: ") +
                                      cudaGetErrorString(error);
        return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
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
std::filesystem::path GetWeightRefittedEnginePath(const std::filesystem::path& stripped_engine_cache_path)
{
    // Preserve the original (43b03293) semantics: return only the basename of the refitted
    // engine cache (e.g. "TRTKernel_XXXXX.engine"), not the full path. The caller probes
    // for that name relative to its own working directory. Whether CWD-relative is the
    // right resolution is a separate question; this fix does not change that behavior.
    std::filesystem::path filename = stripped_engine_cache_path.filename();
    filename.replace_extension();                         // drop ".engine"
    filename.replace_extension(ToPathString(".engine"));  // drop ".stripped", append ".engine"
    return filename;
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
//! - If dump_ep_context_model_ and engine_cache_enabled_ is enabled, TRT EP will dump context model and save engine
//! cache
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
std::filesystem::path GetCtxModelPath(const std::filesystem::path& ep_context_file_path,
                                      const std::filesystem::path& original_model_path)
{
    std::filesystem::path ctx_model_path;

    if (!ep_context_file_path.empty() && !std::filesystem::is_directory(ep_context_file_path))
    {
        ctx_model_path = ep_context_file_path;
    }
    else
    {
        // model_name.onnx -> model_name -> model_name + "_ctx.onnx"
        std::filesystem::path ctx_model_name = original_model_path.stem();
        ctx_model_name += ToPathString("_ctx.onnx");

        if (std::filesystem::is_directory(ep_context_file_path))
        {
            ctx_model_path = ep_context_file_path / ctx_model_name;
        }
        else
        {
            ctx_model_path = ctx_model_name;
        }
    }
    return ctx_model_path;
}

}  // namespace trt_rtx_ep
