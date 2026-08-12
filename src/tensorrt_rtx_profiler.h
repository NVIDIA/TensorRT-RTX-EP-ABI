// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nv_includes.h"

#include <ctime>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#if defined(_WIN32)
#include <process.h>  // _getpid()
#else
#include <unistd.h>  // getpid()
#endif

namespace trt_rtx_ep
{

//!
//! \brief IProfiler implementation that accumulates per-layer GPU timings grouped by EP context node,
//!        and writes Chrome tracing JSON that FTK can correlate 1-to-1 with ORT's kernel_time entries.
//!
//! Usage:
//!   1. Attach to every IExecutionContext via setProfiler() at context-creation time.
//!   2. Call BeginSection(fused_node_name) immediately before each enqueueV3() call.
//!   3. Call FlushToFile() in OnRunEndImpl() - writes and clears the current run's data.
//!
//! pid / tid semantics (stable across runs within a session):
//!   pid = subgraph index - assigned once per unique fused_node_name, never changes.
//!   tid = run counter   - increments each time BeginSection is called for the same name.
//!
//! Example: 2 subgraphs, 3 runs ->
//!   Run 1: (pid=0,tid=0) fused_node_0  (pid=1,tid=0) fused_node_1
//!   Run 2: (pid=0,tid=1) fused_node_0  (pid=1,tid=1) fused_node_1
//!   Run 3: (pid=0,tid=2) fused_node_0  (pid=1,tid=2) fused_node_1
//!
//! In Perfetto: pid groups by subgraph (one lane per subgraph), tid disambiguates runs.
//! For FTK: section count per flush == number of enqueueV3 calls in that run.
//!
class TrtRtxProfiler : public nvinfer1::IProfiler
{
public:
    //! Open a new section for \p context_name. Must be called immediately before each enqueueV3().
    //! Assigns a stable pid on first call for a given name; increments tid on every subsequent call.
    void BeginSection(const std::string& context_name)
    {
        // emplace assigns a stable pid on first appearance; subsequent calls are no-ops.
        const auto [pid_it, inserted] = context_to_pid_.emplace(context_name, next_pid_);
        if (inserted)
        {
            ++next_pid_;
            context_run_count_.emplace(context_name, 0);
        }

        const int32_t pid = pid_it->second;
        const int32_t tid = context_run_count_[context_name]++;

        sections_.push_back({context_name, pid, tid, {}});
    }

    //! Store a TRT-layer-name -> ONNX-node-names mapping extracted from IEngineInspector.
    //! Call once per engine after engine creation when profiling is enabled.
    //! Mapping is keyed by TRT layer name; value is a comma-separated list of ONNX node names.
    void SetLayerOnnxMapping(std::unordered_map<std::string, std::string> mapping)
    {
        // Merge - do not replace. Models with multiple subgraphs call this once per engine;
        // each engine contributes its own layer names so all must coexist in the map.
        for (auto& [k, v] : mapping)
        {
            layer_onnx_map_.insert_or_assign(std::move(k), std::move(v));
        }
    }

    void reportLayerTime(const char* layerName, float ms) noexcept override
    {
        try
        {
            if (sections_.empty())
            {
                // Guard: BeginSection was not called - create an anonymous section so data is never lost.
                sections_.push_back({"<unknown>", next_pid_++, 0, {}});
                context_run_count_["<unknown>"] = 1;
            }
            sections_.back().layers.push_back({layerName ? layerName : "", ms});
        }
        catch (...)
        {
            // Silently drop timing data on allocation failure - profiling is best-effort.
        }
    }

    //! Write all accumulated sections as Chrome tracing JSON to \p path.
    //! Sections accumulate across runs - each call appends the current run's data and rewrites
    //! the file with the complete history. This ensures FTK sees all N runs x M subgraphs.
    //! pid/tid assignment maps are preserved so values remain consistent across runs.
    //! No-op if there is nothing to write or the file cannot be opened (data is retained for next call).
    void FlushToFile(std::string_view path)
    {
        if (sections_.empty())
        {
            return;
        }

        std::ofstream f{std::string(path)};
        if (!f.is_open())
        {
            fprintf(stderr, "[NvTensorRTRTX EP] TrtRtxProfiler: failed to open profile output file: %.*s\n",
                    static_cast<int>(path.size()), path.data());
            sections_.clear();  // prevent unbounded growth if path is permanently invalid
            return;
        }

        f << "[\n";
        bool first_entry = true;

        for (const auto& sec : sections_)
        {
            // Skip sections with no layers - can occur if enqueueV3 fails after BeginSection.
            if (sec.layers.empty())
            {
                continue;
            }

            // process_name metadata event - labels this pid's track in Perfetto.
            if (!first_entry)
                f << ",\n";
            f << "  {\"name\": \"process_name\", \"ph\": \"M\""
              << ", \"pid\": " << sec.pid << ", \"args\": {\"name\": \"" << EscapeJson(sec.name) << "\"}}";
            first_entry = false;

            // Layer X events - ts is cumulative within this section, starting at 0.
            double ts = 0.0;
            for (const auto& layer : sec.layers)
            {
                double dur_us = static_cast<double>(layer.ms) * 1000.0;
                f << ",\n  {\"name\": \"" << EscapeJson(layer.name) << "\", \"ph\": \"X\""
                  << ", \"cat\": \"nv::trt::layer\""
                  << ", \"ts\": " << static_cast<long long>(ts) << ", \"dur\": " << static_cast<long long>(dur_us)
                  << ", \"pid\": " << sec.pid << ", \"tid\": " << sec.tid;
                const auto onnx_it = layer_onnx_map_.find(layer.name);
                if (onnx_it != layer_onnx_map_.end() && !onnx_it->second.empty())
                {
                    f << ", \"args\": {\"onnx_nodes\": \"" << EscapeJson(onnx_it->second) << "\"}";
                }
                f << "}";
                ts += dur_us;
            }
        }

        f << "\n]\n";
        // sections_ intentionally kept - accumulates across runs so FlushToFile always writes
        // the complete history (all N runs × M subgraphs), matching the N×M kernel_time
        // entries in ORT's profile that FTK needs to correlate 1-to-1.
    }

    //! Generate a timestamped output filename suitable as a default when the user provides none.
    static std::string GenerateOutputFilePath()
    {
        std::time_t t = std::time(nullptr);
        char buf[64] = {};
        struct tm stm = {};
#if defined(_MSC_VER)
        gmtime_s(&stm, &t);
#else
        gmtime_r(&t, &stm);
#endif
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &stm);
#if defined(_WIN32)
        const auto pid = static_cast<unsigned long>(_getpid());
#else
        const auto pid = static_cast<unsigned long>(getpid());
#endif
        return std::string("trt_rtx_profile_") + buf + "_" + std::to_string(pid) + ".json";
    }

private:
    struct LayerTiming
    {
        std::string name;
        float ms;
    };

    struct Section
    {
        std::string name;
        int32_t pid;  //!< Stable subgraph index - same name always gets the same pid.
        int32_t tid;  //!< Run counter - increments each BeginSection call for this name.
        std::vector<LayerTiming> layers;
    };

    std::vector<Section> sections_;

    // TRT layer name -> comma-separated ONNX node names (populated via SetLayerOnnxMapping).
    std::unordered_map<std::string, std::string> layer_onnx_map_;

    // Persistent across FlushToFile calls so pid/tid stay consistent for the session lifetime.
    std::unordered_map<std::string, int32_t> context_to_pid_;
    std::unordered_map<std::string, int32_t> context_run_count_;
    int32_t next_pid_ = 0;

    static std::string EscapeJson(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s)
        {
            switch (c)
            {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20)
                {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                }
                else
                {
                    out += static_cast<char>(c);
                }
                break;
            }
        }
        return out;
    }
};

}  // namespace trt_rtx_ep
