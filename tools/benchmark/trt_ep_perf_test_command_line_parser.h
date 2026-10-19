// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define DLL_NAME(name) (DLL_PREFIX name DLL_SUFFIX)
#ifdef _WIN32
#define DLL_PREFIX ""
#define DLL_SUFFIX ".dll"
#else
#define DLL_PREFIX "lib"
#define DLL_SUFFIX ".so"
#endif

enum class IOBindingAllocator
{
    Cpu,
    HostAccessible,
    EpDefault,
};

struct Arguments
{
    std::filesystem::path ep;
    std::string ep_name{"NvTensorRTRTXExecutionProvider"};
    bool context_embed_mode = true;
    std::filesystem::path model;
    std::filesystem::path context;
    std::filesystem::path runtime_cache;
    std::filesystem::path output;
    std::unordered_map<std::string, int64_t> shapes;
    bool overwrite_shapes = false;
    std::vector<std::pair<std::string, std::string>> ep_options;
    bool skip_compile = false;
    bool load_only = false;
    IOBindingAllocator value_allocator = IOBindingAllocator::Cpu;
    std::optional<uint32_t> warmup_iterations;
    std::optional<double> warmup_duration_seconds;
    std::optional<uint32_t> run_iterations;
    std::optional<double> run_duration_seconds;
    uint32_t time_compile_iterations = 1;
    uint32_t time_deserialize_iterations = 1;
    bool async = false;
    bool disable_io = false;
};

class CommandLineParser
{
public:
    static Arguments Parse(int argc, char** argv);
};
