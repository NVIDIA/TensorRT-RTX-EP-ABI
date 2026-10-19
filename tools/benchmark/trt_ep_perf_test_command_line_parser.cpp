// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "trt_ep_perf_test_command_line_parser.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
using OptionValueParser = void (*)(Arguments&, std::string_view);

struct Option
{
    std::string_view name;
    std::string_view value_name;
    std::string_view help;
    bool requires_value;
    OptionValueParser parse;
};

void ParseEp(Arguments& arguments, std::string_view value)
{
    arguments.ep = value;
}
void ParseEpName(Arguments& arguments, std::string_view value)
{
    arguments.ep_name = value;
}
void ParseModel(Arguments& arguments, std::string_view value)
{
    arguments.model = std::filesystem::absolute(value);
}
void ParseContext(Arguments& arguments, std::string_view value)
{
    arguments.context = std::filesystem::absolute(value);
}
void ParseRuntimeCache(Arguments& arguments, std::string_view value)
{
    arguments.runtime_cache = std::filesystem::absolute(value);
}
void ParseOutput(Arguments& arguments, std::string_view value)
{
    arguments.output = std::filesystem::absolute(value);
}
void ParseSkipCompile(Arguments& arguments, std::string_view)
{
    arguments.skip_compile = true;
}
void ParseLoadOnly(Arguments& arguments, std::string_view)
{
    arguments.load_only = true;
}
void ParseCtxEmbed(Arguments& arguments, std::string_view)
{
    arguments.context_embed_mode = false;
}
uint32_t ParseIterationCount(std::string_view name, std::string_view value)
{
    size_t parsed{};
    unsigned long long count{};
    try
    {
        count = std::stoull(std::string(value), &parsed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    if (parsed != value.size() || count == 0 || count > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    return static_cast<uint32_t>(count);
}
double ParseDurationSeconds(std::string_view name, std::string_view value)
{
    size_t parsed{};
    double seconds{};
    try
    {
        seconds = std::stod(std::string(value), &parsed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(name) + " must be a positive number of seconds");
    }
    if (parsed != value.size() || !std::isfinite(seconds) || seconds <= 0.0)
        throw std::runtime_error(std::string(name) + " must be a positive number of seconds");
    return seconds;
}
void ParseWarmupIterations(Arguments& arguments, std::string_view value)
{
    if (arguments.warmup_duration_seconds)
        throw std::runtime_error("--warmup-iterations and --warmup-duration are mutually exclusive");
    arguments.warmup_iterations = ParseIterationCount("--warmup-iterations", value);
}
void ParseWarmupDuration(Arguments& arguments, std::string_view value)
{
    if (arguments.warmup_iterations)
        throw std::runtime_error("--warmup-iterations and --warmup-duration are mutually exclusive");
    arguments.warmup_duration_seconds = ParseDurationSeconds("--warmup-duration", value);
}
void ParseRunIterations(Arguments& arguments, std::string_view value)
{
    if (arguments.run_duration_seconds)
        throw std::runtime_error("--run-iterations and --run-duration are mutually exclusive");
    arguments.run_iterations = ParseIterationCount("--run-iterations", value);
}
void ParseRunDuration(Arguments& arguments, std::string_view value)
{
    if (arguments.run_iterations)
        throw std::runtime_error("--run-iterations and --run-duration are mutually exclusive");
    arguments.run_duration_seconds = ParseDurationSeconds("--run-duration", value);
}
void ParseTimeCompile(Arguments& arguments, std::string_view value)
{
    arguments.time_compile_iterations = ParseIterationCount("--time-compile", value);
}
void ParseTimeDeserialize(Arguments& arguments, std::string_view value)
{
    arguments.time_deserialize_iterations = ParseIterationCount("--time-deserialize", value);
}
void ParseAsync(Arguments& arguments, std::string_view)
{
    arguments.async = true;
}
void ParseDisableIo(Arguments& arguments, std::string_view)
{
    arguments.disable_io = true;
}
void ParseAllocator(Arguments& arguments, std::string_view value)
{
    if (value == "cpu")
        arguments.value_allocator = IOBindingAllocator::Cpu;
    else if (value == "host_accessible" || value == "host_accesible")
        arguments.value_allocator = IOBindingAllocator::HostAccessible;
    else if (value == "ep_default")
        arguments.value_allocator = IOBindingAllocator::EpDefault;
    else
        throw std::runtime_error("--allocator must be cpu, host_accessible, or ep_default");
}

void ParseShape(std::unordered_map<std::string, int64_t>& shapes, std::string_view option, std::string_view value)
{
    const auto equals = value.find('=');
    if (equals == std::string_view::npos || equals == 0)
        throw std::runtime_error(std::string(option) + " must be NAME=DIM");

    const std::string name(value.substr(0, equals));
    if (shapes.contains(name))
        throw std::runtime_error(std::string(option) + " specified more than once for " + name);

    const std::string dimension(value.substr(equals + 1));
    size_t consumed{};
    try
    {
        const int64_t parsed_dimension = std::stoll(dimension, &consumed);
        if (consumed != dimension.size() || parsed_dimension <= 0)
            throw std::runtime_error("invalid dimension");
        shapes.emplace(name, parsed_dimension);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(option) + " dimension must be a positive integer");
    }
}

void ParseShape(Arguments& arguments, std::string_view value)
{
    ParseShape(arguments.shapes, "--shape", value);
}

void ParseOverwriteShape(Arguments& arguments, std::string_view)
{
    arguments.overwrite_shapes = true;
}

void ParseEpOption(Arguments& arguments, std::string_view value)
{
    const auto equals = value.find('=');
    if (equals == std::string_view::npos || equals == 0)
        throw std::runtime_error("--ep-option must be KEY=VALUE");
    arguments.ep_options.emplace_back(value.substr(0, equals), value.substr(equals + 1));
}

constexpr std::array kOptions{
    Option{"--ep", "PATH", "Execution-provider shared library. Defaults to the provider beside the executable.", true,
           ParseEp},
    Option{"--ep-name", "NAME", "Execution-provider name used for registration.", true, ParseEpName},
    Option{"--model", "PATH", "Input ONNX model used to compile an EP context.", true, ParseModel},
    Option{"--context", "PATH", "EP-context ONNX model to create or load.", true, ParseContext},
    Option{"--runtime-cache", "PATH", "Optional directory used for the TensorRT runtime cache.", true,
           ParseRuntimeCache},
    Option{"--output", "PATH", "Write the successful benchmark JSON result to this file instead of stdout.", true,
           ParseOutput},
    Option{"--shape", "NAME=DIM", "Override a named ONNX dimension for inference. May be repeated.", true, ParseShape},
    Option{"--overwrite-shape", "",
           "Apply all --shape values as ONNX Runtime free-dimension overrides before model compilation.", false,
           ParseOverwriteShape},
    Option{"--ep-option", "KEY=VALUE", "Execution-provider option. May be repeated.", true, ParseEpOption},
    Option{"--allocator", "cpu|host_accessible|ep_default", "Allocator for input and output values. Defaults to cpu.",
           true, ParseAllocator},
    Option{"--warmup-iterations", "COUNT",
           "Warmup inference runs. The first run latency is reported separately. Defaults to 5.", true,
           ParseWarmupIterations},
    Option{"--warmup-duration", "SECONDS", "Run warmup inference for this duration instead of --warmup-iterations.",
           true, ParseWarmupDuration},
    Option{"--run-iterations", "COUNT", "Timed inference runs used for min, max, and median latency. Defaults to 50.",
           true, ParseRunIterations},
    Option{"--run-duration", "SECONDS", "Run timed inference for this duration instead of --run-iterations.", true,
           ParseRunDuration},
    Option{"--time-compile", "COUNT", "Number of model compilations to measure. Defaults to 1.", true,
           ParseTimeCompile},
    Option{"--time-deserialize", "COUNT", "Number of context loads to measure. Defaults to 1.", true,
           ParseTimeDeserialize},
    Option{"--async", "", "Disable execution-provider synchronization in each inference run.", false, ParseAsync},
    Option{"--disable-io", "", "Reuse one IoBinding for all inference runs instead of recreating it.", false,
           ParseDisableIo},
    Option{"--skip-compile", "", "Load and measure an existing context instead of compiling the input model.", false,
           ParseSkipCompile},
    Option{"--load-only", "", "Measure context loading only; do not construct inputs or run inference.", false,
           ParseLoadOnly},
    Option{"--disable-ctx-embed", "", "Do not embed the TRT engine on the context file but reference as file path.",
           false, ParseCtxEmbed},
    Option{"--help", "", "Show this help text.", false, nullptr},
};

constexpr bool AllOptionsHaveHelp()
{
    for (const auto& option : kOptions)
        if (option.help.empty())
            return false;
    return true;
}

static_assert(AllOptionsHaveHelp(), "Every command-line option must provide help text.");

std::string Usage()
{
    std::string result = "Usage: trt_ep_perf_test [options]\n\nOptions:\n";
    for (const auto& option : kOptions)
    {
        result += "  ";
        result += option.name;
        if (option.requires_value)
        {
            result += " ";
            result += option.value_name;
        }
        result += "\n      ";
        result += option.help;
        result += "\n";
    }
    return result;
}

const Option* FindOption(std::string_view name)
{
    for (const auto& option : kOptions)
        if (option.name == name)
            return &option;
    return nullptr;
}

void Validate(const Arguments& arguments)
{
    if (arguments.context.empty())
        throw std::runtime_error("--context is required");
    if (!arguments.skip_compile && arguments.model.empty())
        throw std::runtime_error("--model is required unless --skip-compile is specified");
    if ((!arguments.warmup_iterations && !arguments.warmup_duration_seconds) ||
        (arguments.warmup_iterations && arguments.warmup_duration_seconds))
        throw std::runtime_error("one of --warmup-iterations or --warmup-duration is required");
    if ((!arguments.run_iterations && !arguments.run_duration_seconds) ||
        (arguments.run_iterations && arguments.run_duration_seconds))
        throw std::runtime_error("one of --run-iterations or --run-duration is required");
}
}  // namespace

Arguments CommandLineParser::Parse(int argc, char** argv)
{
    Arguments result;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view name = argv[i];
        const Option* option = FindOption(name);
        if (option == nullptr)
            throw std::runtime_error("unknown argument " + std::string(name));
        if (option->parse == nullptr)
            throw std::runtime_error(Usage());

        std::string_view value;
        if (option->requires_value)
        {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + std::string(name));
            value = argv[++i];
        }
        option->parse(result, value);
    }
    if (!result.warmup_iterations && !result.warmup_duration_seconds)
        result.warmup_iterations = 5;
    if (!result.run_iterations && !result.run_duration_seconds)
        result.run_iterations = 50;
    Validate(result);
    return result;
}
