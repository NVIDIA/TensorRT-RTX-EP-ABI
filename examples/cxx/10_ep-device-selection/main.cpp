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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <regex>
#include <vector>

#include "argparsing.h"
#include "lodepng.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

#if ORT_API_VERSION < 23
#    error "Onnx runtime header too old. Version >=1.23.0 assumed"
#endif

static Ort::SessionOptions create_session_options(Ort::Env& env, Opts& opts)
{
    std::vector<Ort::ConstEpDevice> selected_devices;
    auto ep_devices = env.GetEpDevices();
    bool filter_by_vendor = !opts.select_vendor.empty();
    bool filter_by_ep = !opts.select_ep.empty();
    if (!filter_by_ep && !filter_by_vendor)
    {
        LOG("No filter was provided. The default filter by ep will be used.");
        filter_by_ep = true;
        opts.select_ep = kNvTensorRTRTXExecutionProvider;
    }
    LOG("{} devices found", ep_devices.size());
    for (auto& device : ep_devices)
    {
        auto metadata = device.Device().Metadata();
        auto luid = metadata.GetValue("LUID");
        LOG("Vendor: {}, EpName: {}, DeviceId: 0x{:x}, LUID: {}", device.EpVendor(), device.EpName(),
            device.Device().DeviceId(), luid ? luid : "<unavailable>");
        const bool vendor_match =
            !filter_by_vendor || to_lowercase(opts.select_vendor) == to_lowercase(device.Device().Vendor());
        const bool ep_match = !filter_by_ep || to_lowercase(opts.select_ep) == to_lowercase(device.EpName());
        if (vendor_match && ep_match)
        {
            selected_devices.push_back(device);
        }
    }

    Ort::SessionOptions so;
    if (selected_devices.empty())
    {
        THROW_ERROR("No execution-provider devices matched the requested filters.");
    }
    Ort::KeyValuePairs ep_options;
    so.AppendExecutionProvider_V2(env, selected_devices, ep_options);
    so.SetEpSelectionPolicy(opts.ep_device_policy);
    return so;
}

static Ort::Session create_session(Ort::Env& env, std::filesystem::path& model_file,
                                   const Ort::SessionOptions& session_options)
{
    if (!std::filesystem::is_regular_file(model_file))
    {
        THROW_ERROR("Model file \"{}\" does not exist!", model_file.string());
    }

    Ort::Session session(env, toOrtFileString(model_file).c_str(), session_options);
    return session;
}

static unsigned char clampToU8(float val)
{
    return static_cast<unsigned char>(std::clamp(val, 0.0f, 255.0f));
}

auto main(int argc, char** argv) -> int
{
    try
    {
        Opts opts = parse_args(argc, argv);

        auto version_string = Ort::GetVersionString();
        auto build_info = Ort::GetBuildInfoString();

        LOG("Hello from ONNX runtime version: {} (build info {})\n", version_string, build_info);

        auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
        // In next examples this code snipped will be used as function: register_execution_providers(env);
        const auto providers_library = get_executable_parent_path() / DLL_NAME("onnxruntime_providers_nv_tensorrt_rtx");
        if (!std::filesystem::is_regular_file(providers_library))
        {
            LOG("{} does not exist! Skipping execution provider", providers_library.string());
            return EXIT_FAILURE;
        }

        try
        {
            env.RegisterExecutionProviderLibrary(kNvTensorRTRTXExecutionProvider, toOrtFileString(providers_library));
        }
        catch (std::exception& e)
        {
            LOG("Failed to register {}! Skipping execution provider with error {}", providers_library.string(),
                e.what());
            return EXIT_FAILURE;
        }
        auto session_options = create_session_options(env, opts);

        std::string model_file = MODEL_FILE;
        auto model_path = get_executable_path().parent_path() / MODEL_FILE;
        auto model_context_file = std::regex_replace(model_file, std::regex(".onnx$"), "_ctx.onnx");
        auto model_context_path = get_executable_path().parent_path() / model_context_file;
        bool use_model_context = std::filesystem::is_regular_file(model_context_path);
        auto load_path = use_model_context ? model_context_path : model_path;

        // Load input image as RGBA
        uint8_t* image{};
        DEFER(image, free(image));
        uint32_t width{};
        uint32_t height{};
        auto error = lodepng_decode32_file(&image, &width, &height, opts.input_image.c_str());
        if (error)
        {
            LOG("Failed to load image \"{}\"", opts.input_image);
            return EXIT_FAILURE;
        }
        LOG("Loaded image \"{}\" with size {}x{}", opts.input_image, width, height);

        // candy.onnx expects [batch_size, 3, H, W] float32 NCHW RGB input
        session_options.AddFreeDimensionOverrideByName("batch_size", 1);
        if (!use_model_context)
        {
            session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, opts.enableEpContext ? "1" : "0");
        }

        auto infer_session = create_session(env, load_path, session_options);

        Ort::AllocatorWithDefaultOptions cpu_allocator;
        Ort::AllocatedStringPtr input_name = infer_session.GetInputNameAllocated(0, cpu_allocator);
        Ort::AllocatedStringPtr output_name = infer_session.GetOutputNameAllocated(0, cpu_allocator);

        // Convert RGBA uint8 image to float32 NCHW RGB
        std::vector<float> input_data(3 * width * height);
        for (uint32_t y = 0; y < height; y++)
        {
            for (uint32_t x = 0; x < width; x++)
            {
                size_t rgba_idx = (y * width + x) * 4;
                input_data[0 * width * height + y * width + x] = static_cast<float>(image[rgba_idx + 0]);  // R
                input_data[1 * width * height + y * width + x] = static_cast<float>(image[rgba_idx + 1]);  // G
                input_data[2 * width * height + y * width + x] = static_cast<float>(image[rgba_idx + 2]);  // B
            }
        }

        std::array<int64_t, 4> input_shape{1, 3, static_cast<int64_t>(height), static_cast<int64_t>(width)};
        Ort::Value input_value = Ort::Value::CreateTensor<float>(
            cpu_allocator.GetInfo(), input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        std::vector<float> output_data(3 * width * height);
        Ort::Value output_value = Ort::Value::CreateTensor<float>(
            cpu_allocator.GetInfo(), output_data.data(), output_data.size(), input_shape.data(), input_shape.size());

        Ort::IoBinding inference_binding(infer_session);
        inference_binding.BindInput(input_name.get(), input_value);
        inference_binding.BindOutput(output_name.get(), output_value);

        Ort::RunOptions run_options;
        run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
        infer_session.Run(run_options, inference_binding);
        inference_binding.SynchronizeOutputs();

        // Convert float32 NCHW RGB output back to RGBA uint8 for PNG
        std::vector<uint8_t> out_image(width * height * 4);
        for (uint32_t y = 0; y < height; y++)
        {
            for (uint32_t x = 0; x < width; x++)
            {
                size_t rgba_idx = (y * width + x) * 4;
                out_image[rgba_idx + 0] = clampToU8(output_data[0 * width * height + y * width + x]);
                out_image[rgba_idx + 1] = clampToU8(output_data[1 * width * height + y * width + x]);
                out_image[rgba_idx + 2] = clampToU8(output_data[2 * width * height + y * width + x]);
                out_image[rgba_idx + 3] = 255;
            }
        }

        auto encode_error = lodepng_encode32_file(opts.output_image.c_str(), out_image.data(), width, height);
        if (encode_error != 0)
        {
            LOG("Failed to save output image \"{}\": {}", opts.output_image, encode_error);
            return EXIT_FAILURE;
        }
        LOG("Saved stylized output to \"{}\"", opts.output_image);
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
