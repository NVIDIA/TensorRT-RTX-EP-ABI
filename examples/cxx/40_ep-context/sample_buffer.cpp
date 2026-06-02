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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

namespace fs = std::filesystem;

static std::vector<uint8_t> LoadFileToBuffer(const fs::path& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open file: " + filename.string());

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        throw std::runtime_error("Failed to read file: " + filename.string());
    }
    return buffer;
}

template <typename Func>
static double MeasureTime(Func&& func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input_model.onnx> <weights_file.onnx.data>"
                  << " <external_data_filename> [embed_mode]" << std::endl;
        std::cerr << "  external_data_filename: The name used for external data in the model (e.g., 'model.onnx.data')"
                  << std::endl;
        std::cerr << "  embed_mode: 0 = external (default), 1 = embedded" << std::endl;
        return 1;
    }

    fs::path input_model_path = argv[1];
    fs::path weights_model_path = argv[2];
    fs::path external_data_filename_path = argv[4];
    int embed_mode = 0;

    if (argc >= 5)
    {
        try
        {
            embed_mode = std::stoi(argv[5]);
        }
        catch (const std::invalid_argument&)
        {
            std::cerr << "ERROR: Invalid embed_mode value. Must be an integer." << std::endl;
            return 1;
        }
        catch (const std::out_of_range&)
        {
            std::cerr << "ERROR: embed_mode value out of range." << std::endl;
            return 1;
        }
        if (embed_mode != 0 && embed_mode != 1)
        {
            std::cerr << "ERROR: embed_mode must be 0 or 1." << std::endl;
            return 1;
        }
    }

    std::cout << "> Embed mode set to: " << embed_mode << std::endl;
    std::cout << "> External data filename: " << external_data_filename_path << std::endl;

    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EPContextBufferSample");

        register_execution_providers(env);

        auto trt_device = find_trt_rtx_device(env);
        if (!trt_device)
        {
            std::cerr << "ERROR: TensorRT RTX EP device not found. "
                      << "Ensure onnxruntime_providers_nv_tensorrt_rtx is next to the executable." << std::endl;
            return 1;
        }

        Ort::SessionOptions session_options;
        Ort::KeyValuePairs ep_options;
        ep_options.Add("nv_use_external_data_initializer", "1");
        std::vector<Ort::ConstEpDevice> devices = {trt_device};
        session_options.AppendExecutionProvider_V2(env, devices, ep_options);

        // Load model and weights into buffers
        std::vector<uint8_t> model_buffer = LoadFileToBuffer(input_model_path);
        std::vector<uint8_t> weights_buffer = LoadFileToBuffer(weights_model_path);

        // Register external weights (needed in buffer mode)
        std::vector<std::basic_string<ORTCHAR_T>> file_names = {external_data_filename_path.native()};
        std::vector<char*> file_buffers_data = {static_cast<char*>(static_cast<void*>(weights_buffer.data()))};
        std::vector<size_t> file_buffers_size = {weights_buffer.size()};

        session_options.AddExternalInitializersFromFilesInMemory(file_names, file_buffers_data, file_buffers_size);

        // Regular ONNX load (buffer mode)
        std::cout << "> Loading regular ONNX (buffer)..." << std::endl;
        double load_time_normal = MeasureTime(
            [&]()
            {
                Ort::Session session(env, model_buffer.data(), model_buffer.size(), session_options);
            });
        std::cout << "> Session load time: " << load_time_normal << " sec" << std::endl;

        // Compile model from buffer
        std::cout << "> Compiling model (buffer)..." << std::endl;
        void* output_buffer_data = nullptr;
        size_t output_buffer_size = 0;
        Ort::AllocatorWithDefaultOptions output_allocator;

        Ort::ModelCompilationOptions compile_options(env, session_options);
        compile_options.SetEpContextEmbedMode(embed_mode);
        compile_options.SetInputModelFromBuffer(model_buffer.data(), model_buffer.size());
        compile_options.SetOutputModelBuffer(output_allocator, &output_buffer_data, &output_buffer_size);

        double compile_time = MeasureTime(
            [&]()
            {
                Ort::Status status = Ort::CompileModel(env, compile_options);
                if (!status.IsOK())
                {
                    throw Ort::Exception(status.GetErrorMessage(), ORT_FAIL);
                }
            });
        std::cout << "> Compiled successfully!" << std::endl;
        std::cout << "> Compile time: " << compile_time << " sec" << std::endl;
        std::cout << "> Compiled model buffer size: " << output_buffer_size << " bytes" << std::endl;

        // Load compiled model from buffer
        std::cout << "> Loading EP context model (buffer)..." << std::endl;
        double load_time_compiled = MeasureTime(
            [&]()
            {
                Ort::Session compiled_session(env, reinterpret_cast<uint8_t*>(output_buffer_data), output_buffer_size,
                                              session_options);
            });
        std::cout << "> Session load time: " << load_time_compiled << " sec" << std::endl;

        if (output_buffer_data)
        {
            output_allocator.Free(output_buffer_data);
        }
    }
    catch (const Ort::Exception& ex)
    {
        std::cerr << "ONNX Runtime error: " << ex.what() << std::endl;
        return 1;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Standard exception: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
