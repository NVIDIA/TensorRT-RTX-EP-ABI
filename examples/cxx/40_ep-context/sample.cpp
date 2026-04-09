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
#include <iostream>
#include <string>

#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

#include "utils.h"

template <typename Func>
static double MeasureTime(Func&& func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char* argv[]) {
  auto exe_dir = get_executable_path().parent_path();
  std::filesystem::path input_model_path = (exe_dir / "candy.onnx");
  std::filesystem::path output_model_path = (exe_dir / "candy_compiled.onnx");
  int embed_mode = 0;

  if (argc >= 3) {
    input_model_path = argv[1];
    output_model_path = argv[2];
  } else if (argc == 2) {
    std::cerr << "Usage: " << argv[0]
              << " [input_model.onnx] [compiled_model_output.onnx] [embed_mode]" << std::endl;
    std::cerr << "  Defaults to candy.onnx / candy_compiled.onnx next to the executable." << std::endl;
    std::cerr << "  embed_mode: 0 = external (default), 1 = embedded" << std::endl;
    return 1;
  }

  if (argc >= 4) {
    try {
      embed_mode = std::stoi(argv[3]);
    } catch (const std::invalid_argument&) {
      std::cerr << "ERROR: Invalid embed_mode value. Must be an integer." << std::endl;
      return 1;
    }
    if (embed_mode != 0 && embed_mode != 1) {
      std::cerr << "ERROR: embed_mode must be 0 or 1." << std::endl;
      return 1;
    }
  }

  const std::filesystem::path runtime_cache_dir = "ort_runtime_cache";
  try {
    if (std::filesystem::exists(runtime_cache_dir)) {
      std::filesystem::remove_all(runtime_cache_dir);
    }
  } catch (const std::filesystem::filesystem_error& ex) {
    std::cerr << "WARNING: Failed to delete runtime cache directory: " << ex.what() << std::endl;
    std::cerr << "Performance metrics may not be accurate due to existing cache." << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "ONNX Runtime EP Context - TensorRT RTX" << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "> Input Model Path:   " << input_model_path << std::endl;
  std::cout << "> Output Model Path:  " << output_model_path << std::endl;
  std::cout << "> Embed Mode:         " << (embed_mode == 1 ? "Embedded" : "External") << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EPContextSample");

    register_execution_providers(env);

    auto* trt_device = find_trt_rtx_device(env);
    if (!trt_device) {
      std::cerr << "ERROR: TensorRT RTX EP device not found. "
                << "Ensure onnxruntime_providers_nv_tensorrt_rtx is next to the executable." << std::endl;
      return 1;
    }

    // Normal ONNX load
    {
      Ort::SessionOptions session_options;
      append_ep_v2(session_options, env, trt_device);

      std::cout << "> Loading original ONNX model from disk..." << std::endl;
      double load_time_normal = MeasureTime([&]() {
        Ort::Session session(env, input_model_path.c_str(), session_options);
      });
      std::cout << "> Original session load time: " << load_time_normal << " sec" << std::endl;
    }

    // AOT compilation
    {
      Ort::SessionOptions session_options;
      append_ep_v2(session_options, env, trt_device,
                   {{"nv_runtime_cache_path", runtime_cache_dir.string()}});

      std::cout << "> Compiling model with TensorRT RTX..." << std::endl;

      Ort::ModelCompilationOptions compile_options(env, session_options);
      compile_options.SetEpContextEmbedMode(embed_mode);
      compile_options.SetInputModelPath(toOrtFileString(input_model_path).c_str());
      compile_options.SetOutputModelPath(toOrtFileString(output_model_path).c_str());

      double compile_time = MeasureTime([&]() {
        Ort::Status status = Ort::CompileModel(env, compile_options);
        if (!status.IsOK()) {
          throw Ort::Exception(status.GetErrorMessage(), ORT_FAIL);
        }
      });
      std::cout << "> Model compiled successfully!" << std::endl;
      std::cout << "> Compile time: " << compile_time << " sec" << std::endl;
      std::cout << "> Compiled model saved at " << output_model_path << std::endl;
    }

    // Load compiled model
    {
      Ort::SessionOptions session_options;
      append_ep_v2(session_options, env, trt_device);

      std::cout << "> Loading compiled model from disk..." << std::endl;
      double load_time_compiled = MeasureTime([&]() {
        Ort::Session session(env, output_model_path.c_str(), session_options);
      });
      std::cout << "> Context model session load time: " << load_time_compiled << " sec" << std::endl;
    }

    // Load with runtime cache
    {
      Ort::SessionOptions session_options;
      append_ep_v2(session_options, env, trt_device,
                   {{"nv_runtime_cache_path", runtime_cache_dir.string()}});

      double jit_time = MeasureTime([&]() {
        Ort::Session session(env, output_model_path.c_str(), session_options);
      });
      std::cout << "> Context model session load time with runtime cache: " << jit_time << " sec" << std::endl;
      std::cout << "> Runtime cache has been populated at: " << runtime_cache_dir << std::endl;
    }
  } catch (const Ort::Exception& ex) {
    std::cerr << "\nONNX Runtime Exception: " << ex.what() << std::endl;
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "\nStandard Exception: " << ex.what() << std::endl;
    return 1;
  }

  std::cout << "\nProgram finished successfully." << std::endl;
  return 0;
}
