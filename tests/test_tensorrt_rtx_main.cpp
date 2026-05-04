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

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include "test_tensorrt_rtx_utils.h"

#include <filesystem>
#include <iostream>
#include <memory>

// Shared ORT environment for all tests.
// A single Env instance is reused so EP registrations persist across tests.
std::unique_ptr<Ort::Env> ort_env;
std::filesystem::path g_ep_lib_path;

// Resolve the EP library path.
//
// On Windows the EP DLL delay-loads TRT RTX DLLs at session-creation time.
// Windows resolves delay-loaded dependencies relative to the loading DLL, so
// we must load the EP from the test executable's directory (where all sibling
// DLLs were copied) rather than from the original build output path.
// EP_LIB_PATH (the build-output absolute path) is used only as a fallback.
static std::filesystem::path resolve_ep_lib(const char* argv0) {
    const std::filesystem::path build_path = EP_LIB_PATH;
    const auto local_path =
        std::filesystem::absolute(argv0).parent_path() / build_path.filename();
    if (std::filesystem::is_regular_file(local_path)) {
        return local_path;
    }
    return build_path;
}

static void register_ep(Ort::Env& env, const char* argv0) {
    const auto ep_lib = resolve_ep_lib(argv0);
    if (!std::filesystem::is_regular_file(ep_lib)) {
        std::cerr << "[setup] EP library not found at " << ep_lib
                  << " — tests requiring the EP will be skipped.\n";
        return;
    }

#ifdef _WIN32
    auto ep_lib_str = ep_lib.wstring();
#else
    auto ep_lib_str = ep_lib.string();
#endif

    try {
        env.RegisterExecutionProviderLibrary(kEpName, ep_lib_str.c_str());
        std::cout << "[setup] Registered TRT RTX EP from " << ep_lib << "\n";
        g_ep_lib_path = ep_lib;
    } catch (const Ort::Exception& ex) {
        std::cerr << "[setup] Failed to register TRT RTX EP: " << ex.what()
                  << " — tests requiring the EP will be skipped.\n";
    }
}

int main(int argc, char** argv) {
    ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "trt_rtx_ep_tests");
    register_ep(*ort_env, argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
