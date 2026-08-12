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
//
// Tests for the DLL_PROCESS_ATTACH module pin in src/dll_main.cc. The pin keeps
// the EP DLL mapped for the lifetime of the process so the loader never unmaps
// its statically-linked protobuf state (see dll_main.cc for the AVRF rationale).

#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32

#include <filesystem>
#include <memory>

#include <windows.h>

extern std::unique_ptr<Ort::Env> ort_env;
extern std::filesystem::path g_ep_lib_path;

namespace
{
// Matches the CMake target OUTPUT_NAME "onnxruntime_providers_nv_tensorrt_rtx".
constexpr const wchar_t* kEpModuleName = L"onnxruntime_providers_nv_tensorrt_rtx.dll";

// Query whether the EP DLL is currently mapped, without perturbing its load
// count (UNCHANGED_REFCOUNT), so the probe itself can never keep it alive.
bool IsEpModuleLoaded()
{
    HMODULE handle = nullptr;
    return ::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, kEpModuleName, &handle) != FALSE;
}
}  // namespace

// Releasing ORT's reference must NOT unload the DLL. register_ep() in
// test_tensorrt_rtx_main.cpp loads the EP at startup; for an unpinned DLL the
// matching UnregisterExecutionProviderLibrary() would drop the last reference
// and the loader would unmap it. The pin makes the module permanent instead.
TEST(DllMainPinTest, EpDllSurvivesUnregister)
{
    ASSERT_FALSE(g_ep_lib_path.empty()) << "EP library not registered; cannot exercise the pin.";
    ASSERT_TRUE(IsEpModuleLoaded()) << "EP DLL not loaded after startup registration.";

    ort_env->UnregisterExecutionProviderLibrary(kEpName);

    EXPECT_TRUE(IsEpModuleLoaded())
        << "EP DLL was unloaded by UnregisterExecutionProviderLibrary; DllMain pin did not take effect.";

    // Re-register so the EP stays available for subsequent tests (mirrors the
    // load/unload tests in test_tensorrt_rtx_basic.cpp).
    ort_env->RegisterExecutionProviderLibrary(kEpName, toOrtString(g_ep_lib_path).c_str());
}

// A direct LoadLibrary/FreeLibrary round-trip must also leave the module
// resident: the pin pushes the load count past the permanent threshold, so our
// matching FreeLibrary() is a no-op rather than the final unmap.
TEST(DllMainPinTest, EpDllSurvivesFreeLibrary)
{
    ASSERT_FALSE(g_ep_lib_path.empty()) << "EP library path not set.";

    HMODULE ep = ::LoadLibraryW(g_ep_lib_path.wstring().c_str());
    ASSERT_NE(ep, nullptr) << "Failed to load EP DLL (GetLastError=" << ::GetLastError() << ").";

    ASSERT_TRUE(::FreeLibrary(ep));

    EXPECT_TRUE(IsEpModuleLoaded()) << "EP DLL unloaded after FreeLibrary; DllMain pin did not take effect.";
}

#endif  // _WIN32
