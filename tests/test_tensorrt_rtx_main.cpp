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

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstring>
#pragma comment(lib, "dbghelp.lib")
#endif

// Shared ORT environment for all tests.
// A single Env instance is reused so EP registrations persist across tests.
std::unique_ptr<Ort::Env> ort_env;
std::filesystem::path g_ep_lib_path;

#ifdef _WIN32
// Vectored exception handler: dump a symbolized stack on first-chance
// access violation so gtest's SEH translator (which catches AVs but loses
// the stack) doesn't swallow the diagnostic. EXCEPTION_CONTINUE_SEARCH at
// the end keeps gtest's behaviour intact -- test still reports as failed.
static LONG WINAPI flake_av_capture(EXCEPTION_POINTERS* ep) {
    if (ep == nullptr || ep->ExceptionRecord == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    static volatile LONG s_in_handler = 0;
    if (InterlockedExchange(&s_in_handler, 1) != 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // DbgHelp init happens once in main() before the handler is installed.
    // DbgHelp APIs are not thread-safe; initializing here would race with any
    // other thread's symbol lookups. SymGetModuleInfo64 / SymFromAddr /
    // SymGetLineFromAddr64 below safely return failure for modules whose
    // symbols are not loaded, so we just consume the global state.
    HANDLE proc = GetCurrentProcess();

    const ULONG_PTR rw   = ep->ExceptionRecord->ExceptionInformation[0];
    const ULONG_PTR addr = ep->ExceptionRecord->ExceptionInformation[1];
    std::fprintf(stderr,
                 "\n=== FLAKE AV CAPTURE ===\n"
                 "ExceptionCode    : 0x%08lx\n"
                 "ExceptionAddress : %p\n"
                 "AccessKind       : %s (%llu)\n"
                 "AccessAddress    : 0x%llx\n"
                 "--- stack ---\n",
                 (unsigned long)code,
                 ep->ExceptionRecord->ExceptionAddress,
                 (rw == 0 ? "read" : (rw == 1 ? "write" : (rw == 8 ? "DEP" : "unknown"))),
                 (unsigned long long)rw,
                 (unsigned long long)addr);

    void* frames[64] = {};
    USHORT n = CaptureStackBackTrace(0, 64, frames, nullptr);

    char sym_buf[sizeof(SYMBOL_INFO) + 512] = {};
    SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 510;
    IMAGEHLP_LINE64 line = {};
    line.SizeOfStruct = sizeof(line);
    IMAGEHLP_MODULE64 mod = {};
    mod.SizeOfStruct = sizeof(mod);

    for (USHORT i = 0; i < n; ++i) {
        DWORD64 addr64 = reinterpret_cast<DWORD64>(frames[i]);
        DWORD64 displ = 0;
        DWORD line_displ = 0;
        const char* mod_name = "?";
        if (SymGetModuleInfo64(proc, addr64, &mod)) {
            mod_name = mod.ModuleName;
        }
        if (SymFromAddr(proc, addr64, &displ, sym)) {
            if (SymGetLineFromAddr64(proc, addr64, &line_displ, &line)) {
                std::fprintf(stderr, "  %2u: %s!%s+0x%llx  [%s:%lu]  [%p]\n",
                             (unsigned)i, mod_name, sym->Name,
                             (unsigned long long)displ,
                             line.FileName, (unsigned long)line.LineNumber,
                             frames[i]);
            } else {
                std::fprintf(stderr, "  %2u: %s!%s+0x%llx  [%p]\n",
                             (unsigned)i, mod_name, sym->Name,
                             (unsigned long long)displ, frames[i]);
            }
        } else {
            std::fprintf(stderr, "  %2u: %s!???  [%p]\n", (unsigned)i, mod_name, frames[i]);
        }
    }
    std::fprintf(stderr, "=== END AV CAPTURE ===\n");
    std::fflush(stderr);

    InterlockedExchange(&s_in_handler, 0);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

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
#ifdef _WIN32
    // Initialize DbgHelp once, on the main thread, before any exception can
    // fire. DbgHelp is not thread-safe, so doing this inside the vectored
    // exception handler would race with any other thread's symbol lookups.
    // Both calls below are purely diagnostic — gtest's built-in SEH translator
    // still catches AVs and marks tests FAILED on its own, so failures here
    // only degrade stack-trace quality and we log + continue rather than exit.
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    if (!SymInitialize(GetCurrentProcess(), nullptr, TRUE)) {
        std::fprintf(stderr,
                     "[setup] SymInitialize failed (GetLastError=%lu); "
                     "AV stack frames will lack symbol names.\n",
                     GetLastError());
    }
    if (AddVectoredExceptionHandler(1u, flake_av_capture) == nullptr) {
        std::fprintf(stderr,
                     "[setup] AddVectoredExceptionHandler failed (GetLastError=%lu); "
                     "diagnostic AV capture disabled.\n",
                     GetLastError());
    }
#endif
    ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "trt_rtx_ep_tests");
    register_ep(*ort_env, argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
