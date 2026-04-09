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

#ifdef _WIN32

#include "utils/security.h"
#include "utils/filesystem_utils.h"
#include "utils/path_string.h"

#include <windows.h>
#include <delayimp.h>

#include <algorithm>
#include <cstring>
#include <string>

// Auto-link required Windows libraries
#pragma comment(lib, "wintrust")
#pragma comment(lib, "crypt32")

// DLL names are defined by CMake based on TensorRT RTX version
#ifndef TRT_RTX_DLL_NAME
#define TRT_RTX_DLL_NAME "tensorrt_rtx.dll"
#endif

#ifndef TRT_ONNX_PARSER_DLL_NAME
#define TRT_ONNX_PARSER_DLL_NAME "tensorrt_onnxparser_rtx.dll"
#endif

namespace {

//! \brief Case-insensitive string comparison helper
//! \param a First string to compare
//! \param b Second string to compare
//! \return true if strings are equal ignoring case, false otherwise
bool strEqualsIgnoreCase(const char* a, const char* b)
{
    return _stricmp(a, b) == 0;
}

//! \brief Check if the DLL is one we need to verify (TensorRT RTX target)
//! \param dllName Name of the DLL to check
//! \return true if the DLL is a TensorRT RTX DLL that requires verification
bool isTrtRtxTarget(const char* dllName)
{
    if (!dllName)
    {
        return false;
    }

    // Extract just the filename if a path is provided
    const char* filename = dllName;
    const char* lastSlash = strrchr(dllName, '\\');
    if (lastSlash)
    {
        filename = lastSlash + 1;
    }
    const char* lastForwardSlash = strrchr(filename, '/');
    if (lastForwardSlash)
    {
        filename = lastForwardSlash + 1;
    }

    return strEqualsIgnoreCase(filename, TRT_RTX_DLL_NAME) ||
           strEqualsIgnoreCase(filename, TRT_ONNX_PARSER_DLL_NAME);
}

//! \brief Log security warning and abort the delay-load by raising an exception
//! \param dllName Name of the DLL that failed verification
//! \param reason Reason for the security failure
//! \note This function raises a structured exception to abort the delay-load process,
//!       preventing fallback to the default Windows DLL search order.
//!       The caller can catch this exception via SEH if needed.
[[noreturn]] void securityFailure(const wchar_t* dllName, const wchar_t* reason)
{
    // Raise a delay-load exception to abort the load and prevent fallback
    // to the default Windows DLL search order. This is safer than just
    // returning nullptr, which would allow the delay-load helper to try
    // loading from untrusted locations.
    //
    // Exception code format for delay-load: (0xE0000000 | ERROR_MOD_NOT_FOUND)
    // This matches what the delay-load helper expects for module-not-found errors.
    constexpr DWORD kDelayLoadExceptionCode = 0xE0000000 | ERROR_MOD_NOT_FOUND;
    RaiseException(kDelayLoadExceptionCode, 0, 0, nullptr);

    // Never reached, but satisfies [[noreturn]]
    std::abort();
}

//! \brief Extract the filename portion from a DLL path
//! \param dllName Name or path of the DLL (narrow string)
//! \return Filename portion only as a wide string
std::wstring extractDllFilename(const char* dllName)
{
    std::wstring dllNameW = ToWideString(dllName);
    size_t lastSlash = dllNameW.find_last_of(L"\\/");
    if (lastSlash != std::wstring::npos)
    {
        return dllNameW.substr(lastSlash + 1);
    }
    return dllNameW;
}

//! \brief Verify DLL is present in EP directory and (in production) has valid NVIDIA signature
//! \param dllName Name of the DLL to verify
//! \return true if verification passes, false otherwise (exception raised on failure)
//! \note Directory enforcement is always active. Signature verification is only in production builds.
//! \note On failure, this function calls securityFailure which raises an exception.
bool verifyDllSignature(const char* dllName)
{
    // Convert DLL name to wide string once
    std::wstring dllNameW = extractDllFilename(dllName);

    // Get the EP module directory - this must succeed for security
    // Use address of this function as anchor to locate the EP DLL
    std::wstring baseDir = trt_rtx_ep::utils::FileSystemUtils::GetCurrentModuleDirectory(&verifyDllSignature);
    if (baseDir.empty())
    {
        securityFailure(dllNameW.c_str(), L"could not determine the expected library directory.");
        return false;  // Never reached due to [[noreturn]] on securityFailure
    }

    // Construct full path: DLL must be in the EP directory
    std::wstring fullPath = baseDir + L"\\" + dllNameW;

    // Verify the file exists and is not a directory
    DWORD attrib = GetFileAttributesW(fullPath.c_str());
    if (attrib == INVALID_FILE_ATTRIBUTES || (attrib & FILE_ATTRIBUTE_DIRECTORY))
    {
        securityFailure(dllNameW.c_str(), L"library not found in the execution provider directory.");
        return false;  // Never reached due to [[noreturn]] on securityFailure
    }

    // Directory enforcement complete. Now handle signature verification based on build type.
#ifdef TRT_RTX_EP_PRODUCTION_BUILD
    // Production builds: verify NVIDIA signature on the DLL
    if (!VerifyNvidiaSignature(fullPath))
    {
        securityFailure(dllNameW.c_str(), L"digital signature verification failed.");
        return false;  // Never reached due to [[noreturn]] on securityFailure
    }
#endif
    // Non-production builds: signature verification skipped, but directory enforcement was applied

    return true;
}

}  // anonymous namespace

// =============================================================================
// Delay Load Hook Implementation
// =============================================================================

//! \brief Delay load notification hook
//! \param dliNotify Notification type indicating the stage of delay loading
//! \param pdli Pointer to delay load information structure
//! \return Function pointer or module handle, or nullptr to use default processing
//! \note This is called by the delay load helper at various points during DLL loading
//! \note For TensorRT RTX DLLs, this hook enforces loading only from the EP directory.
//!       Non-target DLLs are handled by default delay-load processing.
static FARPROC WINAPI delayLoadNotifyHook(unsigned dliNotify, PDelayLoadInfo pdli)
{
    switch (dliNotify)
    {
        case dliNotePreLoadLibrary:
        {
            // Only intervene for TensorRT RTX DLLs
            if (!isTrtRtxTarget(pdli->szDll))
            {
                return nullptr;  // Let default delay-load handle non-target DLLs
            }

            // Verify DLL location (and signature in production builds)
            // This call raises an exception on failure via securityFailure
            if (!verifyDllSignature(pdli->szDll))
            {
                // verifyDllSignature already raised an exception
                // This line should never be reached
                return nullptr;
            }

            // Load exclusively from the EP directory using absolute path
            std::wstring dllNameW = extractDllFilename(pdli->szDll);
            std::wstring baseDir = trt_rtx_ep::utils::FileSystemUtils::GetCurrentModuleDirectory(&delayLoadNotifyHook);
            if (baseDir.empty())
            {
                securityFailure(dllNameW.c_str(), L"could not determine the expected library directory.");
                return nullptr;  // Never reached
            }

            std::wstring fullPath = baseDir + L"\\" + dllNameW;

            // Load from the EP directory only using absolute path
            // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: Search the folder containing the DLL (from fullPath)
            // LOAD_LIBRARY_SEARCH_SYSTEM32: Allow system DLL dependencies
            // This combination prevents loading from arbitrary search paths
            HMODULE hModule = LoadLibraryExW(fullPath.c_str(), nullptr,
                                              LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                              LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (!hModule)
            {
                securityFailure(dllNameW.c_str(), L"failed to load the library from the execution provider directory.");
                return nullptr;  // Never reached
            }

            return reinterpret_cast<FARPROC>(hModule);
        }

        case dliNotePreGetProcAddress:
            // Called before GetProcAddress is called
            break;

        case dliNoteEndProcessing:
            // Called after all processing is done
            break;

        default:
            break;
    }

    return nullptr;  // Return nullptr to let the default processing continue
}

//! \brief Delay load failure hook
//! \param dliNotify Notification type indicating what failed
//! \param pdli Pointer to delay load information structure
//! \return Always returns nullptr
//! \note This is called when delay load fails (either LoadLibrary or GetProcAddress)
static FARPROC WINAPI delayLoadFailureHook(unsigned dliNotify, PDelayLoadInfo pdli)
{
    if (dliNotify == dliFailLoadLib)
    {
        // LoadLibrary failed
        if (isTrtRtxTarget(pdli->szDll))
        {
            std::wstring dllNameW = ToWideString(pdli->szDll);
            securityFailure(dllNameW.c_str(),
                L"could not be loaded. Please ensure TensorRT RTX is properly installed.");
        }
    }
    else if (dliNotify == dliFailGetProc)
    {
        // GetProcAddress failed
        if (isTrtRtxTarget(pdli->szDll))
        {
            std::wstring dllNameW = ToWideString(pdli->szDll);
            std::wstring msg = L"is incompatible (missing required function: ";
            msg += ToWideString(pdli->dlp.szProcName);
            msg += L").";
            securityFailure(dllNameW.c_str(), msg.c_str());
        }
    }

    return nullptr;
}

// =============================================================================
// Hook Registration
// =============================================================================

// These are the official delay load hook function pointers
// The delay load helper looks for these symbols
extern "C" {
    // Notify hook - called at various stages of delay loading
    const PfnDliHook __pfnDliNotifyHook2 = delayLoadNotifyHook;

    // Failure hook - called when delay loading fails
    const PfnDliHook __pfnDliFailureHook2 = delayLoadFailureHook;
}

#endif  // _WIN32
