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

#pragma once

// Include Windows headers first to avoid conflicts
#ifdef _WIN32
  #ifndef _WINDOWS_
    #include <windows.h>
  #endif
#endif

#include <string>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <cctype>
#include "onnxruntime_cxx_api.h"
#include "path_string.h"


namespace trt_rtx_ep {
namespace utils {

/// <summary>
/// Cross-platform filesystem utilities
/// </summary>
class FileSystemUtils {
 public:
  // Path-typed APIs. The const std::string& overloads are intentionally = deleted to force
  // callers to convert UTF-8 std::string options at the boundary via ToPathString (path_string.h)
  // instead of relying on std::filesystem::path's implicit ANSI-decoding ctor on Windows.

  static bool CreateDirectoryRecursive(const std::filesystem::path& path) {
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
      return std::filesystem::is_directory(path, ec);
    }
    std::filesystem::create_directories(path, ec);
    return !ec;
  }
  static bool CreateDirectoryRecursive(const std::string&) = delete;

  static bool CreateDirectoryRecursive(const std::filesystem::path& path, std::string& error_msg) {
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
      if (ec) {
        error_msg = "Failed to check directory existence: " + ec.message();
        return false;
      }
      if (!std::filesystem::is_directory(path, ec)) {
        error_msg = "Path exists but is not a directory: " + PathToUTF8String(path.native());
        return false;
      }
      return true;
    }
    std::filesystem::create_directories(path, ec);
    if (ec) {
      error_msg = "Failed to create directory '" + PathToUTF8String(path.native()) + "': " + ec.message();
      return false;
    }
    return true;
  }
  static bool CreateDirectoryRecursive(const std::string&, std::string&) = delete;

  static bool PathExists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
  }
  static bool PathExists(const std::string&) = delete;

  static bool IsDirectory(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
  }
  static bool IsDirectory(const std::string&) = delete;

  static bool IsFile(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
  }
  static bool IsFile(const std::string&) = delete;

  static std::filesystem::path GetAbsolutePath(const std::filesystem::path& path) {
    std::error_code ec;
    auto abs_path = std::filesystem::absolute(path, ec);
    if (ec) {
      return path;
    }
    return abs_path;
  }
  static std::filesystem::path GetAbsolutePath(const std::string&) = delete;

  static std::filesystem::path GetParentPath(const std::filesystem::path& path) {
    return path.parent_path();
  }
  static std::filesystem::path GetParentPath(const std::string&) = delete;

  static std::filesystem::path GetFilename(const std::filesystem::path& path) {
    return path.filename();
  }
  static std::filesystem::path GetFilename(const std::string&) = delete;

  static std::filesystem::path JoinPath(const std::filesystem::path& path1,
                                        const std::filesystem::path& path2) {
    return path1 / path2;
  }
  static std::filesystem::path JoinPath(const std::string&, const std::string&) = delete;

  static std::filesystem::path NormalizePath(const std::filesystem::path& path) {
    return path.lexically_normal();
  }
  static std::filesystem::path NormalizePath(const std::string&) = delete;

  /// <summary>
  /// Sanitizes a file path to help eliminate possible malicious code or path traversal attacks.
  /// Trims whitespace, checks for null bytes, parent directory traversal, and shell meta-characters.
  /// </summary>
  /// <param name="path">Input file path to sanitize</param>
  /// <param name="error_msg">Output error message if sanitization fails</param>
  /// <returns>Sanitized path string, or empty string if validation fails</returns>
  static std::string SanitizeFilePath(const std::string& path, std::string& error_msg)
  {
    std::string sanitized_value = path;

    // Remove whitespace from beginning and end
    size_t first_non_ws = sanitized_value.find_first_not_of(" \t\n\r\f\v");
    if (first_non_ws == std::string::npos)
    {
      // String is empty or whitespace-only
      error_msg = "Invalid file path: empty or whitespace-only.";
      return "";
    }
    sanitized_value.erase(0, first_non_ws);
    sanitized_value.erase(sanitized_value.find_last_not_of(" \t\n\r\f\v") + 1);

    // Check for null bytes or control characters
    for (unsigned char ch : sanitized_value)
    {
      if (ch == '\0' || std::iscntrl(ch))
      {
        error_msg = "Invalid file path: contains control character.";
        return "";
      }
    }

    // Disallow relative parent directories or suspicious patterns
    if (sanitized_value.find("..") != std::string::npos)
    {
      error_msg = "Invalid file path: parent directory traversal is not allowed.";
      return "";
    }

    // Disallow possible shell meta-characters known for command injection
    const std::string forbidden_chars = "|&;`$><!#";
    if (sanitized_value.find_first_of(forbidden_chars) != std::string::npos)
    {
      error_msg = "Invalid file path: forbidden characters detected.";
      return "";
    }

    error_msg.clear();
    return sanitized_value;
  }

  /// <summary>
  /// Sanitizes a file path and throws an ORT error if validation fails.
  /// </summary>
  /// <param name="path">Input file path to sanitize</param>
  /// <param name="ort_api">ORT API for error handling</param>
  /// <returns>Sanitized path string</returns>
  static std::string SanitizeFilePath(const std::string& path, const OrtApi& ort_api)
  {
    std::string error_msg;
    std::string sanitized = SanitizeFilePath(path, error_msg);
    if (!error_msg.empty())
    {
      Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str()));
    }
    return sanitized;
  }

#ifdef _WIN32
  /// <summary>
  /// Gets the directory of the current executing module (DLL/EXE)
  /// </summary>
  /// <param name="moduleFunc">Address of a function in the module to locate (default: this function)</param>
  /// <returns>Directory path as wide string, or empty string on failure</returns>
  static std::wstring GetCurrentModuleDirectory(const void* moduleFunc = nullptr)
  {
    HMODULE hModule = NULL;

    // Use the address of this function if no specific address provided
    const void* addrToUse = moduleFunc ? moduleFunc : reinterpret_cast<const void*>(&GetCurrentModuleDirectory);

    // Get handle to the DLL/EXE containing the specified address
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(addrToUse),
                            &hModule))
    {
      return L"";
    }

    wchar_t buffer[MAX_PATH];
    DWORD len = GetModuleFileNameW(hModule, buffer, MAX_PATH);
    if (len == 0 || len >= MAX_PATH)
    {
      return L"";
    }

    std::wstring path(buffer);
    size_t lastSlash = path.find_last_of(L"\\/");
    if (lastSlash != std::wstring::npos)
    {
      return path.substr(0, lastSlash);
    }
    return path;
  }
#endif  // _WIN32

  
};

inline std::vector<char> ReadFile(const std::filesystem::path& path, const OrtLogger& logger, const OrtApi& ort_api) {
  if (!std::filesystem::exists(path)) {
    // Cache miss: return empty so the caller proceeds. No log here — this
    // path can be reached from EP destructor context after the logger has
    // been torn down.
    (void)logger;
    (void)ort_api;
    return {};
  }
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    std::string message = "Failed to open file: " + PathToUTF8String(path.native());
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Check if file size determination failed
  if (size < 0) {
    std::string message = "Failed to determine file size: " + PathToUTF8String(path.native());
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }

  // Explicitly cast to size_t after validation
  size_t usize = static_cast<size_t>(size);

  // Allocate buffer with explicit error handling for allocation failures
  std::vector<char> buffer;
  try {
    buffer.resize(usize);
  } catch (const std::bad_alloc&) {
    std::string message = "Failed to allocate memory for file read (" +
                          std::to_string(usize) + " bytes): " + PathToUTF8String(path.native());
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }

  if (usize > 0 && !file.read(buffer.data(), static_cast<std::streamsize>(usize))) {
    std::string message = "Failed to read file: " + PathToUTF8String(path.native());
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }
  return buffer;
}
inline std::vector<char> ReadFile(const std::string&, const OrtLogger&, const OrtApi&) = delete;

inline void WriteFile(const std::filesystem::path& path, const void* data, size_t size, const OrtLogger& logger, const OrtApi& ort_api) {
  if (std::filesystem::exists(path)) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
      std::string message = "Failed to open file for writing: " + PathToUTF8String(path.native());
      Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
    }
    file.write(static_cast<const char*>(data), size);
  } else {
    // First-time cache write. No log here — this path is reached from the
    // EP destructor (IExecutionContextDeleter -> WriteFile), where the
    // logger may already have been torn down.
    (void)logger;
    (void)ort_api;
    // Create new file
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file) {
      std::string error_message = "Failed to create file: " + PathToUTF8String(path.native());
      Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, error_message.c_str()));
    }
    file.write(static_cast<const char*>(data), size);
  }
}
inline void WriteFile(const std::string&, const void*, size_t, const OrtLogger&, const OrtApi&) = delete;

inline void WriteFile(const std::filesystem::path& path, const std::vector<char>& data, const OrtLogger& logger, const OrtApi& ort_api) { WriteFile(path, data.data(), data.size(), logger, ort_api); }
inline void WriteFile(const std::string&, const std::vector<char>&, const OrtLogger&, const OrtApi&) = delete;

}  // namespace utils
}  // namespace trt_rtx_ep

