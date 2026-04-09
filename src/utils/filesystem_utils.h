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


namespace trt_rtx_ep {
namespace utils {

/// <summary>
/// Cross-platform filesystem utilities
/// </summary>
class FileSystemUtils {
 public:
  /// <summary>
  /// Creates a directory and all necessary parent directories
  /// </summary>
  /// <param name="path">Directory path to create</param>
  /// <returns>True if successful or directory already exists, false otherwise</returns>
  static bool CreateDirectoryRecursive(const std::string& path) {
    std::error_code ec;
    std::filesystem::path dir_path(path);
    
    // Check if directory already exists
    if (std::filesystem::exists(dir_path, ec)) {
      return std::filesystem::is_directory(dir_path, ec);
    }
    
    // Create directory with all parent directories
    std::filesystem::create_directories(dir_path, ec);
    return !ec;
  }

  /// <summary>
  /// Creates a directory and all necessary parent directories with error message
  /// </summary>
  /// <param name="path">Directory path to create</param>
  /// <param name="error_msg">Output error message if creation fails</param>
  /// <returns>True if successful or directory already exists, false otherwise</returns>
  static bool CreateDirectoryRecursive(const std::string& path, std::string& error_msg) {
    std::error_code ec;
    std::filesystem::path dir_path(path);
    
    // Check if directory already exists
    if (std::filesystem::exists(dir_path, ec)) {
      if (ec) {
        error_msg = "Failed to check directory existence: " + ec.message();
        return false;
      }
      
      if (!std::filesystem::is_directory(dir_path, ec)) {
        error_msg = "Path exists but is not a directory: " + path;
        return false;
      }
      
      return true;
    }
    
    // Create directory with all parent directories
    bool success = std::filesystem::create_directories(dir_path, ec);
    
    if (ec) {
      error_msg = "Failed to create directory '" + path + "': " + ec.message();
      return false;
    }
    
    return true;
  }

  /// <summary>
  /// Checks if a path exists
  /// </summary>
  /// <param name="path">Path to check</param>
  /// <returns>True if path exists, false otherwise</returns>
  static bool PathExists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
  }

  /// <summary>
  /// Checks if a path is a directory
  /// </summary>
  /// <param name="path">Path to check</param>
  /// <returns>True if path is a directory, false otherwise</returns>
  static bool IsDirectory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
  }

  /// <summary>
  /// Checks if a path is a regular file
  /// </summary>
  /// <param name="path">Path to check</param>
  /// <returns>True if path is a regular file, false otherwise</returns>
  static bool IsFile(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
  }

  /// <summary>
  /// Gets the absolute path from a relative or absolute path
  /// </summary>
  /// <param name="path">Input path</param>
  /// <returns>Absolute path as string</returns>
  static std::string GetAbsolutePath(const std::string& path) {
    std::error_code ec;
    auto abs_path = std::filesystem::absolute(path, ec);
    if (ec) {
      return path;  // Return original path if conversion fails
    }
    return abs_path.string();
  }

  /// <summary>
  /// Gets the parent directory of a path
  /// </summary>
  /// <param name="path">Input path</param>
  /// <returns>Parent directory path</returns>
  static std::string GetParentPath(const std::string& path) {
    std::filesystem::path p(path);
    return p.parent_path().string();
  }

  /// <summary>
  /// Gets the filename from a path
  /// </summary>
  /// <param name="path">Input path</param>
  /// <returns>Filename</returns>
  static std::string GetFilename(const std::string& path) {
    std::filesystem::path p(path);
    return p.filename().string();
  }

  /// <summary>
  /// Joins two paths together
  /// </summary>
  /// <param name="path1">First path component</param>
  /// <param name="path2">Second path component</param>
  /// <returns>Joined path</returns>
  static std::string JoinPath(const std::string& path1, const std::string& path2) {
    std::filesystem::path p1(path1);
    std::filesystem::path p2(path2);
    return (p1 / p2).string();
  }

  /// <summary>
  /// Normalizes a path (removes .. and . components)
  /// </summary>
  /// <param name="path">Input path</param>
  /// <returns>Normalized path</returns>
  static std::string NormalizePath(const std::string& path) {
    std::filesystem::path p(path);
    return p.lexically_normal().string();
  }

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

inline std::vector<char> ReadFile(const std::string& path, const OrtLogger& logger, const OrtApi& ort_api) {
  if (!std::filesystem::exists(path)) {

    std::string message = "TensorRT RTX could not find the file and will create a new one " + path;

    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger,
                                                  ORT_LOGGING_LEVEL_INFO,
                                                  message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    return {};
  }
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    std::string message = "Failed to open file: " + path;
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  // Check if file size determination failed
  if (size < 0) {
    std::string message = "Failed to determine file size: " + path;
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
                          std::to_string(usize) + " bytes): " + path;
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }

  if (usize > 0 && !file.read(buffer.data(), static_cast<std::streamsize>(usize))) {
    std::string message = "Failed to read file: " + path;
    Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
  }
  return buffer;
}

inline void WriteFile(const std::string& path, const void* data, size_t size, const OrtLogger& logger, const OrtApi& ort_api) {
  if (std::filesystem::exists(path)) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
      std::string message = "Failed to open file for writing: " + path;
      Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, message.c_str()));
    }
    file.write(static_cast<const char*>(data), size);
  } else {
    std::string log_message = "TensorRT RTX a new file cache was written to " + path;
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger,
                                                  ORT_LOGGING_LEVEL_INFO,
                                                  log_message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    // Create new file
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file) {
      std::string error_message = "Failed to create file: " + path;
      Ort::ThrowOnError(ort_api.CreateStatus(ORT_EP_FAIL, error_message.c_str()));
    }
    file.write(static_cast<const char*>(data), size);
  }
}

inline void WriteFile(const std::string& path, const std::vector<char>& data, const OrtLogger& logger, const OrtApi& ort_api) { WriteFile(path, data.data(), data.size(), logger, ort_api); }

}  // namespace utils
}  // namespace trt_rtx_ep

