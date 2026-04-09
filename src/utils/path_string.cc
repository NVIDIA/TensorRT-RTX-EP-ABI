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

#include "path_string.h"

#ifdef _WIN32
#include <windows.h>
#include <stdexcept>

std::string ToUTF8String(std::wstring_view s) {
  if (s.empty()) {
    return std::string();
  }

  // Get the required buffer size
  int size_needed = WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), 
                                        nullptr, 0, nullptr, nullptr);
  if (size_needed <= 0) {
    throw std::runtime_error("Failed to convert wide string to UTF-8");
  }

  std::string result(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), 
                     &result[0], size_needed, nullptr, nullptr);
  
  return result;
}

std::wstring ToWideString(std::string_view s) {
  if (s.empty()) {
    return std::wstring();
  }

  // Get the required buffer size
  int size_needed = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), 
                                        nullptr, 0);
  if (size_needed <= 0) {
    throw std::runtime_error("Failed to convert UTF-8 string to wide string");
  }

  std::wstring result(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), 
                     &result[0], size_needed);
  
  return result;
}

#endif  // _WIN32

