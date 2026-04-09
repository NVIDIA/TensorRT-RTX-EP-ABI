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

#include "utils.h"
#include <filesystem>
#include <iostream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <limits.h>
#include <mach-o/dyld.h>
#elif __linux__
#include <limits.h>
#include <unistd.h>
#endif

std::filesystem::path get_executable_path() {
#ifdef _WIN32
  std::vector<wchar_t> pathBuf(MAX_PATH);
  DWORD length = GetModuleFileNameW(NULL, pathBuf.data(), static_cast<DWORD>(pathBuf.size()));

  while (length == pathBuf.size()) {
    pathBuf.resize(pathBuf.size() * 2);
    length = GetModuleFileNameW(NULL, pathBuf.data(), static_cast<DWORD>(pathBuf.size()));
  }

  if (length == 0) {
    std::cerr << "Error: GetModuleFileNameW failed with error "
              << GetLastError() << std::endl;
    return {};
  }
  return std::filesystem::path(pathBuf.data());

#elif __APPLE__
  std::vector<char> pathBuf(PATH_MAX);
  uint32_t length = pathBuf.size();
  if (_NSGetExecutablePath(pathBuf.data(), &length) != 0) {
    pathBuf.resize(length + 1);
    if (_NSGetExecutablePath(pathBuf.data(), &length) != 0) {
      std::cerr << "Error: _NSGetExecutablePath failed" << std::endl;
      return {};
    }
  }
  return std::filesystem::canonical(pathBuf.data());

#elif __linux__
  return std::filesystem::canonical(
      std::filesystem::read_symlink("/proc/self/exe"));
#endif
}
