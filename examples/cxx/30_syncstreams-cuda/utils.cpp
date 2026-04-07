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
#include "lodepng.h"

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <limits.h>
#include <mach-o/dyld.h>
#elif __linux__
#include <limits.h>
#include <unistd.h>
#endif

std::filesystem::path get_executable_parent_path() { return get_executable_path().parent_path(); }

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

constexpr int image_dim = 224;

void loadInputImage(float* pData, const char* imageFileName) {
  unsigned char* image;
  unsigned int width, height;
  unsigned int error =
      lodepng_decode32_file(&image, &width, &height, imageFileName);
  if (error) {
    printf("\nFailed to load the input image: %s. Exiting\n", lodepng_error_text(error));
    exit(EXIT_FAILURE);
  }

  if (width != image_dim || height != image_dim) {
    printf("\nImage not of right size (%ux%u, expected %ux%u). Exiting\n",
           width, height, image_dim, image_dim);
    exit(EXIT_FAILURE);
  }

  for (uint32_t y = 0; y < height; y++)
    for (uint32_t x = 0; x < width; x++) {
      unsigned char r = image[(y * width + x) * 4 + 0];
      unsigned char g = image[(y * width + x) * 4 + 1];
      unsigned char b = image[(y * width + x) * 4 + 2];

      pData[0 * width * height + y * width + x] = (float)b;
      pData[1 * width * height + y * width + x] = (float)g;
      pData[2 * width * height + y * width + x] = (float)r;
    }

  free(image);
}

static unsigned char clampAndConvert(float val) {
  if (val < 0) val = 0;
  if (val > 255) val = 255;
  return (unsigned char)val;
}

void saveOutputImage(float* pData, const char* imageFileName) {
  unsigned int width = image_dim, height = image_dim;

  std::vector<unsigned char> image(width * height * 4);
  for (uint32_t y = 0; y < height; y++)
    for (uint32_t x = 0; x < width; x++) {
      float b = pData[0 * width * height + y * width + x];
      float g = pData[1 * width * height + y * width + x];
      float r = pData[2 * width * height + y * width + x];

      image[(y * width + x) * 4 + 0] = clampAndConvert(r);
      image[(y * width + x) * 4 + 1] = clampAndConvert(g);
      image[(y * width + x) * 4 + 2] = clampAndConvert(b);
      image[(y * width + x) * 4 + 3] = 255;
    }

  lodepng_encode32_file(imageFileName, &image[0], width, height);
}
