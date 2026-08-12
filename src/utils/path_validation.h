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

#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "path_string.h"

// NOTE: all helpers in this header perform lexical checks only.
// Symlinks are NOT resolved: a symlink whose textual name passes validation
// but resolves outside the intended directory is not caught here.
// This is a known limitation — resolving symlinks requires the path to exist
// at validation time, which cannot be guaranteed in all call sites.

namespace trt_rtx_ep
{

//! \brief Returns true if \p path_string represents an absolute filesystem path.
inline bool IsAbsolutePath(const std::string& path_string)
{
    PathString ort_path_string = ToPathString(path_string);
    return std::filesystem::path(ort_path_string.c_str()).is_absolute();
}

//! \brief Returns true if \p path_string contains a parent-directory traversal component ("..").
//!
//! Uses lexical normalisation so that "a/../b" resolves to "b" (not flagged), while
//! "../b" or "a/../../b" still contain ".." after normalisation and ARE flagged.
//! Filenames that merely contain ".." as a substring (e.g. "file..name.txt") are NOT flagged.
inline bool IsRelativePathToParentPath(const std::string& path_string)
{
    if (path_string.empty())
    {
        return false;
    }

#if defined(_WIN32)
    PathString ort_path_string = ToPathString(path_string);
    std::filesystem::path path(ort_path_string.c_str());
#else
    std::filesystem::path path(path_string);
#endif

    std::filesystem::path normalized_path = path.lexically_normal();
    for (const auto& component : normalized_path)
    {
        if (component == "..")
        {
            return true;
        }
    }
    return false;
}

//! \brief Sanitizes a model-derived name for use as a runtime-cache filename.
//!
//! If \p raw_name is absolute or contains parent-directory traversal components,
//! only the base filename is kept. Returns std::nullopt if the result is still
//! unsafe or empty after stripping (e.g. a bare "..").
//!
//! \return Sanitized single-component path, or std::nullopt if rejected.
inline std::optional<std::filesystem::path> SanitizeCacheFilename(const std::string& raw_name)
{
    std::filesystem::path name_path(ToPathString(raw_name));
    if (IsAbsolutePath(raw_name) || IsRelativePathToParentPath(raw_name))
    {
        name_path = name_path.filename();
    }
    std::string safe = PathToUTF8String(name_path.native());
    if (safe.empty() || IsAbsolutePath(safe) || IsRelativePathToParentPath(safe))
    {
        return std::nullopt;
    }
    return name_path;
}

}  // namespace trt_rtx_ep
