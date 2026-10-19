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

//! \brief Returns true if \p path_string carries a root-name (e.g. "D:") or root-directory
//! (e.g. a leading '\' or '/') component — i.e. it is not a plain relative path.
//!
//! On Windows, drive-relative ("D:foo") and root-relative ("\foo") paths report
//! is_absolute()==false (which requires BOTH a root-name and a root-directory) and contain
//! no "..", so they pass IsAbsolutePath() and IsRelativePathToParentPath() undetected. Both
//! still escape a base directory when composed via std::filesystem::path::operator/=, which
//! discards the base path's directory component whenever the appended path supplies its own
//! root-name or root-directory. This check catches that gap.
inline bool HasRootComponent(const std::string& path_string)
{
    PathString ort_path_string = ToPathString(path_string);
    std::filesystem::path path(ort_path_string.c_str());
    return path.has_root_name() || path.has_root_directory();
}

//! \brief Result of a single-parse classification of a path string's safety-relevant properties.
//! Combines what IsAbsolutePath(), IsRelativePathToParentPath(), and HasRootComponent() each
//! check individually, without re-converting/re-parsing the string for each one.
struct PathClassification
{
    bool is_absolute;           //!< Has both a root-name and a root-directory (e.g. "C:\foo").
    bool has_parent_traversal;  //!< Contains ".." after lexical normalisation (e.g. "../foo").
    bool has_root_component;    //!< Has a root-name or root-directory alone (e.g. "D:foo", "\foo").
};

//! \brief Parses \p path_string once and reports all three safety-relevant path properties.
//!
//! Equivalent to calling IsAbsolutePath(), IsRelativePathToParentPath(), and HasRootComponent()
//! on the same string, but does the UTF conversion and std::filesystem::path construction once
//! instead of three times. Prefer this over the individual checks when more than one of them is
//! needed for the same path string.
inline PathClassification ClassifyPath(const std::string& path_string)
{
    PathClassification result{false, false, false};
    if (path_string.empty())
    {
        return result;
    }

    PathString ort_path_string = ToPathString(path_string);
    std::filesystem::path path(ort_path_string.c_str());

    result.is_absolute = path.is_absolute();
    result.has_root_component = path.has_root_name() || path.has_root_directory();

    std::filesystem::path normalized_path = path.lexically_normal();
    for (const auto& component : normalized_path)
    {
        if (component == "..")
        {
            result.has_parent_traversal = true;
            break;
        }
    }
    return result;
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
