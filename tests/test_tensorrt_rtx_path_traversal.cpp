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

// Unit tests for runtime-cache path sanitization logic.
//
// Exercises the production SanitizeCacheFilename helper (and the underlying
// IsAbsolutePath / IsRelativePathToParentPath validators) used by
// tensorrt_rtx_execution_provider.cc. No GPU or live TRT-RTX runtime required.

#include "utils/path_validation.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

using trt_rtx_ep::IsAbsolutePath;
using trt_rtx_ep::IsRelativePathToParentPath;
using trt_rtx_ep::SanitizeCacheFilename;

// ---------------------------------------------------------------------------
// ComposeSanitizedCachePath uses the production SanitizeCacheFilename helper
// so path-composition corner-cases can be tested without a live EP.
// ---------------------------------------------------------------------------
struct SanitizeResult
{
    bool ok;
    std::filesystem::path path;
};

static SanitizeResult ComposeSanitizedCachePath(const std::filesystem::path& cache_dir, const std::string& model_name)
{
    auto sanitized = SanitizeCacheFilename(model_name);
    if (!sanitized)
        return {false, {}};
    return {true, cache_dir / *sanitized};
}

// ---------------------------------------------------------------------------
// Helper: assert that result is within cache_dir.
// ---------------------------------------------------------------------------
static void ExpectWithinCacheDir(const std::filesystem::path& cache_dir, const SanitizeResult& res)
{
    ASSERT_TRUE(res.ok) << "expected a valid path but got rejection";
    std::filesystem::path norm_result = res.path.lexically_normal();
    std::filesystem::path norm_cache = cache_dir.lexically_normal();
    auto [cache_end, _] = std::mismatch(norm_cache.begin(), norm_cache.end(), norm_result.begin());
    EXPECT_TRUE(cache_end == norm_cache.end()) << "result escapes cache directory: " << res.path;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(RuntimeCachePathTraversalTest, BenignName)
{
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "my_model");
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "my_model");
}

TEST(RuntimeCachePathTraversalTest, ParentTraversalRelative)
{
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "../../evil");
    // After sanitization the only surviving component is "evil" — safe.
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "evil");
}

TEST(RuntimeCachePathTraversalTest, BareDotDotRejected)
{
    // A bare ".." whose filename() is also ".." — must be rejected outright.
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "..");
    EXPECT_FALSE(res.ok) << "bare '..' must be rejected";
}

TEST(RuntimeCachePathTraversalTest, AbsolutePath)
{
    std::filesystem::path cache("C:\\runtime_cache");
#if defined(_WIN32)
    auto res = ComposeSanitizedCachePath(cache, "C:\\Windows\\System32\\evil");
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "evil");
#else
    auto res = ComposeSanitizedCachePath(cache, "/etc/passwd");
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "passwd");
#endif
}

TEST(RuntimeCachePathTraversalTest, NestedTraversal)
{
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "subdir/../../../etc/shadow");
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "shadow");
}

TEST(RuntimeCachePathTraversalTest, DotsInNameNotTraversal)
{
    // "model..name" has no traversal component and must be accepted as-is.
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "model..name");
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "model..name");
}

TEST(RuntimeCachePathTraversalTest, EngineIdTraversal)
{
    // Simulates the engine_id code path in CreateNodeComputeInfoFromGraph.
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "../../other_dir/stolen_cache");
    ExpectWithinCacheDir(cache, res);
    EXPECT_EQ(res.path.filename(), "stolen_cache");
}

TEST(RuntimeCachePathTraversalTest, EmptyNameRejected)
{
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "");
    EXPECT_FALSE(res.ok) << "empty name must be rejected unconditionally";
}

// ---------------------------------------------------------------------------
// Single-dot behaviour: "." and "./" prefixes are not traversal components
// and must not cause failures or security violations.
// ---------------------------------------------------------------------------

TEST(RuntimeCachePathTraversalTest, SingleDotAloneIsNotTraversal)
{
    // "." refers to the current directory — not a parent traversal.
    EXPECT_FALSE(IsRelativePathToParentPath("."));
    EXPECT_FALSE(IsAbsolutePath("."));
}

TEST(RuntimeCachePathTraversalTest, SingleDotPrefixIsNotTraversal)
{
    // "./model" is equivalent to "model" — no parent component.
    EXPECT_FALSE(IsRelativePathToParentPath("./model.onnx"));
    EXPECT_FALSE(IsAbsolutePath("./model.onnx"));
}

TEST(RuntimeCachePathTraversalTest, SingleDotDotSlashFlagged)
{
    // "../sibling" contains a ".." component and must be rejected.
    EXPECT_TRUE(IsRelativePathToParentPath("../sibling.onnx"));
}

// ---------------------------------------------------------------------------
// Symlink note: these checks are lexical only. A symlink named "safe_name"
// that resolves outside the cache directory would NOT be caught by
// IsAbsolutePath / IsRelativePathToParentPath / SanitizeCacheFilename.
// Symlink resolution requires the path to exist at validation time and is
// not performed here — this is a known limitation documented in
// path_validation.h.
// ---------------------------------------------------------------------------

TEST(RuntimeCachePathTraversalTest, SymlinkLimitationDocumented)
{
    // A textually safe name passes validation regardless of what it might
    // resolve to via a symlink. This test documents the known limitation:
    // the result here is "accepted" even though in a real filesystem this
    // name could be a symlink pointing anywhere.
    std::filesystem::path cache("C:\\runtime_cache");
    auto res = ComposeSanitizedCachePath(cache, "safe_looking_symlink");
    EXPECT_TRUE(res.ok) << "lexically safe name accepted (symlink target not checked — known limitation)";
}
