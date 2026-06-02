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

#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

// Use std::thread instead of std::async for concurrent tests.
#define USE_STD_THREAD 1

extern std::unique_ptr<Ort::Env> ort_env;

// ---------------------------------------------------------------------------
// Base fixture – checks device availability and model file before every test.
// ---------------------------------------------------------------------------

class CompileModelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty())
            << "No TRT RTX EP devices found. "
               "Ensure the EP library is present and an NVIDIA RTX GPU is available.";
        ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath)) << "Model not found at: " << kModelPath;
    }
};

// ---------------------------------------------------------------------------
// Helpers shared by all compile tests
// ---------------------------------------------------------------------------

namespace
{

constexpr int kNumThreads = 5;

// ---------------------------------------------------------------------------
// CompileConfig – controls which API paths are exercised in a single compile.
// ---------------------------------------------------------------------------

struct CompileConfig
{
    bool embed_mode;        ///< passed to SetEpContextEmbedMode
    bool input_from_buffer; ///< use SetInputModelFromBuffer instead of SetInputModelPath
    bool output_to_buffer;  ///< use SetOutputModelBuffer instead of SetOutputModelPath
};

/// Short human-readable label used as the GTest parameter suffix.
std::string config_name(const CompileConfig& c)
{
    return std::string(c.embed_mode ? "EmbedOn" : "EmbedOff")
        + "_In" + (c.input_from_buffer ? "Buf" : "File")
        + "_Out" + (c.output_to_buffer ? "Buf" : "File");
}

// ---------------------------------------------------------------------------
// toOrtDirString – like toOrtString but with a guaranteed trailing separator.
//
// ORT's SetEpContextBinaryInformation internally constructs a
// std::filesystem::path and rejects it when
//   has_filename() == true && extension() == ""
// (a bare name without extension looks like a file, not a directory).
// A trailing '/' makes has_filename() return false, bypassing the check.
#ifdef _WIN32
inline std::wstring toOrtDirString(const std::filesystem::path& dir_path)
{
    auto s = dir_path.wstring();
    if (s.empty() || (s.back() != L'/' && s.back() != L'\\'))
        s += L'/';
    return s;
}
#else
inline std::string toOrtDirString(const std::filesystem::path& dir_path)
{
    auto s = dir_path.string();
    if (s.empty() || s.back() != '/')
        s += '/';
    return s;
}
#endif

// ---------------------------------------------------------------------------
// CompileResult – holds either a file path or an in-memory buffer.
// ---------------------------------------------------------------------------

struct CompileResult
{
    std::filesystem::path path;           ///< non-empty when output went to a file
    std::vector<char>     buffer;         ///< non-empty when output went to a buffer
    /// Always set by compile_model_flexible() to the user's intended output path.
    /// For buffer output + embed_mode=false this also tells the EP where to find the
    /// external engine binary when the compiled buffer is later loaded into a session
    /// (passed through kOrtSessionOptionEpContextFilePath in make_session_from_result).
    std::filesystem::path intended_ctx_path;

    bool is_file()   const { return !path.empty(); }
    bool is_buffer() const { return !buffer.empty(); }
};

// ---------------------------------------------------------------------------
// Session-option factory reused by every test.
// ---------------------------------------------------------------------------

Ort::SessionOptions make_session_options()
{
    // static cudaStream_t stream = nullptr;
    // if (stream == nullptr)
    //     cudaStreamCreate(&stream);

    Ort::SessionOptions so;
    Ort::KeyValuePairs  ep_opts;
    ep_opts.Add("enable_cuda_graph", "1");
    ep_opts.Add("nv_runtime_cache_path", "./rt_cache");
    // ep_opts.Add("has_user_compute_stream", "1");
    // auto stream_ptr = std::to_string(reinterpret_cast<uint64_t>(stream));
    // ep_opts.Add("user_compute_stream", stream_ptr.c_str());
    so.AppendExecutionProvider_V2(*ort_env, get_trt_rtx_devices(*ort_env), ep_opts);
    return so;
}

// ---------------------------------------------------------------------------
// Read an entire file into a byte vector.
// ---------------------------------------------------------------------------

std::vector<char> read_file_bytes(const std::filesystem::path& p)
{
    std::ifstream file(p, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + p.string());
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// ---------------------------------------------------------------------------
// compile_model_flexible – single compile entry-point for all parameterized tests.
//
//   input_path   : ONNX model to compile.
//   output_path  : always required –
//                    !cfg.output_to_buffer ? compiled model written here
//                     cfg.output_to_buffer ? parent/stem provide the binary-
//                     context location when embed_mode == false
//
// Throws std::runtime_error on failure.
// ---------------------------------------------------------------------------

CompileResult compile_model_flexible(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const CompileConfig&         cfg)
{
    auto                         so = make_session_options();
    Ort::ModelCompilationOptions opts(*ort_env, so);
    opts.SetEpContextEmbedMode(cfg.embed_mode);

    // -- input --
    std::vector<char> input_bytes; // must outlive Ort::CompileModel
    if (cfg.input_from_buffer)
    {
        input_bytes = read_file_bytes(input_path);
        opts.SetInputModelFromBuffer(input_bytes.data(), input_bytes.size());
    }
    else
    {
        opts.SetInputModelPath(toOrtString(input_path).c_str());
    }

    // -- output --
    Ort::AllocatorWithDefaultOptions allocator;
    void*  out_ptr  = nullptr;
    size_t out_size = 0;

    if (cfg.output_to_buffer)
    {
        opts.SetOutputModelBuffer(allocator, &out_ptr, &out_size);
        // When not embedding, the binary context still needs a location on disk.
        if (!cfg.embed_mode)
        {
            const auto ctx_dir = output_path.parent_path();
            std::filesystem::create_directories(ctx_dir);
            opts.SetEpContextBinaryInformation(
                toOrtDirString(ctx_dir).c_str(),
                toOrtString(output_path.stem()).c_str());
        }
    }
    else
    {
        opts.SetOutputModelPath(toOrtString(output_path).c_str());
    }

    const auto status = Ort::CompileModel(*ort_env, opts);
    if (!status.IsOK())
        throw std::runtime_error(status.GetErrorMessage());

    CompileResult result;
    result.intended_ctx_path = output_path;
    if (cfg.output_to_buffer)
    {
        if (out_ptr != nullptr && out_size > 0)
        {
            result.buffer.assign(
                static_cast<char*>(out_ptr),
                static_cast<char*>(out_ptr) + out_size);
        }
        allocator.Free(out_ptr);
    }
    else
    {
        result.path = output_path;
    }
    return result;
}

// ---------------------------------------------------------------------------
// make_session_from_result – open a session from a CompileResult.
// Each call creates its own SessionOptions (not thread-safe to share).
// ---------------------------------------------------------------------------

Ort::Session make_session_from_result(const CompileResult& result)
{
    auto so = make_session_options();

    // When loading the compiled model from a buffer, there is no on-disk model location
    // that the EP can use to resolve the relative ep_cache_context path back to an
    // external engine binary. Follow the same pattern as the QNN EP tests and pass the
    // intended context-model path via kOrtSessionOptionEpContextFilePath so the EP can
    // anchor the relative path lookup to the correct directory.
    if (result.is_buffer() && !result.intended_ctx_path.empty())
    {
        // kOrtSessionOptionEpContextFilePath is a UTF-8 narrow string per the
        // WS5 fix (decoded via MultiByteToWideChar(CP_UTF8, ...) inside the
        // EP). path::string() returns CP_ACP on Windows, which mis-decodes
        // for any non-ASCII char. Use u8string() to feed UTF-8 bytes.
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath,
                          reinterpret_cast<const char*>(result.intended_ctx_path.u8string().c_str()));
    }

    if (result.is_file())
        return Ort::Session(*ort_env, toOrtString(result.path).c_str(), so);
    return Ort::Session(*ort_env, result.buffer.data(), result.buffer.size(), so);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Parameterized fixture – inherits SetUp checks from CompileModelTest.
// ---------------------------------------------------------------------------

class CompileModelParamTest
    : public CompileModelTest
    , public ::testing::WithParamInterface<CompileConfig>
{};

// ---------------------------------------------------------------------------
// Test 1: single-threaded compile then session load + shape check.
// ---------------------------------------------------------------------------

TEST_P(CompileModelParamTest, CompilesModel)
{
    const auto& cfg         = GetParam();
    const auto  output_path = kModelPath.parent_path() / "context.onnx";
    std::filesystem::remove(output_path);

    CompileResult result;
    ASSERT_NO_THROW(result = compile_model_flexible(kModelPath, output_path, cfg));

    if (result.is_file())
    {
        ASSERT_TRUE(std::filesystem::is_regular_file(result.path))
            << "Compiled model not found at: " << result.path;
    }
    else
    {
        ASSERT_FALSE(result.buffer.empty()) << "Output buffer is empty after compilation";
    }

    // Verify the compiled model loads into a valid session.
    Ort::Session session{nullptr};
    ASSERT_NO_THROW(session = make_session_from_result(result));
    EXPECT_EQ(session.GetInputCount(),  1u);
    EXPECT_EQ(session.GetOutputCount(), 1u);
}

// ---------------------------------------------------------------------------
// Test 2: kNumThreads concurrent compilations of the same source model,
//         each writing to a unique output destination.
// ---------------------------------------------------------------------------

TEST_P(CompileModelParamTest, ConcurrentCompile)
{
    const auto& cfg = GetParam();

    std::vector<std::filesystem::path> output_paths(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i)
    {
        output_paths[i] =
            kModelPath.parent_path() / ("context_thread_" + std::to_string(i) + ".onnx");
        std::filesystem::remove(output_paths[i]);
    }

#if USE_STD_THREAD
    std::vector<CompileResult>      results(kNumThreads);
    std::vector<std::exception_ptr> exceptions(kNumThreads);
    std::vector<std::thread>        threads;
    threads.reserve(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i)
    {
        threads.emplace_back(
            [&output_paths, &results, &exceptions, &cfg, i]()
            {
                try
                {
                    results[i] = compile_model_flexible(kModelPath, output_paths[i], cfg);
                }
                catch (...)
                {
                    exceptions[i] = std::current_exception();
                }
            });
    }
    for (auto& t : threads)
        t.join();

    for (int i = 0; i < kNumThreads; ++i)
    {
        EXPECT_FALSE(exceptions[i]) << "Thread " << i << " compilation failed";
        if (!exceptions[i])
        {
            if (results[i].is_file())
                EXPECT_TRUE(std::filesystem::is_regular_file(results[i].path))
                    << "Thread " << i << " output file missing: " << results[i].path;
            else
                EXPECT_FALSE(results[i].buffer.empty())
                    << "Thread " << i << " output buffer is empty";
        }
    }
#else
    std::vector<std::future<CompileResult>> futures;
    futures.reserve(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i)
    {
        futures.push_back(std::async(std::launch::async,
            [&output_paths, &cfg, i]()
            {
                return compile_model_flexible(kModelPath, output_paths[i], cfg);
            }));
    }

    for (int i = 0; i < kNumThreads; ++i)
    {
        CompileResult result;
        ASSERT_NO_THROW(result = futures[i].get()) << "Thread " << i << " compilation failed";
        if (result.is_file())
            EXPECT_TRUE(std::filesystem::is_regular_file(result.path))
                << "Thread " << i << " output file missing: " << result.path;
        else
            EXPECT_FALSE(result.buffer.empty())
                << "Thread " << i << " output buffer is empty";
    }
#endif
}

// ---------------------------------------------------------------------------
// Test 3: compile once, then kNumThreads sessions are instantiated concurrently
//         from the same result, followed by sequential inference on each.
// ---------------------------------------------------------------------------

//(TODO) Enable after fix
/*
TEST_P(CompileModelParamTest, ConcurrentSessionInstantiationAndSequentialInference)
{
    const auto& cfg      = GetParam();
    const auto  ctx_path = kModelPath.parent_path() / "context_concurrent.onnx";
    std::filesystem::remove(ctx_path);

    // Compile once (single-threaded).
    CompileResult compiled;
    ASSERT_NO_THROW(compiled = compile_model_flexible(kModelPath, ctx_path, cfg));

#if USE_STD_THREAD
    std::vector<std::unique_ptr<Ort::Session>> sessions(kNumThreads);
    std::vector<std::exception_ptr>            exceptions(kNumThreads);
    std::vector<std::thread>                   threads;
    threads.reserve(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i)
    {
        threads.emplace_back(
            [&compiled, &sessions, &exceptions, i]()
            {
                try
                {
                    sessions[i] =
                        std::make_unique<Ort::Session>(make_session_from_result(compiled));
                }
                catch (...)
                {
                    exceptions[i] = std::current_exception();
                }
            });
    }
    for (auto& t : threads)
        t.join();
    for (int i = 0; i < kNumThreads; ++i)
        ASSERT_FALSE(exceptions[i]) << "Thread " << i << " failed to instantiate session";
#else
    std::vector<std::future<std::unique_ptr<Ort::Session>>> futures;
    futures.reserve(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i)
    {
        futures.push_back(std::async(std::launch::async,
            [&compiled]()
            {
                return std::make_unique<Ort::Session>(make_session_from_result(compiled));
            }));
    }

    std::vector<std::unique_ptr<Ort::Session>> sessions;
    sessions.reserve(kNumThreads);
    for (int i = 0; i < kNumThreads; ++i)
        ASSERT_NO_THROW(sessions.push_back(futures[i].get()))
            << "Thread " << i << " failed to instantiate session";
#endif

    // Sequential inference on each session.
    for (int i = 0; i < static_cast<int>(sessions.size()); ++i)
    {
        ASSERT_NE(sessions[i], nullptr);
        ASSERT_NO_THROW(run_with_cpu_bindings(*sessions[i], 1)) << "Inference failed on session " << i;
    }
}
*/

// ---------------------------------------------------------------------------
// Instantiate all 8 combinations: embed × input-mode × output-mode
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    TensorRTRTXEpTest_AllModes,
    CompileModelParamTest,
    ::testing::Values(
        CompileConfig{false, false, false},
        CompileConfig{false, false, true },
        CompileConfig{false, true,  false},
        CompileConfig{false, true,  true },
        CompileConfig{true,  false, false},
        CompileConfig{true,  false, true },
        CompileConfig{true,  true,  false},
        CompileConfig{true,  true,  true }
    ),
    [](const ::testing::TestParamInfo<CompileConfig>& info)
    {
        return config_name(info.param);
    });

// ---------------------------------------------------------------------------
// Special-character and runtime-cache path tests
// ---------------------------------------------------------------------------

namespace
{

// Test-case descriptor for the parameterized path tests below.
//   subdir_name -- folder name exercising a specific character class. Stored
//                  as std::filesystem::path so the wide-native form on Windows
//                  carries the right code points regardless of source-file
//                  encoding. Literals below use \uXXXX / \UXXXXXXXX escapes
//                  exclusively so the source stays ASCII-portable and does
//                  not depend on the /utf-8 compile flag.
//   label       -- ASCII-safe identifier used by GoogleTest to name the
//                  parameterized instance (test output, --gtest_filter, etc.).
struct SpecialCharsCase
{
    std::filesystem::path subdir_name;
    const char*           label;
};

// Returns the subdirectory under the model directory rooted at the supplied
// subdir name. Used by the parameterized special-chars tests below.
inline std::filesystem::path special_chars_base_dir(
    const std::filesystem::path& subdir)
{
    return kModelPath.parent_path() / subdir;
}

// CP1252-mappable character classes -- safe to exercise via path-load (the
// existing test architecture). These characters all have CP_ACP byte
// mappings on Western-Windows runners, so the wide -> narrow conversion
// inside framework code (model_metadef_id_generator.cc:43) succeeds.
//
// All Windows wide literals use \uXXXX escapes so the source file stays
// ASCII-portable; no /utf-8 compile flag is required.
inline std::vector<SpecialCharsCase> cp1252_mappable_cases()
{
    return {
#ifdef _WIN32
        // The original case carried through from the pre-MR test.
        { std::filesystem::path{L"special (chars) r\u00e9sum\u00e9"},   "EAcuteResume" },
        { std::filesystem::path{L"Verd\u00fa2"},                        "VerduTwo"     },  // u-acute U+00FA
        { std::filesystem::path{L"M\u00fcnchen"},                       "Muenchen"     },  // u-umlaut U+00FC
        { std::filesystem::path{L"Caf\u00e9_Gr\u00f6\u00dfe"},          "CafeGroesse"  },  // e-acute, o-umlaut, sharp-s
#else
        { std::filesystem::path{u8"special (chars) r\u00e9sum\u00e9"},  "EAcuteResume" },
        { std::filesystem::path{u8"Verd\u00fa2"},                       "VerduTwo"     },
        { std::filesystem::path{u8"M\u00fcnchen"},                      "Muenchen"     },
        { std::filesystem::path{u8"Caf\u00e9_Gr\u00f6\u00dfe"},         "CafeGroesse"  },
#endif
    };
}

// Full character-class matrix: CP1252-mappable PLUS classes outside the
// system ANSI code page (non-CP1252 Latin, CJK, Cyrillic, emoji, mixed-
// script). The non-CP1252 entries are NOT safe to exercise via path-load --
// Microsoft framework code at model_metadef_id_generator.cc:43 stringifies
// the held wide path via CP_ACP and throws on any char without a CP1252
// mapping. They MUST be exercised via buffer-load (Ort::Session with bytes,
// not a wchar_t* path) -- see TensorRTRTXEpTest_BufferLoadPathTest.
inline std::vector<SpecialCharsCase> all_special_chars_cases()
{
    auto v = cp1252_mappable_cases();
#ifdef _WIN32
    v.push_back({ std::filesystem::path{L"\u0141ukasz"},                                  "Lukasz"          });
    v.push_back({ std::filesystem::path{L"\u0141\u00f3d\u017a"},                          "Lodz"            });
    v.push_back({ std::filesystem::path{L"\u017dlu\u0165ou\u010dk\u00fd"},                "Zlutoucky"       });
    v.push_back({ std::filesystem::path{L"\u5c71\u7530"},                                 "Yamada_CJK"      });
    v.push_back({ std::filesystem::path{L"\u041c\u043e\u0441\u043a\u0432\u0430"},         "Moskva_Cyrillic" });
    v.push_back({ std::filesystem::path{L"test_\U0001F389_party"},                        "Party_Emoji"     });
    v.push_back({ std::filesystem::path{L"\u00e9_\u6f22_\u03b1_\u03a9"},                  "MultiScript"     });
#else
    v.push_back({ std::filesystem::path{u8"\u0141ukasz"},                                 "Lukasz"          });
    v.push_back({ std::filesystem::path{u8"\u0141\u00f3d\u017a"},                         "Lodz"            });
    v.push_back({ std::filesystem::path{u8"\u017dlu\u0165ou\u010dk\u00fd"},               "Zlutoucky"       });
    v.push_back({ std::filesystem::path{u8"\u5c71\u7530"},                                "Yamada_CJK"      });
    v.push_back({ std::filesystem::path{u8"\u041c\u043e\u0441\u043a\u0432\u0430"},        "Moskva_Cyrillic" });
    v.push_back({ std::filesystem::path{u8"test_\U0001F389_party"},                       "Party_Emoji"     });
    v.push_back({ std::filesystem::path{u8"\u00e9_\u6f22_\u03b1_\u03a9"},                 "MultiScript"     });
#endif
    return v;
}

// Compile input_path ? output_path (embed_mode=true, file-to-file) and point
// nv_runtime_cache_path at cache_dir.  Pass an empty cache_dir to omit the option.
void compile_with_cache(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const std::filesystem::path& cache_dir = {},
    const std::string& cache_dir_str = {})
{
    Ort::SessionOptions so;
    Ort::KeyValuePairs  ep_opts;
    ep_opts.Add("enable_cuda_graph", "1");
    if (!cache_dir.empty())
    {
        // The EP's nv_runtime_cache_path option expects a UTF-8 narrow string
        // (the WS5 patch decodes it via MultiByteToWideChar(CP_UTF8, ...)).
        // path::string() on Windows would produce CP_ACP-encoded bytes, which
        // mis-decode under UTF-8 once any non-ASCII char appears. C++17's
        // path::u8string() returns std::string encoded in UTF-8.
        ep_opts.Add("nv_runtime_cache_path", reinterpret_cast<const char*>(cache_dir.u8string().c_str()));
    }
    so.AppendExecutionProvider_V2(*ort_env, get_trt_rtx_devices(*ort_env), ep_opts);

    Ort::ModelCompilationOptions opts(*ort_env, so);
    opts.SetEpContextEmbedMode(true);
    opts.SetInputModelPath(toOrtString(input_path).c_str());
    opts.SetOutputModelPath(toOrtString(output_path).c_str());
    const auto status = Ort::CompileModel(*ort_env, opts);
    if (!status.IsOK())
        throw std::runtime_error(status.GetErrorMessage());
}

// Create a session from a compiled context file with a specific runtime cache directory.
Ort::Session make_session_with_cache(
    const std::filesystem::path& ctx_path,
    const std::filesystem::path& cache_dir)
{
    Ort::SessionOptions so;
    Ort::KeyValuePairs  ep_opts;
    ep_opts.Add("enable_cuda_graph", "1");
    // See note in compile_with_cache: the EP expects UTF-8 narrow on
    // nv_runtime_cache_path; path::string() returns CP_ACP on Windows.
    ep_opts.Add("nv_runtime_cache_path", reinterpret_cast<const char*>(cache_dir.u8string().c_str()));
    so.AppendExecutionProvider_V2(*ort_env, get_trt_rtx_devices(*ort_env), ep_opts);
    return Ort::Session(*ort_env, toOrtString(ctx_path).c_str(), so);
}

// Returns true if dir contains at least one regular file (ignores sub-dirs).
bool dir_has_files(const std::filesystem::path& dir)
{
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.is_regular_file())
            return true;
    return false;
}

} // anonymous namespace

class TensorRTRTXEpTest_PathTest : public CompileModelTest {};

// Parameterized fixture for special-chars tests that go through path-load
// (Ort::Session(env, wchar_t* path, opts)). Limited to CP1252-mappable cases
// to stay inside the EP scope -- non-CP1252 chars trip the Microsoft
// framework throw at model_metadef_id_generator.cc:43 which is upstream of
// the NVIDIA EP. The full character matrix is exercised via buffer-load
// in TensorRTRTXEpTest_BufferLoadPathTest below.
class TensorRTRTXEpTest_PathTestP
    : public CompileModelTest,
      public ::testing::WithParamInterface<SpecialCharsCase>
{};

// -- Context input path: source ONNX lives on a path with special characters -

TEST_P(TensorRTRTXEpTest_PathTestP, ContextInputPathSpecialChars)
{
    const auto base     = special_chars_base_dir(GetParam().subdir_name);
    const auto src_dir  = base / "input";
    const auto src_copy = src_dir / kModelPath.filename();
    const auto ctx_path = base / "ctx_from_special" / "context.onnx";

    // Clear any artifacts from prior runs so subsequent assertions reflect the
    // CURRENT run's behavior, not stale files.
    std::error_code ec;
    std::filesystem::remove_all(src_dir, ec);
    std::filesystem::remove_all(ctx_path.parent_path(), ec);

    std::filesystem::create_directories(src_dir);
    std::filesystem::create_directories(ctx_path.parent_path());
    std::filesystem::copy_file(kModelPath, src_copy,
                               std::filesystem::copy_options::overwrite_existing);

    CompileResult result;
    ASSERT_NO_THROW(result = compile_model_flexible(src_copy, ctx_path,
                                                    CompileConfig{true, false, false}));
    ASSERT_TRUE(std::filesystem::is_regular_file(result.path));

    Ort::Session session{nullptr};
    ASSERT_NO_THROW(session = make_session_from_result(result));
    EXPECT_EQ(session.GetInputCount(),  1u);
    EXPECT_EQ(session.GetOutputCount(), 1u);
}

// -- Context output path: compiled context model written to a special-chars path

TEST_P(TensorRTRTXEpTest_PathTestP, ContextOutputPathSpecialChars)
{
    const auto base     = special_chars_base_dir(GetParam().subdir_name);
    const auto ctx_path = base / "ctx_to_special" / "context.onnx";

    // Clear any artifacts from prior runs.
    std::error_code ec;
    std::filesystem::remove_all(ctx_path.parent_path(), ec);

    std::filesystem::create_directories(ctx_path.parent_path());

    CompileResult result;
    ASSERT_NO_THROW(result = compile_model_flexible(kModelPath, ctx_path,
                                                    CompileConfig{true, false, false}));
    ASSERT_TRUE(std::filesystem::is_regular_file(result.path));

    Ort::Session session{nullptr};
    ASSERT_NO_THROW(session = make_session_from_result(result));
    EXPECT_EQ(session.GetInputCount(),  1u);
    EXPECT_EQ(session.GetOutputCount(), 1u);
}

// -- nv_runtime_cache_path: plain ASCII directory ----------------------------

TEST_F(TensorRTRTXEpTest_PathTest, RuntimeCacheNormalPath)
{
    const auto cache_dir = kModelPath.parent_path() / "rt_cache_normal_test";
    const auto ctx_path  = kModelPath.parent_path() / "rt_cache_normal_ctx" / "context.onnx";

    // Clear any artifacts from prior runs so dir_has_files(cache_dir) below
    // reflects the current run's cache write, not stale files.
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
    std::filesystem::remove_all(ctx_path.parent_path(), ec);

    std::filesystem::create_directories(ctx_path.parent_path());

    ASSERT_NO_THROW(compile_with_cache(kModelPath, ctx_path, cache_dir));
    ASSERT_TRUE(std::filesystem::is_regular_file(ctx_path));

    // First session: EP creates the engine and writes the cache on destruction.
    {
        Ort::Session s = make_session_with_cache(ctx_path, cache_dir);
        EXPECT_EQ(s.GetInputCount(),  1u);
        EXPECT_EQ(s.GetOutputCount(), 1u);
        ASSERT_NO_THROW(run_with_cpu_bindings(s, 1));
    } // IExecutionContextDeleter fires here ? cache serialized

    EXPECT_TRUE(dir_has_files(cache_dir))
        << "Expected runtime cache files in: " << cache_dir;

    // Second session: EP should deserialize from the written cache.
    ASSERT_NO_THROW({
        Ort::Session s = make_session_with_cache(ctx_path, cache_dir);
        EXPECT_EQ(s.GetInputCount(),  1u);
        EXPECT_EQ(s.GetOutputCount(), 1u);
    });
}

// -- nv_runtime_cache_path: directory with special characters ----------------

TEST_P(TensorRTRTXEpTest_PathTestP, RuntimeCacheSpecialCharsPath)
{
    const auto base      = special_chars_base_dir(GetParam().subdir_name);
    const auto cache_dir = base / "rt_cache";
    const auto ctx_path  = base / "rt_cache_ctx" / "context.onnx";

    // Clear any artifacts from prior runs so dir_has_files(cache_dir) below
    // reflects the current run's cache write, not stale files.
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
    std::filesystem::remove_all(ctx_path.parent_path(), ec);

    std::filesystem::create_directories(ctx_path.parent_path());

    ASSERT_NO_THROW(compile_with_cache(kModelPath, ctx_path, cache_dir));
    ASSERT_TRUE(std::filesystem::is_regular_file(ctx_path));

    // First session: EP creates the engine and writes the cache on destruction.
    {
        Ort::Session s = make_session_with_cache(ctx_path, cache_dir);
        EXPECT_EQ(s.GetInputCount(),  1u);
        EXPECT_EQ(s.GetOutputCount(), 1u);
        ASSERT_NO_THROW(run_with_cpu_bindings(s, 1));
    } // IExecutionContextDeleter fires here ? cache serialized

    EXPECT_TRUE(dir_has_files(cache_dir))
        << "Expected runtime cache files in: " << cache_dir;

    // Second session: EP should deserialize from the special-chars cache path.
    ASSERT_NO_THROW({
        Ort::Session s = make_session_with_cache(ctx_path, cache_dir);
        EXPECT_EQ(s.GetInputCount(),  1u);
        EXPECT_EQ(s.GetOutputCount(), 1u);
    });
}

INSTANTIATE_TEST_SUITE_P(
    CP1252Mappable,
    TensorRTRTXEpTest_PathTestP,
    ::testing::ValuesIn(cp1252_mappable_cases()),
    [](const ::testing::TestParamInfo<SpecialCharsCase>& info) {
        return std::string(info.param.label);
    });

// ===========================================================================
// Buffer-load coverage: exercises the SAME EP code paths as the path-load
// tests above, but reads the ctx-model bytes into memory via wide-aware
// std::ifstream and constructs Ort::Session(env, bytes, size, opts). This
// bypasses the Microsoft framework throw at model_metadef_id_generator.cc:43
// (which fires for any character not representable in the system ANSI code
// page when the session is created from a wchar_t* path).
//
// Result: the EP plugin's own load code is verified across CP1252-mappable
// Latin, non-CP1252 Latin, CJK, Cyrillic, emoji, and mixed-script paths.
// ===========================================================================

namespace
{

// Wide-aware file read -- std::ifstream(std::filesystem::path) preserves
// non-ACP chars in the underlying handle.
inline std::vector<uint8_t> read_all_bytes(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("read_all_bytes: failed to open file");
    return { std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>() };
}

// Buffer-load counterpart of make_session_with_cache. Reads ctx bytes via
// wide I/O (no narrow conversion of the model path), then opens the session
// from those bytes.
Ort::Session make_session_from_buffer_with_cache(
    const std::filesystem::path& ctx_path,
    const std::filesystem::path& cache_dir)
{
    auto bytes = read_all_bytes(ctx_path);
    Ort::SessionOptions so;
    Ort::KeyValuePairs  ep_opts;
    ep_opts.Add("enable_cuda_graph", "1");
    // UTF-8 narrow for the EP option (see compile_with_cache for why).
    ep_opts.Add("nv_runtime_cache_path", reinterpret_cast<const char*>(cache_dir.u8string().c_str()));
    so.AppendExecutionProvider_V2(*ort_env, get_trt_rtx_devices(*ort_env), ep_opts);
    return Ort::Session(*ort_env, bytes.data(), bytes.size(), so);
}

} // anonymous namespace

class TensorRTRTXEpTest_BufferLoadPathTest
    : public CompileModelTest,
      public ::testing::WithParamInterface<SpecialCharsCase>
{};

// -- Buffer-load: nv_runtime_cache_path with full character-class coverage ---

TEST_P(TensorRTRTXEpTest_BufferLoadPathTest, RuntimeCacheBufferLoad)
{
    const auto base      = special_chars_base_dir(GetParam().subdir_name);
    const auto cache_dir = base / "rt_cache_buf";
    const auto ctx_path  = base / "rt_cache_buf_ctx" / "context.onnx";

    // Clear any artifacts from prior runs so dir_has_files(cache_dir) below
    // reflects the current run's cache write, not stale files.
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);
    std::filesystem::remove_all(ctx_path.parent_path(), ec);

    std::filesystem::create_directories(ctx_path.parent_path());

    // Compile step: input model is at kModelPath (ASCII), so the framework's
    // model-path stringification (GenerateId:43) sees only ASCII bytes and
    // does not throw even when the OUTPUT path or cache_dir is non-CP1252.
    ASSERT_NO_THROW(compile_with_cache(kModelPath, ctx_path, cache_dir));
    ASSERT_TRUE(std::filesystem::is_regular_file(ctx_path));

    // First session via buffer-load -- avoids the framework GenerateId:43
    // throw entirely (no wchar_t* path enters the session ctor).
    {
        Ort::Session s = make_session_from_buffer_with_cache(ctx_path, cache_dir);
        EXPECT_EQ(s.GetInputCount(),  1u);
        EXPECT_EQ(s.GetOutputCount(), 1u);
        ASSERT_NO_THROW(run_with_cpu_bindings(s, 1));
    } // IExecutionContextDeleter fires here ? cache serialized

    EXPECT_TRUE(dir_has_files(cache_dir))
        << "Expected runtime cache files in: " << cache_dir;

    // Second session: EP should deserialize from the special-chars cache
    // path (cache_dir is UTF-8-narrow on the EP option boundary).
    ASSERT_NO_THROW({
        Ort::Session s = make_session_from_buffer_with_cache(ctx_path, cache_dir);
        EXPECT_EQ(s.GetInputCount(),  1u);
        EXPECT_EQ(s.GetOutputCount(), 1u);
    });
}

INSTANTIATE_TEST_SUITE_P(
    AllCharClasses,
    TensorRTRTXEpTest_BufferLoadPathTest,
    ::testing::ValuesIn(all_special_chars_cases()),
    [](const ::testing::TestParamInfo<SpecialCharsCase>& info) {
        return std::string(info.param.label);
    });

