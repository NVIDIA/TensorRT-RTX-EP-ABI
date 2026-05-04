// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from onnxruntime/test/providers/nv_tensorrt_rtx/nv_ep_context_test.cc
// Uses only public ORT SDK APIs.

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <cmath>
#include <string>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

// Helper: append TRT RTX EP to session options.
static void AppendTrtRtxEp(
    Ort::SessionOptions& so,
    const std::unordered_map<std::string, std::string>& options = {}) {
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv_options;
    for (auto& [k, v] : options) {
        kv_options.Add(k.c_str(), v.c_str());
    }
    so.AppendExecutionProvider_V2(*ort_env, devices, kv_options);
}

// Helper: read a file into a byte vector.
static std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    file.seekg(0, std::ios::end);
    if (!file.good()) {
        throw std::runtime_error("Failed to seek in file: " + path);
    }
    auto size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to determine size of file: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!file.read(buf.data(), size)) {
        throw std::runtime_error("Failed to read " + std::to_string(size)
                                 + " bytes from file: " + path);
    }
    return buf;
}

static std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open text file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

static onnx::ModelProto readOnnxModel(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open ONNX model: " + path.string());
    }

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&file)) {
        throw std::runtime_error("Failed to parse ONNX model: " + path.string());
    }
    return model;
}

static size_t countNodesByOpType(
    const onnx::GraphProto& graph,
    const std::string& op_type,
    const std::string& domain = "") {
    size_t count = 0;
    for (const auto& node : graph.node()) {
        if (node.op_type() == op_type &&
            (domain.empty() || node.domain() == domain)) {
            ++count;
        }

        for (const auto& attr : node.attribute()) {
            if (attr.type() == onnx::AttributeProto_AttributeType_GRAPH) {
                count += countNodesByOpType(attr.g(), op_type, domain);
            } else if (attr.type() == onnx::AttributeProto_AttributeType_GRAPHS) {
                for (const auto& nested_graph : attr.graphs()) {
                    count += countNodesByOpType(nested_graph, op_type, domain);
                }
            }
        }
    }

    return count;
}

static std::vector<float> ComputeFastGeluReference(
    const std::vector<float>& input_data) {
    std::vector<float> output;
    output.reserve(input_data.size());

    for (float x : input_data) {
        const float y =
            x * (0.5f + 0.5f * std::tanh(x * (0.035677408136300125f * x * x +
                                              0.7978845608028654f)));
        output.push_back(y);
    }

    return output;
}

static std::vector<float> ComputeExpectedLoweredAsymmetricDqMatMulFastGeluOutput(
    const std::vector<float>& x_data) {
    constexpr int kRows = 2;
    constexpr int kInner = 3;
    constexpr int kCols = 2;
    const std::vector<int8_t> q_weights = {16, -8, 5, 12, -3, 9};
    constexpr int8_t kZeroPoint = 3;
    constexpr float kScale = 0.25f;

    std::vector<float> dequantized_weights;
    dequantized_weights.reserve(q_weights.size());
    for (const int8_t q : q_weights) {
        dequantized_weights.push_back(
            (static_cast<float>(q) - static_cast<float>(kZeroPoint)) * kScale);
    }

    std::vector<float> matmul_output(kRows * kCols, 0.0f);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < kInner; ++inner) {
                acc += x_data[row * kInner + inner] *
                       dequantized_weights[inner * kCols + col];
            }
            matmul_output[row * kCols + col] = acc;
        }
    }

    return ComputeFastGeluReference(matmul_output);
}

static std::vector<float> ComputeExpectedLoweredAsymmetricQdqMatMulFastGeluOutput(
    const std::vector<float>& x_data) {
    constexpr int kRows = 2;
    constexpr int kInner = 3;
    constexpr int kCols = 2;
    constexpr float kScale = 0.25f;
    constexpr int8_t kZeroPoint = 5;
    constexpr int32_t kQMin = -128;
    constexpr int32_t kQMax = 127;
    const std::vector<float> weights = {
        1.25f, -0.75f,
        0.50f,  2.00f,
       -1.50f,  0.25f};

    std::vector<float> dequantized_x;
    dequantized_x.reserve(x_data.size());
    for (float x : x_data) {
        const float shifted = std::nearbyint(x / kScale) +
                              static_cast<float>(kZeroPoint);
        const int32_t clamped = std::max(
            kQMin, std::min(kQMax, static_cast<int32_t>(shifted)));
        const int8_t quantized = static_cast<int8_t>(clamped);
        dequantized_x.push_back(
            (static_cast<float>(quantized) - static_cast<float>(kZeroPoint)) *
            kScale);
    }

    std::vector<float> matmul_output(kRows * kCols, 0.0f);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < kInner; ++inner) {
                acc += dequantized_x[row * kInner + inner] *
                       weights[inner * kCols + col];
            }
            matmul_output[row * kCols + col] = acc;
        }
    }

    return ComputeFastGeluReference(matmul_output);
}

// =============================================================================
// CompileApiTest — parameterized
// =============================================================================

struct CompileParam {
    bool embed_mode;
    bool bytestream_io;
    bool external_initializer_for_parser = false;

    std::string to_string() const {
        return "embed_mode_" + std::to_string(embed_mode)
             + "_bytestream_io_" + std::to_string(bytestream_io)
             + "_ext_init_" + std::to_string(external_initializer_for_parser);
    }
};

class CompileApiTest
    : public ::testing::TestWithParam<CompileParam> {
 protected:
    void SetUp() override {
        ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty())
            << "No TRT RTX EP devices found.";
    }
};

static void SmallModelTest(CompileParam test_param, bool fully_supported_model) {
    std::string test_name = test_param.to_string();
    if (!fully_supported_model) test_name += "_fast_gelu";

    const std::string model_name =
        "nv_execution_provider_compile_" + test_name + ".onnx";
    const std::string model_name_ctx =
        "nv_execution_provider_compile_" + test_name + "_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    model_builder::CreateBaseModel(
        model_name, "test", {1, 3, 2}, !fully_supported_model);

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options,
                   {{"nv_use_external_data_initializer",
                     std::to_string(test_param.external_initializer_for_parser)}});

    Ort::ModelCompilationOptions compile_opts(*ort_env, session_options);
    compile_opts.SetEpContextEmbedMode(test_param.embed_mode);

    void* output_context = nullptr;
    size_t output_context_size = 0;
    std::vector<char> input_onnx;

    if (test_param.bytestream_io) {
        input_onnx = readBinaryFile(model_name);
        compile_opts.SetInputModelFromBuffer(input_onnx.data(), input_onnx.size());
        compile_opts.SetOutputModelBuffer(
            Ort::AllocatorWithDefaultOptions(), &output_context, &output_context_size);
    } else {
        compile_opts.SetInputModelPath(toOrtString(model_name).c_str());
        compile_opts.SetOutputModelPath(toOrtString(model_name_ctx).c_str());
    }

    // AOT compile
    ASSERT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());

    // JIT: load compiled model and run inference
    Ort::Session session{nullptr};
    if (test_param.bytestream_io) {
        session = Ort::Session(*ort_env, output_context, output_context_size,
                               session_options);
        Ort::AllocatorWithDefaultOptions().Free(output_context);
    } else {
        session = Ort::Session(*ort_env, toOrtString(model_name_ctx).c_str(),
                               session_options);
    }

    auto io_binding = generate_io_binding(session);
    Ort::RunOptions run_options;
    session.Run(run_options, io_binding);
}

TEST_P(CompileApiTest, SmallModel) {
    SmallModelTest(GetParam(), /*fully_supported_model=*/true);
}

TEST_P(CompileApiTest, SmallSplitModel) {
    SmallModelTest(GetParam(), /*fully_supported_model=*/false);
}

// Large model compilation with external-data I/O. embed_mode=1 is skipped
// because an embedded context for a large model would exceed the 2 GB proto
// limit.
TEST_P(CompileApiTest, LargeModel) {
    const CompileParam test_param = GetParam();
    if (test_param.embed_mode) {
        GTEST_SKIP() << "embed_mode=1 would exceed 2GB proto limit for large model.";
    }

    const std::string test_name = test_param.to_string();
    const std::string model_name =
        "nv_execution_provider_compile_large_" + test_name + ".onnx";
    const std::string external_data_name =
        "nv_execution_provider_compile_large_" + test_name + ".onnx_data";
    const std::string model_name_ctx =
        "nv_execution_provider_compile_large_" + test_name + "_ctx.onnx";
    const std::string model_name_ctx_data =
        "nv_execution_provider_compile_large_" + test_name + "_ctx.onnx_data";
    clearFileIfExists(model_name_ctx);
    clearFileIfExists(model_name_ctx_data);

    // Reuse the model across test iterations for speed.
    if (!std::filesystem::exists(model_name) ||
        !std::filesystem::exists(external_data_name)) {
        model_builder::CreateLargeModel(model_name, external_data_name);
    }

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options,
                   {{"nv_use_external_data_initializer",
                     std::to_string(test_param.bytestream_io ||
                                    test_param.external_initializer_for_parser)}});

    Ort::ModelCompilationOptions compile_opts(*ort_env, session_options);
    compile_opts.SetEpContextEmbedMode(test_param.embed_mode);

    void* output_context = nullptr;
    size_t output_context_size = 0;
    std::vector<char> input_onnx;
    std::vector<char> input_data;
    std::vector<std::basic_string<ORTCHAR_T>> ext_file_names;
    std::vector<char*> ext_file_buffers;
    std::vector<size_t> ext_lengths;

    if (test_param.bytestream_io) {
        input_onnx = readBinaryFile(model_name);
        input_data = readBinaryFile(external_data_name);
        // "location" stored in the ONNX model is the filename only (see
        // CreateLargeModel). Match that here so ORT can resolve the mapping.
        const auto filename = std::filesystem::path(external_data_name).filename();
        ext_file_names   = {toOrtString(filename)};
        ext_file_buffers = {input_data.data()};
        ext_lengths      = {input_data.size()};
        session_options.AddExternalInitializersFromFilesInMemory(
            ext_file_names, ext_file_buffers, ext_lengths);

        compile_opts.SetInputModelFromBuffer(input_onnx.data(), input_onnx.size());
        compile_opts.SetOutputModelBuffer(
            Ort::AllocatorWithDefaultOptions(), &output_context, &output_context_size);
    } else {
        compile_opts.SetInputModelPath(toOrtString(model_name).c_str());
        compile_opts.SetOutputModelPath(toOrtString(model_name_ctx).c_str());
        compile_opts.SetOutputModelExternalInitializersFile(
            toOrtString(model_name_ctx_data).c_str(), 1024);
    }

    ASSERT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());

    // JIT: load compiled model and run inference
    std::unique_ptr<Ort::Session> session;
    if (test_param.bytestream_io) {
        session = std::make_unique<Ort::Session>(
            *ort_env, output_context, output_context_size, session_options);
    } else {
        session = std::make_unique<Ort::Session>(
            *ort_env, toOrtString(model_name_ctx).c_str(), session_options);
    }

    auto io_binding = generate_io_binding(*session);
    Ort::RunOptions run_options;
    session->Run(run_options, io_binding);

    if (output_context != nullptr) {
        Ort::AllocatorWithDefaultOptions().Free(output_context);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TensorRTRTXEpTest_EpContext, CompileApiTest,
    ::testing::Values(
        CompileParam{true, false},
        CompileParam{false, false},
        CompileParam{true, true},
        CompileParam{false, true},
        CompileParam{true, true, true},
        CompileParam{true, false, true}),
    [](const ::testing::TestParamInfo<CompileApiTest::ParamType>& info) {
        return info.param.to_string();
    });

// Validates the runtime split-graph path for lowered asymmetric DQ:
//   * the constant-weight DequantizeLinear(zp != 0) + MatMul region stays on TRT-RTX
//   * FastGelu is intentionally excluded from TRT-RTX and executes on CPU
//   * the final output still matches a CPU-side golden reference for the full graph
TEST(TensorRTRTXEpTest_EpContext, LoweredAsymmetricDqMatMulPreservesSplitGraph) {
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty())
        << "No TRT RTX EP devices found.";

    const std::string model_path =
        "ep_context_lowered_asymmetric_dq_fast_gelu.onnx";
    clearFileIfExists(model_path);

    model_builder::CreateAsymmetricDqMatMulFastGeluModel(
        model_path, "LoweredAsymmetricDqFastGeluGraph");

    Ort::SessionOptions session_options;
    // FastGelu is intentionally kept on CPU so this test can validate a stable
    // mixed-partition graph: the asymmetric DQ+MatMul region on TRT-RTX and
    // FastGelu on CPU.
    AppendTrtRtxEp(session_options, {{"nv_op_types_to_exclude", "FastGelu"}});
    const std::filesystem::path profile_prefix =
        "ep_context_lowered_asymmetric_dq_fast_gelu_profile";
    session_options.EnableProfiling(toOrtString(profile_prefix).c_str());

    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);
    Ort::RunOptions run_options;

    const std::vector<int64_t> input_shape = {2, 3};
    const std::vector<float> input_data = {
        1.0f, 2.0f, -1.0f,
        0.5f, -0.25f, 3.0f};
    const std::vector<float> expected_output =
        ComputeExpectedLoweredAsymmetricDqMatMulFastGeluOutput(input_data);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), const_cast<float*>(input_data.data()), input_data.size(),
        input_shape.data(), input_shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"O"};

    std::vector<Ort::Value> outputs;
    ASSERT_NO_THROW(outputs = session.Run(run_options, input_names, &input_tensor,
                                          1, output_names, 1));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());

    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    EXPECT_EQ(output_info.GetElementCount(), expected_output.size());
    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5f)
            << "Mismatch at output index " << i;
    }

    auto profile_path_alloc = session.EndProfilingAllocated(allocator);
    ASSERT_NE(profile_path_alloc.get(), nullptr);

    const std::filesystem::path profile_path{profile_path_alloc.get()};
    ASSERT_TRUE(std::filesystem::is_regular_file(profile_path))
        << "Profiling output not found at: " << profile_path.string();

    const std::string profile = readTextFile(profile_path);

    EXPECT_NE(profile.find("FastGelu"), std::string::npos)
        << "Expected FastGelu to appear in ORT profiling output";
    EXPECT_NE(profile.find("CPUExecutionProvider"), std::string::npos)
        << "Expected CPUExecutionProvider to appear in ORT profiling output";
    EXPECT_NE(profile.find(kEpName), std::string::npos)
        << "Expected NvTensorRTRTXExecutionProvider to appear in ORT profiling output";
}

// Validates the runtime split-graph path for lowered asymmetric Q->DQ:
//   * the QuantizeLinear/DequantizeLinear + MatMul region stays on TRT-RTX
//   * FastGelu is intentionally excluded from TRT-RTX and executes on CPU
//   * the final output still matches a CPU-side golden reference for the full graph
TEST(TensorRTRTXEpTest_EpContext, LoweredAsymmetricQdqMatMulPreservesSplitGraph) {
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty())
        << "No TRT RTX EP devices found.";

    const std::string model_path =
        "ep_context_lowered_asymmetric_qdq_fast_gelu.onnx";
    clearFileIfExists(model_path);

    model_builder::CreateAsymmetricQdqMatMulFastGeluModel(
        model_path, "LoweredAsymmetricQdqFastGeluGraph");

    Ort::SessionOptions session_options;
    // FastGelu is intentionally kept on CPU so this test can validate a stable
    // mixed-partition graph: the asymmetric Q/DQ + MatMul region on TRT-RTX and
    // FastGelu on CPU.
    AppendTrtRtxEp(session_options, {{"nv_op_types_to_exclude", "FastGelu"}});
    const std::filesystem::path profile_prefix =
        "ep_context_lowered_asymmetric_qdq_fast_gelu_profile";
    session_options.EnableProfiling(toOrtString(profile_prefix).c_str());

    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);
    Ort::RunOptions run_options;

    const std::vector<int64_t> input_shape = {2, 3};
    const std::vector<float> input_data = {
        1.10f, -0.35f, 2.30f,
       -1.70f,  0.40f, 0.95f};
    const std::vector<float> expected_output =
        ComputeExpectedLoweredAsymmetricQdqMatMulFastGeluOutput(input_data);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), const_cast<float*>(input_data.data()),
        input_data.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"O"};

    std::vector<Ort::Value> outputs;
    ASSERT_NO_THROW(outputs = session.Run(run_options, input_names, &input_tensor,
                                          1, output_names, 1));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());

    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    EXPECT_EQ(output_info.GetElementCount(), expected_output.size());
    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5f)
            << "Mismatch at output index " << i;
    }

    auto profile_path_alloc = session.EndProfilingAllocated(allocator);
    ASSERT_NE(profile_path_alloc.get(), nullptr);

    const std::filesystem::path profile_path{profile_path_alloc.get()};
    ASSERT_TRUE(std::filesystem::is_regular_file(profile_path))
        << "Profiling output not found at: " << profile_path.string();

    const std::string profile = readTextFile(profile_path);

    EXPECT_NE(profile.find("FastGelu"), std::string::npos)
        << "Expected FastGelu to appear in ORT profiling output";
    EXPECT_NE(profile.find("CPUExecutionProvider"), std::string::npos)
        << "Expected CPUExecutionProvider to appear in ORT profiling output";
    EXPECT_NE(profile.find(kEpName), std::string::npos)
        << "Expected NvTensorRTRTXExecutionProvider to appear in ORT profiling output";
}

// Validates the AOT compile/load path for the lowered asymmetric DQ split graph:
//   * compiling with TRT-RTX produces a mixed model that preserves EPContext + FastGelu
//   * loading the compiled model keeps the TRT-RTX / CPU split intact at runtime
//   * the final output still matches the same CPU-side golden reference
TEST(TensorRTRTXEpTest_EpContext,
     CompileLoweredAsymmetricDqMatMulPreservesSplitGraphAndOutput) {
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty())
        << "No TRT RTX EP devices found.";

    const std::string model_path =
        "ep_context_compile_lowered_asymmetric_dq_fast_gelu.onnx";
    const std::string compiled_model_path =
        "ep_context_compile_lowered_asymmetric_dq_fast_gelu_ctx.onnx";
    clearFileIfExists(model_path);
    clearFileIfExists(compiled_model_path);

    model_builder::CreateAsymmetricDqMatMulFastGeluModel(
        model_path, "CompileLoweredAsymmetricDqFastGeluGraph");

    Ort::SessionOptions compile_session_options;
    AppendTrtRtxEp(compile_session_options,
                   {{"nv_op_types_to_exclude", "FastGelu"}});

    Ort::ModelCompilationOptions compile_opts(*ort_env, compile_session_options);
    compile_opts.SetEpContextEmbedMode(true);
    compile_opts.SetInputModelPath(toOrtString(model_path).c_str());
    compile_opts.SetOutputModelPath(toOrtString(compiled_model_path).c_str());

    ASSERT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());
    ASSERT_TRUE(std::filesystem::is_regular_file(compiled_model_path))
        << "Compiled model not found at: " << compiled_model_path;

    // The compiled ONNX should preserve the split shape we care about:
    // one or more EPContext nodes for TRT-RTX-owned regions and a standalone
    // FastGelu node left outside the TRT-RTX partition.
    const onnx::ModelProto compiled_model = readOnnxModel(compiled_model_path);
    EXPECT_GE(countNodesByOpType(compiled_model.graph(), "EPContext"), 1u)
        << "Expected compiled model to contain at least one EPContext node.";
    EXPECT_EQ(countNodesByOpType(compiled_model.graph(), "FastGelu", "com.microsoft"), 1u)
        << "Expected compiled model to preserve FastGelu outside the TRT-RTX partition.";

    Ort::SessionOptions load_session_options;
    AppendTrtRtxEp(load_session_options, {{"nv_op_types_to_exclude", "FastGelu"}});
    const std::filesystem::path profile_prefix =
        "ep_context_compile_lowered_asymmetric_dq_fast_gelu_profile";
    load_session_options.EnableProfiling(toOrtString(profile_prefix).c_str());

    Ort::Session session(*ort_env, toOrtString(compiled_model_path).c_str(),
                         load_session_options);
    Ort::RunOptions run_options;

    const std::vector<int64_t> input_shape = {2, 3};
    const std::vector<float> input_data = {
        1.0f, 2.0f, -1.0f,
        0.5f, -0.25f, 3.0f};
    const std::vector<float> expected_output =
        ComputeExpectedLoweredAsymmetricDqMatMulFastGeluOutput(input_data);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(), const_cast<float*>(input_data.data()), input_data.size(),
        input_shape.data(), input_shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"O"};

    std::vector<Ort::Value> outputs;
    ASSERT_NO_THROW(outputs = session.Run(run_options, input_names, &input_tensor,
                                          1, output_names, 1));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());

    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    EXPECT_EQ(output_info.GetElementCount(), expected_output.size());
    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected_output.size(); ++i) {
        EXPECT_NEAR(output_data[i], expected_output[i], 1e-5f)
            << "Mismatch at output index " << i;
    }

    auto profile_path_alloc = session.EndProfilingAllocated(allocator);
    ASSERT_NE(profile_path_alloc.get(), nullptr);

    const std::filesystem::path profile_path{profile_path_alloc.get()};
    ASSERT_TRUE(std::filesystem::is_regular_file(profile_path))
        << "Profiling output not found at: " << profile_path.string();

    const std::string profile = readTextFile(profile_path);
    EXPECT_NE(profile.find("FastGelu"), std::string::npos)
        << "Expected FastGelu to appear in ORT profiling output";
    EXPECT_NE(profile.find("CPUExecutionProvider"), std::string::npos)
        << "Expected CPUExecutionProvider to appear in ORT profiling output";
    EXPECT_NE(profile.find(kEpName), std::string::npos)
        << "Expected NvTensorRTRTXExecutionProvider to appear in ORT profiling output";
}

// =============================================================================
// EPContext node "source" attribute tests
//
// Verifies that the TRT RTX EP:
//   * skips EPContext nodes belonging to a foreign EP (e.g. OpenVINO),
//   * skips EPContext nodes belonging to the classic TensorRT EP,
//   * still claims EPContext nodes with no "source" attribute (backward compat).
// =============================================================================

TEST(TensorRTRTXEpTest_EpContext, EPContextNode_ForeignSourceSkipped) {
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty());
    const std::string model_path = "ep_context_foreign_source_nv.onnx";
    model_builder::CreateSyntheticEPContextModel(
        model_path, "OpenVINOExecutionProvider");

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options);

    try {
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(),
                             session_options);
        FAIL()
            << "Expected session creation to fail for EPContext node with foreign source";
    } catch (const Ort::Exception& e) {
        const std::string error_msg = e.what();
        EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos)
            << "Error should mention EPContext. Actual: " << error_msg;
    }

    std::filesystem::remove(model_path);
}

TEST(TensorRTRTXEpTest_EpContext, EPContextNode_ClassicTrtSourceSkipped) {
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty());
    const std::string model_path = "ep_context_classic_trt_source_nv.onnx";
    model_builder::CreateSyntheticEPContextModel(
        model_path, "TensorrtExecutionProvider");

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options);

    try {
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(),
                             session_options);
        FAIL()
            << "Expected session creation to fail for EPContext node with classic TRT source";
    } catch (const Ort::Exception& e) {
        const std::string error_msg = e.what();
        EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos)
            << "Error should mention EPContext. Actual: " << error_msg;
    }

    std::filesystem::remove(model_path);
}

TEST(TensorRTRTXEpTest_EpContext, EPContextNode_NoSourceAttribute_BackwardCompat) {
    ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty());
    const std::string model_path = "ep_context_no_source_nv.onnx";
    model_builder::CreateSyntheticEPContextModel(
        model_path, /*source_attr=*/"", /*include_source_attr=*/false);

    Ort::SessionOptions session_options;
    AppendTrtRtxEp(session_options);

    try {
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(),
                             session_options);
        // If session creation succeeds, backward compatibility is working.
    } catch (const Ort::Exception& e) {
        const std::string error_msg = e.what();
        // Failure must NOT be "not compatible with any execution provider",
        // which would indicate the node was never claimed by the EP.
        EXPECT_TRUE(error_msg.find("is not compatible with any execution provider")
                    == std::string::npos)
            << "Legacy EPContext node without source should still be claimed by EP. "
            << "Error: " << error_msg;
    }

    std::filesystem::remove(model_path);
}
