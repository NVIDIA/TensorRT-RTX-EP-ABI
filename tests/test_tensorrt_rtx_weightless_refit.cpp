// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the weightless-refit table format + source-name collection, with focus on the
// Constant/fixed_data kinds (kCONSTANT_NODE / kCONSTANT_OF_SHAPE). These kinds are correct-by-construction
// but not reachable via crafted ONNX through the ORT EP pipeline (ORT normalizes Constant nodes into named
// initializers, and TRT bakes uniform ConstantOfShape into the engine), so an end-to-end model can't
// exercise them. These tests drive the deterministic parts directly: capture (record construction),
// format round-trip (serialize -> deserialize), and source-name exclusion. The remaining stage (live TRT
// replay) shares the exact named-path machinery verified bit-exact elsewhere, differing only by
// `weights.values = record.fixed_data.data()`.

#include <NvOnnxParser.h>  // nvonnxparser::RefitTransformKind (compile-time enum constants only)

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "weightless_refit.h"
#include <gtest/gtest.h>

// For the end-to-end test below (compile weight-stripped -> reload weightless -> run).
#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"

using trt_rtx_ep::CollectWeightlessRefitSourceNames;
using trt_rtx_ep::DeserializeWeightlessRefitTable;
using trt_rtx_ep::SerializeWeightlessRefitTable;
using trt_rtx_ep::WeightlessRefitRecord;
using trt_rtx_ep::detail::TryGetWeightlessBufferByteSize;

namespace
{
constexpr int32_t kIdentity = static_cast<int32_t>(nvonnxparser::RefitTransformKind::kIDENTITY);               // 0
constexpr int32_t kConstNode = static_cast<int32_t>(nvonnxparser::RefitTransformKind::kCONSTANT_NODE);         // 4
constexpr int32_t kConstOfShape = static_cast<int32_t>(nvonnxparser::RefitTransformKind::kCONSTANT_OF_SHAPE);  // 5

WeightlessRefitRecord MakeRecord(const std::string& name, int32_t kind, std::vector<std::string> srcNames,
                                 std::vector<uint8_t> fixed_data)
{
    WeightlessRefitRecord r;
    r.trt_name = name;
    r.kind = kind;
    r.onnx_dtype = 1;  // FLOAT
    r.trt_dtype = 0;
    r.count = static_cast<int64_t>(fixed_data.size());
    r.epsilon = 1e-5f;
    r.source_onnx_names = std::move(srcNames);
    r.fixed_data = std::move(fixed_data);
    return r;
}

void ExpectRecordEq(const WeightlessRefitRecord& a, const WeightlessRefitRecord& b)
{
    EXPECT_EQ(a.trt_name, b.trt_name);
    EXPECT_EQ(a.kind, b.kind);
    EXPECT_EQ(a.onnx_dtype, b.onnx_dtype);
    EXPECT_EQ(a.trt_dtype, b.trt_dtype);
    EXPECT_EQ(a.count, b.count);
    EXPECT_FLOAT_EQ(a.epsilon, b.epsilon);
    EXPECT_EQ(a.source_onnx_names, b.source_onnx_names);
    EXPECT_EQ(a.fixed_data, b.fixed_data);
}
}  // namespace

// The Constant kinds carry their weight bytes verbatim in fixed_data (no source initializer). Verify the
// format round-trips that data losslessly.
TEST(WeightlessRefitTable, RoundTripConstantKinds)
{
    std::vector<WeightlessRefitRecord> in = {
        MakeRecord("const_node_w", kConstNode, /*srcNames*/ {}, /*fixed_data*/ {1, 2, 3, 4, 5, 6, 7, 8}),
        MakeRecord("const_of_shape_w", kConstOfShape, {}, {0xAA, 0xBB, 0xCC, 0xDD}),
    };
    std::string blob = SerializeWeightlessRefitTable(in);
    ASSERT_FALSE(blob.empty());

    std::vector<WeightlessRefitRecord> out;
    ASSERT_TRUE(DeserializeWeightlessRefitTable(blob, out));
    ASSERT_EQ(out.size(), in.size());
    ExpectRecordEq(in[0], out[0]);
    ExpectRecordEq(in[1], out[1]);
    // fixed_data must survive byte-for-byte -- this is the constant weight itself.
    EXPECT_EQ(out[0].fixed_data, (std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}));
}

// The format stores `kind` as a plain uint32 and is kind-agnostic: an UNKNOWN kind (future SDK) must
// round-trip unchanged so the *consumer* can hard-fail on it (see the load-time default: guard). This is
// the format-level complement to the runtime "unknown RefitTransformKind" hard-fail.
TEST(WeightlessRefitTable, RoundTripMixedAndUnknownKind)
{
    const int32_t kUnknownFutureKind = 999;
    std::vector<WeightlessRefitRecord> in = {
        MakeRecord("named_w", kIdentity, {"src_init"}, {}),
        MakeRecord("future_w", kUnknownFutureKind, {"src_future"}, {9, 9, 9}),
    };
    std::string blob = SerializeWeightlessRefitTable(in);
    ASSERT_FALSE(blob.empty());

    std::vector<WeightlessRefitRecord> out;
    ASSERT_TRUE(DeserializeWeightlessRefitTable(blob, out));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].kind, kIdentity);
    EXPECT_EQ(out[1].kind, kUnknownFutureKind);  // preserved verbatim, not clamped/dropped
}

// Constant kinds embed their data in fixed_data and have NO source initializer to keep; named kinds must
// have their source names collected (so those initializers are kept as EPContext node inputs). Applying
// engine B's names to engine A would be corruption; excluding constants prevents a phantom missing input.
TEST(WeightlessRefitSourceNames, ExcludesConstantKindsIncludesNamed)
{
    std::vector<WeightlessRefitRecord> records = {
        MakeRecord("named_a", kIdentity, {"w_named_a"}, {}),
        MakeRecord("const_b", kConstNode, {"w_should_be_ignored"}, {1, 2, 3}),
        MakeRecord("cos_c", kConstOfShape, {"w_also_ignored"}, {4, 5}),
        MakeRecord("named_d", kIdentity, {"w_named_d"}, {}),
    };
    std::vector<std::string> names = CollectWeightlessRefitSourceNames(records);

    // Named kinds' source names are present; constant kinds' are excluded.
    EXPECT_NE(std::find(names.begin(), names.end(), "w_named_a"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "w_named_d"), names.end());
    EXPECT_EQ(std::find(names.begin(), names.end(), "w_should_be_ignored"), names.end());
    EXPECT_EQ(std::find(names.begin(), names.end(), "w_also_ignored"), names.end());
    EXPECT_EQ(names.size(), 2u);
}

// Robustness: a truncated/garbage blob must be rejected (return false), never partially parsed.
TEST(WeightlessRefitTable, RejectsMalformedBlob)
{
    std::vector<WeightlessRefitRecord> out;
    EXPECT_FALSE(DeserializeWeightlessRefitTable("not-a-valid-blob", out));
    EXPECT_FALSE(DeserializeWeightlessRefitTable(std::string(3, '\0'), out));  // shorter than header
}

//!
//! \brief Verifies oversized weight counts are rejected before byte arithmetic or allocation.
//!
TEST(WeightlessRefitValidation, RejectsOverflowingBufferSizes)
{
    size_t byte_size = 0;
    EXPECT_TRUE(TryGetWeightlessBufferByteSize(4, sizeof(double), byte_size));
    EXPECT_EQ(byte_size, 4 * sizeof(double));

    EXPECT_TRUE(TryGetWeightlessBufferByteSize(0, sizeof(double), byte_size));
    EXPECT_EQ(byte_size, 0u);

    EXPECT_FALSE(TryGetWeightlessBufferByteSize(-1, sizeof(double), byte_size));
    EXPECT_FALSE(TryGetWeightlessBufferByteSize(1, 0, byte_size));

    // The previous kDOUBLE_TO_FLOAT expression wrapped this count * sizeof(double) to zero on 64-bit hosts.
    EXPECT_FALSE(TryGetWeightlessBufferByteSize(int64_t{1} << 61, sizeof(double), byte_size));
}

// =============================================================================
// End-to-end weightless-refit test (GPU / TRT-RTX). Mirrors sample_weightless in-tree:
//   1) build a small MatMul model with a named initializer weight W,
//   2) compile it WEIGHT-STRIPPED (nv_weight_stripped_engine_enable_experimental=1) to an EPContext model,
//   3) reload that EPContext in a FRESH session (no original model), which refits from the kept
//      weight source at load time, then Run() it,
//   4) assert the output equals the analytic MatMul X*W -- a wrong/misapplied refit changes the
//      numbers, so this exercises the actual refit-and-run path, not just "it loaded".
// =============================================================================

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{
// Local copy of the test-suite's EP-append helper (the original is file-static in another TU).
void AppendTrtRtxEpWithOptions(Ort::SessionOptions& so, const std::unordered_map<std::string, std::string>& options)
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv;
    for (const auto& [k, v] : options)
    {
        kv.Add(k.c_str(), v.c_str());
    }
    so.AppendExecutionProvider_V2(*ort_env, devices, kv);
}
}  // namespace

TEST(WeightlessRefitE2E, StrippedCompileReloadRunMatchesMatMul)
{
    const std::string model_name = "weightless_e2e_matmul.onnx";
    const std::string model_name_ctx = "weightless_e2e_matmul_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    // MatMul: X[1,4] * W[4,3] -> Y[1,3], with W = {1..12} row-major (see CreateInitializerMatMulModel).
    model_builder::CreateInitializerMatMulModel(model_name);

    // 1) Weight-STRIPPED compile -> self-contained EPContext (embed_mode=1).
    {
        Ort::SessionOptions so;
        AppendTrtRtxEpWithOptions(so, {{"nv_weight_stripped_engine_enable_experimental", "1"}});
        Ort::ModelCompilationOptions co(*ort_env, so);
        co.SetEpContextEmbedMode(1);
        co.SetInputModelPath(toOrtString(model_name).c_str());
        co.SetOutputModelPath(toOrtString(model_name_ctx).c_str());
        ASSERT_TRUE(Ort::CompileModel(*ort_env, co).IsOK()) << "weight-stripped CompileModel failed";
    }

    // 2) Reload the EPContext in a FRESH session with NO original model registered; refit happens at load.
    Ort::SessionOptions so;
    AppendTrtRtxEpWithOptions(so, {{"nv_weight_stripped_engine_enable_experimental", "1"}});
    Ort::Session session(*ort_env, toOrtString(model_name_ctx).c_str(), so);

    // 3) Run with a known input.
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> x_shape = {1, 4};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), x_shape.data(), x_shape.size());
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    Ort::RunOptions run_options;
    std::vector<Ort::Value> outputs;
    ASSERT_NO_THROW(outputs = session.Run(run_options, input_names, &x_tensor, 1, output_names, 1));
    ASSERT_EQ(outputs.size(), 1u);

    // 4) Assert Y == X * W (a wrong refit would change these). W columns dotted with X={1,2,3,4}:
    //    Y0=1*1+2*4+3*7+4*10=70, Y1=1*2+2*5+3*8+4*11=80, Y2=1*3+2*6+3*9+4*12=90.
    const float* y = outputs[0].GetTensorData<float>();
    EXPECT_NEAR(y[0], 70.0f, 1e-2f);
    EXPECT_NEAR(y[1], 80.0f, 1e-2f);
    EXPECT_NEAR(y[2], 90.0f, 1e-2f);
}
