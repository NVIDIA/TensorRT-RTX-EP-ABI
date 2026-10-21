// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <NvOnnxParser.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include "weightless_refit.h"
#include <gtest/gtest.h>

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{
// Exercise the persisted table through the public EP API, after the compile parser has been destroyed.
class ParseTimeRefitTest : public ::testing::TestWithParam<bool>
{
protected:
    void SetUp() override
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        test_dir_ = std::filesystem::temp_directory_path() / ("trt_parse_refit_" + std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directory(test_dir_));
        const auto source_path = test_dir_ / "source.onnx";
        context_path_ = test_dir_ / "context.onnx";
        model_builder::CreateInitializerMatMulModel(source_path.string());
        {
            Ort::SessionOptions options;
            ASSERT_NO_FATAL_FAILURE(AppendProvider(options));
            Ort::ModelCompilationOptions compilation(*ort_env, options);
            compilation.SetEpContextEmbedMode(GetParam() ? 1 : 0);
            compilation.SetInputModelPath(toOrtString(source_path.string()).c_str());
            compilation.SetOutputModelPath(toOrtString(context_path_.string()).c_str());
            const auto status = Ort::CompileModel(*ort_env, compilation);
            ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
        }
        std::ifstream input(context_path_, std::ios::binary);
        ASSERT_TRUE(context_.ParseFromIstream(&input));
        auto* table = FindTable();
        ASSERT_NE(table, nullptr) << "Compile must persist a table, not silently use the legacy refit path";
        ASSERT_TRUE(trt_rtx_ep::DeserializeWeightlessRefitTable(table->s(), records_));
        ASSERT_EQ(records_.size(), 1U) << "The MatMul model has exactly one named weight";
        ASSERT_EQ(records_[0].source_onnx_names, std::vector<std::string>{"W"});
        // Reload must be self-contained: legacy fallback cannot find the original model.
        ASSERT_TRUE(std::filesystem::remove(source_path));
    }

    void TearDown() override
    {
        if (!test_dir_.empty())
        {
            std::error_code error;
            std::filesystem::remove_all(test_dir_, error);
            EXPECT_FALSE(error) << error.message();
        }
    }

    void AppendProvider(Ort::SessionOptions& options)
    {
        auto devices = get_trt_rtx_devices(*ort_env);
        ASSERT_FALSE(devices.empty()) << "A working TRT RTX EP device is required";
        Ort::KeyValuePairs values;
        values.Add("nv_weight_stripped_engine_enable_experimental", "1");
        options.AppendExecutionProvider_V2(*ort_env, devices, values);
    }

    onnx::AttributeProto* FindTable()
    {
        for (auto& node : *context_.mutable_graph()->mutable_node())
        {
            if (node.op_type() == "EPContext")
            {
                for (auto& attribute : *node.mutable_attribute())
                {
                    if (attribute.name() == trt_rtx_ep::EP_REFIT_TABLE)
                    {
                        return &attribute;
                    }
                }
            }
        }
        return nullptr;
    }

    void SaveContext()
    {
        auto* table = FindTable();
        ASSERT_NE(table, nullptr);
        auto blob = trt_rtx_ep::SerializeWeightlessRefitTable(records_);
        ASSERT_FALSE(blob.empty());
        table->set_s(blob);
        std::ofstream output(context_path_, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(context_.SerializeToOstream(&output));
    }

    void ExpectOutput(const std::array<float, 3>& expected)
    {
        Ort::SessionOptions options;
        ASSERT_NO_FATAL_FAILURE(AppendProvider(options));
        Ort::Session session(*ort_env, toOrtString(context_path_.string()).c_str(), options);
        std::array<float, 4> x{1.0F, 2.0F, 3.0F, 4.0F};
        const std::array<int64_t, 2> shape{1, 4};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, x.data(), x.size(), shape.data(), shape.size());
        const char* inputs[] = {"X"};
        const char* outputs[] = {"Y"};
        auto result = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
        ASSERT_EQ(result.size(), 1U);
        ASSERT_EQ(result[0].GetTensorTypeAndShapeInfo().GetShape(), (std::vector<int64_t>{1, 3}));
        const auto* actual = result[0].GetTensorData<float>();
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 1e-3F) << "Output " << i;
        }
    }

    void ExpectLoadFailure(const std::string& message)
    {
        Ort::SessionOptions options;
        ASSERT_NO_FATAL_FAILURE(AppendProvider(options));
        try
        {
            Ort::Session session(*ort_env, toOrtString(context_path_.string()).c_str(), options);
            FAIL() << "A corrupted refit table must fail before inference";
        }
        catch (const Ort::Exception& error)
        {
            EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
        }
    }

    std::filesystem::path test_dir_;
    std::filesystem::path context_path_;
    onnx::ModelProto context_;
    std::vector<trt_rtx_ep::WeightlessRefitRecord> records_;
};

TEST_P(ParseTimeRefitTest, PersistsMetadataAndReloadsWithoutOriginalModel)
{
    const auto& record = records_[0];
    EXPECT_FALSE(record.trt_name.empty());
    EXPECT_EQ(record.kind, static_cast<int32_t>(nvonnxparser::RefitTransformKind::kIDENTITY));
    EXPECT_EQ(record.onnx_dtype, onnx::TensorProto_DataType_FLOAT);
    EXPECT_EQ(record.trt_dtype, static_cast<int32_t>(nvinfer1::DataType::kFLOAT));
    EXPECT_EQ(record.count, 12);
    EXPECT_TRUE(record.fixed_data.empty());
    ASSERT_NO_FATAL_FAILURE(ExpectOutput({70.0F, 80.0F, 90.0F}));
}

TEST_P(ParseTimeRefitTest, ReplaysChangedInitializerInsteadOfBuildTimeWeights)
{
    bool changed = false;
    for (auto& initializer : *context_.mutable_graph()->mutable_initializer())
    {
        if (initializer.name() == "W")
        {
            const std::array<float, 12> replacement{2, -1, 3, 2, -1, 3, 2, -1, 3, 2, -1, 3};
            initializer.clear_float_data();
            initializer.clear_external_data();
            initializer.set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
            initializer.set_raw_data(replacement.data(), replacement.size() * sizeof(float));
            changed = true;
        }
    }
    ASSERT_TRUE(changed) << "The captured table's source must be retained in EPContext";
    ASSERT_NO_FATAL_FAILURE(SaveContext());
    ASSERT_NO_FATAL_FAILURE(ExpectOutput({20.0F, -10.0F, 30.0F}));
}

TEST_P(ParseTimeRefitTest, IgnoresSurplusRecordBeforeResolvingItsSource)
{
    auto surplus = records_[0];
    surplus.trt_name += "_folded_away";
    surplus.source_onnx_names = {"source_not_retained"};
    surplus.kind = 999;
    surplus.count = -1;
    records_.insert(records_.begin(), surplus);
    ASSERT_NO_FATAL_FAILURE(SaveContext());
    ASSERT_NO_FATAL_FAILURE(ExpectOutput({70.0F, 80.0F, 90.0F}));
}

TEST_P(ParseTimeRefitTest, RejectsTableThatDoesNotCoverEngineWeights)
{
    // A nonempty but unrelated table must not leave the stripped engine's real weight unset.
    records_[0].trt_name += "_not_in_engine";
    ASSERT_NO_FATAL_FAILURE(SaveContext());
    ASSERT_NO_FATAL_FAILURE(ExpectLoadFailure("does not cover 1 of 1 refittable engine weight(s)"));
}

TEST_P(ParseTimeRefitTest, RejectsMissingSourceForRequiredWeight)
{
    records_[0].source_onnx_names = {"missing_initializer"};
    ASSERT_NO_FATAL_FAILURE(SaveContext());
    ASSERT_NO_FATAL_FAILURE(ExpectLoadFailure("was not supplied to the EPContext node"));
}

TEST_P(ParseTimeRefitTest, RejectsUnknownTransformForRequiredWeight)
{
    records_[0].kind = 999;
    ASSERT_NO_FATAL_FAILURE(SaveContext());
    ASSERT_NO_FATAL_FAILURE(ExpectLoadFailure("unknown RefitTransformKind"));
}

INSTANTIATE_TEST_SUITE_P(EngineStorage, ParseTimeRefitTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info)
                         {
                             return info.param ? "Embedded" : "External";
                         });
}  // namespace
