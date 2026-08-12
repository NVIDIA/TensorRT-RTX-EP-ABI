// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "utils/ort_graph_to_proto.h"
#include "utils/ort_model_dump.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>
#include <onnxruntime_cxx_api.h>

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{

std::filesystem::path TestPath(const std::string& file_name)
{
    return std::filesystem::current_path() / file_name;
}

void RemoveIfExists(const std::filesystem::path& path)
{
    std::error_code ec;
    (void)std::filesystem::remove(path, ec);
}

void RemoveDirectoryIfExists(const std::filesystem::path& path)
{
    std::error_code ec;
    (void)std::filesystem::remove_all(path, ec);
}

class ScopedCurrentPath
{
public:
    explicit ScopedCurrentPath(const std::filesystem::path& path)
        : previous_path_(std::filesystem::current_path())
    {
        std::filesystem::current_path(path);
    }

    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath(ScopedCurrentPath&&) = delete;
    ScopedCurrentPath& operator=(ScopedCurrentPath&&) = delete;

    ~ScopedCurrentPath()
    {
        std::error_code ec;
        std::filesystem::current_path(previous_path_, ec);
    }

private:
    std::filesystem::path previous_path_;
};

std::set<std::filesystem::path> ListOnnxFiles(const std::filesystem::path& directory)
{
    std::set<std::filesystem::path> files;
    if (!std::filesystem::is_directory(directory))
    {
        return files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx")
        {
            files.insert(std::filesystem::absolute(entry.path()));
        }
    }
    return files;
}

std::vector<std::filesystem::path> Difference(const std::set<std::filesystem::path>& after,
                                              const std::set<std::filesystem::path>& before)
{
    std::vector<std::filesystem::path> result;
    for (const auto& path : after)
    {
        if (before.find(path) == before.end())
        {
            result.push_back(path);
        }
    }
    return result;
}

std::string GetExternalDataValue(const onnx::TensorProto& tensor, const std::string& key)
{
    for (const auto& entry : tensor.external_data())
    {
        if (entry.key() == key)
        {
            return entry.value();
        }
    }
    return {};
}

bool IsMemAddrTensor(const onnx::TensorProto& tensor)
{
    return tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL &&
           GetExternalDataValue(tensor, "location") == model_builder::kMemAddrExternalDataLocation;
}

bool GraphHasMemAddrTensor(const onnx::GraphProto& graph)
{
    for (const auto& tensor : graph.initializer())
    {
        if (IsMemAddrTensor(tensor))
        {
            return true;
        }
    }

    for (const auto& node : graph.node())
    {
        for (const auto& attr : node.attribute())
        {
            if (attr.has_t() && IsMemAddrTensor(attr.t()))
            {
                return true;
            }

            for (const auto& tensor : attr.tensors())
            {
                if (IsMemAddrTensor(tensor))
                {
                    return true;
                }
            }

            if (attr.has_g() && GraphHasMemAddrTensor(attr.g()))
            {
                return true;
            }

            for (const auto& nested_graph : attr.graphs())
            {
                if (GraphHasMemAddrTensor(nested_graph))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

onnx::ModelProto LoadModel(const std::filesystem::path& path)
{
    onnx::ModelProto model;
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
    {
        ADD_FAILURE() << "Failed to open " << path;
        return model;
    }
    if (!model.ParseFromIstream(&input))
    {
        ADD_FAILURE() << "Failed to parse " << path;
    }
    return model;
}

std::vector<float> RunSingleInputSingleOutputFloatModel(const std::filesystem::path& model_path,
                                                        const std::vector<float>& input_data,
                                                        const std::vector<int64_t>& input_shape)
{
    Ort::SessionOptions session_options;
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
    {
        throw std::runtime_error("Expected a single-input, single-output model.");
    }

    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::array<const char*, 1> input_names = {input_name.get()};
    std::array<const char*, 1> output_names = {output_name.get()};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()),
                                                        input_data.size(), input_shape.data(), input_shape.size());

    Ort::RunOptions run_options;
    auto outputs = session.Run(run_options, input_names.data(), &input_tensor, input_names.size(), output_names.data(),
                               output_names.size());

    if (outputs.size() != 1)
    {
        throw std::runtime_error("Expected one output from model run.");
    }

    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error("Expected float output from model run.");
    }

    const auto element_count = static_cast<size_t>(output_info.GetElementCount());
    const float* output_data = outputs[0].GetTensorData<float>();
    return std::vector<float>(output_data, output_data + element_count);
}

void RunModelWithTrtRtxDump(const std::filesystem::path& model_path, const std::vector<Ort::ConstEpDevice>& devices,
                            const std::vector<float>& input_data, const std::vector<int64_t>& input_shape,
                            const char* excluded_op_types = nullptr)
{
    Ort::SessionOptions session_options;
    Ort::KeyValuePairs provider_options;
    provider_options.Add("nv_dump_subgraphs", "1");
    if (excluded_op_types != nullptr)
    {
        provider_options.Add("nv_op_types_to_exclude", excluded_op_types);
    }
    session_options.AppendExecutionProvider_V2(*ort_env, devices, provider_options);
    Ort::Session session(*ort_env, toOrtString(model_path).c_str(), session_options);

    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
    {
        throw std::runtime_error("Expected a single-input, single-output model.");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::array<const char*, 1> input_names = {input_name.get()};
    std::array<const char*, 1> output_names = {output_name.get()};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()),
                                                        input_data.size(), input_shape.data(), input_shape.size());

    Ort::RunOptions run_options;
    auto outputs = session.Run(run_options, input_names.data(), &input_tensor, input_names.size(), output_names.data(),
                               output_names.size());

    if (outputs.size() != 1)
    {
        throw std::runtime_error("Expected one output from model run.");
    }
}

std::vector<char> ReadBinaryFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    EXPECT_TRUE(input.good()) << "Failed to open " << path;
    return std::vector<char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void ExpectAllClose(const std::vector<float>& actual, const std::vector<float>& expected)
{
    constexpr float kAbsTolerance = 1.0e-6F;
    constexpr float kRelTolerance = 1.0e-5F;
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const float tolerance = kAbsTolerance + (kRelTolerance * std::fabs(expected[i]));
        EXPECT_LE(std::fabs(actual[i] - expected[i]), tolerance) << "Output mismatch at index " << i;
    }
}

}  // namespace

TEST(TensorRTRTXSubgraphDumpTest, SmallDumpEmbedsMemAddrInitializersInline)
{
    const auto onnx_path = TestPath("dump_model_with_initializers_inline.onnx");
    const auto data_path = TestPath("dump_model_with_initializers_inline.data");
    RemoveIfExists(onnx_path);
    RemoveIfExists(data_path);

    const std::vector<float> weight = {1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> attr_tensor = {5.0F};
    const std::vector<float> attr_tensor_list = {6.0F};
    const std::vector<float> nested_graph_initializer = {7.0F};
    const std::vector<float> nested_graph_list_initializer = {8.0F};

    auto model = model_builder::CreateDumpTestModelWithNestedMemAddrTensors(
        weight.data(), weight.size() * sizeof(float), attr_tensor.data(), attr_tensor.size() * sizeof(float),
        attr_tensor_list.data(), attr_tensor_list.size() * sizeof(float), nested_graph_initializer.data(),
        nested_graph_initializer.size() * sizeof(float), nested_graph_list_initializer.data(),
        nested_graph_list_initializer.size() * sizeof(float));

    const auto status = OrtEpUtils::DumpModelWithInitializers(model, onnx_path.string(), data_path.string());
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_TRUE(std::filesystem::is_regular_file(onnx_path));
    EXPECT_FALSE(std::filesystem::exists(data_path));

    const auto dumped = LoadModel(onnx_path);
    ASSERT_FALSE(GraphHasMemAddrTensor(dumped.graph()));
    ASSERT_EQ(dumped.graph().initializer_size(), 1);
    EXPECT_EQ(dumped.graph().initializer(0).raw_data(),
              std::string(reinterpret_cast<const char*>(weight.data()), weight.size() * sizeof(float)));

    RemoveIfExists(onnx_path);
    RemoveIfExists(data_path);
}

TEST(TensorRTRTXSubgraphDumpTest, SingleSubgraphDumpRunsStandaloneAndMatchesOriginalOutput)
{
    ASSERT_NE(ort_env.get(), nullptr);
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";

    const auto work_dir = TestPath("subgraph_dump_functional_test");
    RemoveDirectoryIfExists(work_dir);
    std::filesystem::create_directories(work_dir);
    ScopedCurrentPath scoped_path(work_dir);

    const auto model_path = std::filesystem::current_path() / "dump_functional_initializer_matmul.onnx";
    model_builder::CreateInitializerMatMulModel(model_path.string());

    const std::vector<float> input_data = {1.0F, -2.0F, 0.5F, 3.0F};
    const std::vector<int64_t> input_shape = {1, 4};
    const auto expected = RunSingleInputSingleOutputFloatModel(model_path, input_data, input_shape);

    const auto before = ListOnnxFiles(std::filesystem::current_path());
    RunModelWithTrtRtxDump(model_path, devices, input_data, input_shape);

    const auto after = ListOnnxFiles(std::filesystem::current_path());
    const auto dumps = Difference(after, before);
    ASSERT_EQ(dumps.size(), 1U);
    auto dump_data_path = dumps[0];
    dump_data_path.replace_extension(".data");
    EXPECT_FALSE(std::filesystem::exists(dump_data_path));

    const auto dumped = LoadModel(dumps[0]);
    ASSERT_FALSE(GraphHasMemAddrTensor(dumped.graph()));
    ASSERT_GT(dumped.graph().initializer_size(), 0);

    const auto actual = RunSingleInputSingleOutputFloatModel(dumps[0], input_data, input_shape);
    ExpectAllClose(actual, expected);

    RemoveIfExists(dumps[0]);
    RemoveIfExists(dump_data_path);
}

TEST(TensorRTRTXSubgraphDumpTest, MultipleSubgraphsCreateUniqueStandaloneDumps)
{
    ASSERT_NE(ort_env.get(), nullptr);
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";

    const auto work_dir = TestPath("subgraph_dump_multiple_test");
    RemoveDirectoryIfExists(work_dir);
    std::filesystem::create_directories(work_dir);
    ScopedCurrentPath scoped_path(work_dir);

    const auto model_path = std::filesystem::current_path() / "two_initializer_matmul_subgraphs.onnx";
    model_builder::CreateTwoInitializerMatMulSubgraphsModel(model_path.string());

    const std::vector<float> input_data = {1.0F, -2.0F, 0.5F, 3.0F};
    const std::vector<int64_t> input_shape = {1, 4};

    const auto before = ListOnnxFiles(std::filesystem::current_path());
    RunModelWithTrtRtxDump(model_path, devices, input_data, input_shape, "FastGelu");

    const auto after = ListOnnxFiles(std::filesystem::current_path());
    const auto dumps = Difference(after, before);
    ASSERT_GE(dumps.size(), 2U);

    std::set<std::string> dump_names;
    for (const auto& dump : dumps)
    {
        SCOPED_TRACE(dump.string());
        EXPECT_TRUE(dump_names.insert(dump.filename().string()).second);

        auto dump_data_path = dump;
        dump_data_path.replace_extension(".data");
        EXPECT_FALSE(std::filesystem::exists(dump_data_path));

        const auto dumped = LoadModel(dump);
        ASSERT_FALSE(GraphHasMemAddrTensor(dumped.graph()));
        ASSERT_GT(dumped.graph().initializer_size(), 0);
        (void)RunSingleInputSingleOutputFloatModel(dump, input_data, input_shape);

        RemoveIfExists(dump);
        RemoveIfExists(dump_data_path);
    }
}

TEST(TensorRTRTXSubgraphDumpTest, ResNetDumpRunsSuccessfully)
{
    ASSERT_NE(ort_env.get(), nullptr);
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath)) << "Model not found at: " << kModelPath;

    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";

    const auto work_dir = TestPath("subgraph_dump_resnet_test");
    RemoveDirectoryIfExists(work_dir);
    std::filesystem::create_directories(work_dir);
    ScopedCurrentPath scoped_path(work_dir);

    const auto before = ListOnnxFiles(std::filesystem::current_path());

    Ort::SessionOptions session_options;
    Ort::KeyValuePairs provider_options;
    provider_options.Add("nv_dump_subgraphs", "1");
    session_options.AppendExecutionProvider_V2(*ort_env, devices, provider_options);

    Ort::Session session(*ort_env, toOrtString(kModelPath).c_str(), session_options);
    auto io_binding = generate_io_binding(session);
    Ort::RunOptions run_options;
    ASSERT_NO_THROW(session.Run(run_options, io_binding));

    const auto after = ListOnnxFiles(std::filesystem::current_path());
    const auto dumps = Difference(after, before);
    ASSERT_EQ(dumps.size(), 1U);

    const auto& dump = dumps[0];
    SCOPED_TRACE(dump.string());
    Ort::SessionOptions dump_session_options;
    Ort::Session dump_session(*ort_env, toOrtString(dump).c_str(), dump_session_options);
    auto dump_io_binding = generate_io_binding(dump_session);
    Ort::RunOptions dump_run_options;
    ASSERT_NO_THROW(dump_session.Run(dump_run_options, dump_io_binding));

    auto dump_data_path = dump;
    dump_data_path.replace_extension(".data");
    RemoveIfExists(dump);
    RemoveIfExists(dump_data_path);
}

TEST(TensorRTRTXSubgraphDumpTest, LargeDumpWritesInitializersToExternalDataFile)
{
    const auto onnx_path = TestPath("dump_model_with_initializers_external.onnx");
    const auto data_path = TestPath("dump_model_with_initializers_external.data");
    RemoveIfExists(onnx_path);
    RemoveIfExists(data_path);

    constexpr size_t kWeightBytes = 4096U;
    std::vector<char> weight(kWeightBytes);
    for (size_t i = 0; i < weight.size(); ++i)
    {
        weight[i] = static_cast<char>(i % 251U);
    }

    auto model = model_builder::CreateDumpTestMatMulModelWithMemAddrWeight(weight.data(), weight.size(), 1024, 1);

    const auto status = OrtEpUtils::DumpModelWithInitializers(model, onnx_path.string(), data_path.string());
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_TRUE(std::filesystem::is_regular_file(onnx_path));
    ASSERT_TRUE(std::filesystem::is_regular_file(data_path));

    const auto dumped = LoadModel(onnx_path);
    ASSERT_EQ(dumped.graph().initializer_size(), 1);
    const auto& dumped_tensor = dumped.graph().initializer(0);
    EXPECT_FALSE(dumped_tensor.has_raw_data());
    EXPECT_EQ(dumped_tensor.data_location(), onnx::TensorProto_DataLocation_EXTERNAL);
    EXPECT_EQ(GetExternalDataValue(dumped_tensor, "location"), data_path.filename().string());
    EXPECT_EQ(GetExternalDataValue(dumped_tensor, "offset"), "0");
    EXPECT_EQ(GetExternalDataValue(dumped_tensor, "length"), std::to_string(weight.size()));

    const auto data_file_bytes = ReadBinaryFile(data_path);
    EXPECT_EQ(data_file_bytes, weight);

    RemoveIfExists(onnx_path);
    RemoveIfExists(data_path);
}
