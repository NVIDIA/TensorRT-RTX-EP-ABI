// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from onnxruntime/test/providers/nv_tensorrt_rtx/nv_basic_test.cc
// Uses only public ORT SDK APIs.

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;
extern std::filesystem::path g_ep_lib_path;

// Returns true if a CUDA device of SM >= 12.0 (Blackwell+) is present.
// Returns false if CUDA is unavailable or the device is below SM 12.0.
static bool IsBlackwellOrAbove() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return false;
    }
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return false;
    }
    const int cuda_arch = prop.major * 100 + prop.minor * 10;
    return cuda_arch >= 1200;  // SM 12.0 = Blackwell
}

// Helper: append TRT RTX EP to session options using the global registration.
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

// =============================================================================
// Context embed and reload tests
// =============================================================================

TEST(TensorRTRTXEpTest, ContextEmbedAndReload) {
    const std::string model_name = "nv_execution_provider_test.onnx";
    const std::string model_name_ctx = "nv_execution_provider_test_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    model_builder::CreateBaseModel(model_name, "test", {1, 3, 2});

    // AOT: compile and embed EP context
    {
        auto start = std::chrono::high_resolution_clock::now();
        Ort::SessionOptions so;
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx.c_str());
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Session creation AOT: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " ms\n";

        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }

    // JIT: reload cached context
    {
        auto start = std::chrono::high_resolution_clock::now();
        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name_ctx).c_str(), so);
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Session creation JIT: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " ms\n";

        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }
}

TEST(TensorRTRTXEpTest, ContextEmbedAndReloadDynamic) {
    const std::string model_name = "nv_execution_provider_dyn_test.onnx";
    const std::string model_name_ctx = "nv_execution_provider_dyn_test_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    model_builder::CreateBaseModel(model_name, "test", {1, -1, -1});

    // AOT
    {
        Ort::SessionOptions so;
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx.c_str());
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }

    // JIT with shape overrides
    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name_ctx).c_str(), so);

        std::map<std::string, std::vector<int64_t>> shape_overwrites;
        shape_overwrites["X"] = {1, 5, 5};
        shape_overwrites["Y"] = {1, 5, 1};
        auto io_binding = generate_io_binding(session, shape_overwrites);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }
}

TEST(TensorRTRTXEpTest, ContextEmbedAndReloadDataDynamic) {
    const std::string model_name = "nv_execution_provider_data_dyn_test.onnx";
    const std::string model_name_ctx = "nv_execution_provider_data_dyn_test_ctx.onnx";
    clearFileIfExists(model_name_ctx);

    model_builder::CreateBaseModel(model_name, "test", {1, -1, -1});

    // AOT
    {
        Ort::SessionOptions so;
        so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
        so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx.c_str());
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }

    // JIT with shape overrides
    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name_ctx).c_str(), so);

        std::map<std::string, std::vector<int64_t>> shape_overwrites;
        shape_overwrites["X"] = {1, 5, 5};
        shape_overwrites["Y"] = {1, 5, 5};
        auto io_binding = generate_io_binding(session, shape_overwrites);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }
}

// =============================================================================
// Type tests (parameterized)
// =============================================================================

static std::string getTypeAsName(int dtype) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "fp32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  return "fp16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bf16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return "int32";
        default: return "unknown";
    }
}

class TypeTests : public ::testing::TestWithParam<int> {};

TEST_P(TypeTests, IOTypes) {
    const int dtype = GetParam();
    const std::string dtype_name = getTypeAsName(dtype);
    ASSERT_NE(dtype_name, "unknown");

    const std::string model_name =
        "nv_execution_provider_" + dtype_name + ".onnx";

    model_builder::CreateBaseModel(model_name, "test_" + dtype_name,
                                   {1, 5, 10}, false, dtype);
    {
        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

        auto io_binding = generate_io_binding(session);
        Ort::RunOptions run_options;
        session.Run(run_options, io_binding);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TensorRTRTXEpTest, TypeTests,
    ::testing::Values(
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
    [](const ::testing::TestParamInfo<int>& info) {
        return getTypeAsName(info.param);
    });

// =============================================================================
// TestSessionOutputs
// =============================================================================

TEST(TensorRTRTXEpTest, TestSessionOutputs) {
    // Model #1: TopK with 4 outputs of different types
    {
        const std::string model_path = "topk_and_multiple_graph_outputs.onnx";
        model_builder::CreateTopkAndMultipleGraphOutputsModel(model_path);

        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        so.AddFreeDimensionOverrideByName("N", 300);
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), so);

        ASSERT_EQ(session.GetOutputCount(), 4u);
    }

    // Model #2: Dropout with unused mask output
    {
        const std::string model_path = "node_output_not_used.onnx";
        model_builder::CreateNodeOutputNotUsedModel(model_path);

        Ort::SessionOptions so;
        AppendTrtRtxEp(so);
        Ort::Session session(*ort_env, toOrtString(model_path).c_str(), so);

        ASSERT_EQ(session.GetOutputCount(), 1u);
    }
}

// =============================================================================
// GetSharedAllocator
// =============================================================================

TEST(TensorRTRTXEpTest, GetSharedAllocator) {
    const OrtApi& c_api = Ort::GetApi();
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty());

    const OrtMemoryInfo* device_mem_info =
        c_api.EpDevice_MemoryInfo(devices[0], OrtDeviceMemoryType_DEFAULT);

    OrtAllocator* allocator = nullptr;
    Ort::ThrowOnError(c_api.GetSharedAllocator(*ort_env, device_mem_info, &allocator));
    ASSERT_NE(allocator, nullptr);

    const OrtMemoryInfo* host_mem_info =
        c_api.EpDevice_MemoryInfo(devices[0], OrtDeviceMemoryType_HOST_ACCESSIBLE);

    OrtAllocator* host_allocator = nullptr;
    Ort::ThrowOnError(c_api.GetSharedAllocator(*ort_env, host_mem_info, &host_allocator));
    ASSERT_NE(host_allocator, nullptr);
}

// =============================================================================
// LoadUnloadPluginLibrary (C API)
// =============================================================================

TEST(TensorRTRTXEpTest, LoadUnloadPluginLibrary) {
    ASSERT_FALSE(g_ep_lib_path.empty()) << "EP library path not set.";
    const OrtApi& c_api = Ort::GetApi();

    // Unregister the globally registered EP first
    Ort::ThrowOnError(c_api.UnregisterExecutionProviderLibrary(
        *ort_env, kEpName));

    // Register
    Ort::ThrowOnError(c_api.RegisterExecutionProviderLibrary(
        *ort_env, kEpName, toOrtString(g_ep_lib_path).c_str()));

    // Verify device exists
    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_devices = 0;
    Ort::ThrowOnError(c_api.GetEpDevices(*ort_env, &ep_devices, &num_devices));

    auto count = std::count_if(
        ep_devices, ep_devices + num_devices,
        [&](const OrtEpDevice* d) {
            return std::string(c_api.EpDevice_EpName(d)) == kEpName;
        });
    ASSERT_GE(count, 1) << "Expected at least one OrtEpDevice for TRT RTX EP.";

    // Unregister
    Ort::ThrowOnError(c_api.UnregisterExecutionProviderLibrary(
        *ort_env, kEpName));

    // Re-register for subsequent tests
    Ort::ThrowOnError(c_api.RegisterExecutionProviderLibrary(
        *ort_env, kEpName, toOrtString(g_ep_lib_path).c_str()));
}

// =============================================================================
// LoadUnloadPluginLibraryCxxApi
// =============================================================================

TEST(TensorRTRTXEpTest, LoadUnloadPluginLibraryCxxApi) {
    ASSERT_FALSE(g_ep_lib_path.empty()) << "EP library path not set.";

    // Unregister the globally registered EP first
    ort_env->UnregisterExecutionProviderLibrary(kEpName);

    // Register via C++ API
    ort_env->RegisterExecutionProviderLibrary(kEpName,
                                              toOrtString(g_ep_lib_path).c_str());

    auto ep_devices = ort_env->GetEpDevices();
    auto it = std::find_if(
        ep_devices.begin(), ep_devices.end(),
        [](const Ort::ConstEpDevice& d) {
            return std::string(d.EpName()) == kEpName;
        });
    ASSERT_NE(it, ep_devices.end())
        << "Expected an OrtEpDevice for TRT RTX EP.";

    ASSERT_STREQ(it->EpVendor(), "NVIDIA");

    Ort::ConstHardwareDevice device = it->Device();
    ASSERT_EQ(device.Type(), OrtHardwareDeviceType_GPU);
    ASSERT_GE(device.VendorId(), 0);
    ASSERT_GE(device.DeviceId(), 0);
    ASSERT_NE(device.Vendor(), nullptr);

    Ort::ConstKeyValuePairs metadata = device.Metadata();
    auto entries = metadata.GetKeyValuePairs();
    ASSERT_GT(entries.size(), 0u);

    // Unregister
    ort_env->UnregisterExecutionProviderLibrary(kEpName);

    // Re-register for subsequent tests
    ort_env->RegisterExecutionProviderLibrary(kEpName,
                                              toOrtString(g_ep_lib_path).c_str());
}

// =============================================================================
// DataTransfer
// =============================================================================

TEST(TensorRTRTXEpTest, DataTransfer) {
    const OrtApi& c_api = Ort::GetApi();
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty());

    const OrtMemoryInfo* device_mem_info =
        c_api.EpDevice_MemoryInfo(devices[0], OrtDeviceMemoryType_DEFAULT);

    // Create CPU tensor with random data
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    std::vector<int64_t> shape{2, 3, 4};
    const size_t num_elements = 2 * 3 * 4;

    std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> input_data(num_elements);
    std::generate(input_data.begin(), input_data.end(),
                  [&]() { return dist(rng); });

    Ort::Value cpu_tensor = Ort::Value::CreateTensor<float>(
        cpu_allocator.GetInfo(), input_data.data(), input_data.size(),
        shape.data(), shape.size());

    // Create GPU tensor using shared allocator
    OrtAllocator* gpu_allocator = nullptr;
    Ort::ThrowOnError(c_api.GetSharedAllocator(
        *ort_env, device_mem_info, &gpu_allocator));
    ASSERT_NE(gpu_allocator, nullptr);

    Ort::Value device_tensor = Ort::Value::CreateTensor<float>(
        gpu_allocator, shape.data(), shape.size());

    // Copy CPU -> GPU
    {
        const OrtValue* src[] = {cpu_tensor};
        OrtValue* dst[] = {device_tensor};
        Ort::ThrowOnError(c_api.CopyTensors(
            *ort_env, src, dst, nullptr, 1));
    }

    // Copy GPU -> CPU (new tensor)
    Ort::Value cpu_copy = Ort::Value::CreateTensor<float>(
        cpu_allocator, shape.data(), shape.size());
    {
        const OrtValue* src[] = {device_tensor};
        OrtValue* dst[] = {cpu_copy};
        Ort::ThrowOnError(c_api.CopyTensors(
            *ort_env, src, dst, nullptr, 1));
    }

    // Verify round-trip data integrity
    const float* original = cpu_tensor.GetTensorData<float>();
    const float* copied = cpu_copy.GetTensorData<float>();
    ASSERT_NE(original, copied) << "Should be different memory locations";

    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_FLOAT_EQ(original[i], copied[i])
            << "Mismatch at index " << i;
    }

    // Release GPU tensor before EP teardown
    device_tensor = Ort::Value(nullptr);
}

// =============================================================================
// AutoEp_PreferGpu — session picks the TRT RTX EP via PREFER_GPU policy.
//
// Note: the ORT public C/C++ API does not expose the list of EPs actually
// bound to a running session, so (unlike the internal-API source test) we can
// only verify that session creation and inference succeed with the
// PREFER_GPU policy and that the TRT RTX EpDevice is visible in the env.
// =============================================================================

TEST(TensorRTRTXEpTest, AutoEp_PreferGpu) {
    ASSERT_FALSE(g_ep_lib_path.empty()) << "EP library path not set.";

    const std::string model_name = "nv_execution_provider_auto_ep.onnx";
    model_builder::CreateBaseModel(model_name, "test", {1, 3, 2});

    // The EP is already registered from main(); verify its device is visible.
    const auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty())
        << "TRT RTX EpDevice not visible in env; PREFER_GPU has nothing to pick.";

    Ort::SessionOptions so;
    so.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

    // Session creation succeeds -> some EP was auto-selected.
    Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

    // Sanity: run inference once so any failure in the selected EP surfaces.
    auto io_binding = generate_io_binding(session);
    Ort::RunOptions run_options;
    session.Run(run_options, io_binding);
}

// =============================================================================
// FP8 / FP4 custom-op models — exercise TensorRT FP8 and FP4 custom ops.
// Both require Blackwell (SM 12.0+) hardware.
// =============================================================================

TEST(TensorRTRTXEpTest, FP8CustomOpModel) {
    if (!IsBlackwellOrAbove()) {
        GTEST_SKIP() << "Test requires SM 12.0+ GPU (Blackwell+).";
    }

    const std::string model_name =
        "nv_execution_provider_fp8_quantize_dequantize_test.onnx";
    clearFileIfExists(model_name);

    model_builder::CreateFP8CustomOpModel(
        model_name, "nv_execution_provider_fp8_quantize_dequantize_graph");
    ASSERT_TRUE(std::filesystem::exists(model_name));

    Ort::SessionOptions so;
    AppendTrtRtxEp(so);
    Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

    ASSERT_EQ(session.GetInputCount(), 1u);
    ASSERT_EQ(session.GetOutputCount(), 1u);

    // Input: FP16 [4,64] — values in [0, 1). ORT represents FP16 as uint16_t.
    const std::vector<int64_t> input_shape = {4, 64};
    const size_t num_elements = 4 * 64;
    std::vector<uint16_t> input_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        input_data[i] = model_builder::Float32ToFloat16(
            static_cast<float>(i % 100) / 100.0f);
    }

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        cpu_allocator.GetInfo(), input_data.data(),
        input_data.size() * sizeof(uint16_t),
        input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    const char* input_names[]  = {"X"};
    const char* output_names[] = {"Y"};
    Ort::RunOptions run_options;
    auto outputs = session.Run(run_options, input_names, &input_tensor, 1,
                               output_names, 1);

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());

    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(out_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    auto out_shape = out_info.GetShape();
    ASSERT_EQ(out_shape.size(), 2u);
    EXPECT_EQ(out_shape[0], 4);
    EXPECT_EQ(out_shape[1], 64);
}

TEST(TensorRTRTXEpTest, FP4CustomOpModel) {
    if (!IsBlackwellOrAbove()) {
        GTEST_SKIP() << "Test requires SM 12.0+ GPU (Blackwell+).";
    }

    const std::string model_name =
        "nv_execution_provider_fp4_dynamic_quantize_test.onnx";
    clearFileIfExists(model_name);

    model_builder::CreateFP4CustomOpModel(
        model_name, "nv_execution_provider_fp4_dynamic_quantize_graph");
    ASSERT_TRUE(std::filesystem::exists(model_name));

    Ort::SessionOptions so;
    AppendTrtRtxEp(so);
    Ort::Session session(*ort_env, toOrtString(model_name).c_str(), so);

    ASSERT_EQ(session.GetInputCount(), 1u);
    ASSERT_EQ(session.GetOutputCount(), 1u);

    // Input: FP16 [64,64]
    const std::vector<int64_t> input_shape = {64, 64};
    const size_t num_elements = 64 * 64;
    std::vector<uint16_t> input_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        input_data[i] = model_builder::Float32ToFloat16(
            static_cast<float>(i % 100) / 100.0f);
    }

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        cpu_allocator.GetInfo(), input_data.data(),
        input_data.size() * sizeof(uint16_t),
        input_shape.data(), input_shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    const char* input_names[]  = {"X"};
    const char* output_names[] = {"X_dequantized"};
    Ort::RunOptions run_options;
    auto outputs = session.Run(run_options, input_names, &input_tensor, 1,
                               output_names, 1);

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());

    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(out_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    auto out_shape = out_info.GetShape();
    ASSERT_EQ(out_shape.size(), 2u);
    EXPECT_EQ(out_shape[0], 64);
    EXPECT_EQ(out_shape[1], 64);
}
