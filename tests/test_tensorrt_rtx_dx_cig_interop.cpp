// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_rtx_provider_options.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#if defined(_WIN32)
#include <d3d12.h>
#include <Windows.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

extern std::unique_ptr<Ort::Env> ort_env;
extern std::filesystem::path g_ep_lib_path;

// =============================================================================
// D3D12 Helpers
// =============================================================================
namespace
{

struct D3D12CreateDeviceLoadResult
{
    HMODULE dxgi_module = nullptr;
    HMODULE module = nullptr;
    typedef HRESULT(WINAPI* PFN_D3D12CreateDevice)(IUnknown* pAdapter, D3D_FEATURE_LEVEL MinimumFeatureLevel,
                                                   REFIID riid, void** ppDevice);
    PFN_D3D12CreateDevice pfn = nullptr;
};

D3D12CreateDeviceLoadResult LoadD3D12CreateDevice()
{
    D3D12CreateDeviceLoadResult r;
    // Load DXGI first — D3D12 depends on it for adapter enumeration, and pre-loading
    // ensures DXGI state is initialized before any CUDA/D3D12 interop.
    r.dxgi_module = LoadLibraryW(L"dxgi.dll");
    r.module = LoadLibraryW(L"d3d12.dll");
    if (r.module)
    {
        r.pfn = reinterpret_cast<D3D12CreateDeviceLoadResult::PFN_D3D12CreateDevice>(
            GetProcAddress(r.module, "D3D12CreateDevice"));
    }
    return r;
}

void CreateD3D12Buffer(ID3D12Device* pDevice, size_t size, ID3D12Resource** ppResource, D3D12_RESOURCE_STATES initState)
{
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Width = size;
    bufferDesc.Height = 1;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.SampleDesc.Quality = 0;
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    HRESULT hr = pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, initState, nullptr,
                                                  IID_PPV_ARGS(ppResource));
    if (FAILED(hr))
    {
        GTEST_FAIL() << "Failed creating D3D12 resource, HRESULT: 0x" << std::hex << hr;
    }
}

void CreateUploadBuffer(ID3D12Device* pDevice, size_t size, ID3D12Resource** ppResource)
{
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = size;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(ppResource));
    if (FAILED(hr))
    {
        GTEST_FAIL() << "Failed creating D3D12 upload resource, HRESULT: 0x" << std::hex << hr;
    }
}

void CreateReadBackBuffer(ID3D12Device* pDevice, size_t size, ID3D12Resource** ppResource)
{
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = size;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                  D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(ppResource));
    if (FAILED(hr))
    {
        GTEST_FAIL() << "Failed creating D3D12 readback resource, HRESULT: 0x" << std::hex << hr;
    }
}

void FlushAndWait(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue)
{
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (hEvent == nullptr)
    {
        GTEST_FAIL() << "CreateEvent failed, error: " << GetLastError();
        return;
    }
    ComPtr<ID3D12Fence> pFence;
    HRESULT hr = pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));
    if (FAILED(hr))
    {
        CloseHandle(hEvent);
        GTEST_FAIL() << "CreateFence failed, HRESULT: 0x" << std::hex << hr;
        return;
    }
    hr = pQueue->Signal(pFence.Get(), 1);
    if (FAILED(hr))
    {
        CloseHandle(hEvent);
        GTEST_FAIL() << "Signal failed, HRESULT: 0x" << std::hex << hr;
        return;
    }
    hr = pFence->SetEventOnCompletion(1, hEvent);
    if (FAILED(hr))
    {
        CloseHandle(hEvent);
        GTEST_FAIL() << "SetEventOnCompletion failed, HRESULT: 0x" << std::hex << hr;
        return;
    }
    WaitForSingleObject(hEvent, INFINITE);
    CloseHandle(hEvent);
}

}  // namespace

// =============================================================================
// Test 0: EP registration and device enumeration (works with any ORT version)
// =============================================================================
TEST(CigInteropTest, EpRegistrationAndDeviceEnumeration)
{
    ASSERT_NE(ort_env.get(), nullptr);

    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "TRT RTX EP device not found. Is a supported GPU available?";
    const char* ep_name = devices[0].EpName();
    ASSERT_NE(ep_name, nullptr);
    EXPECT_STREQ(ep_name, kEpName);
}

// =============================================================================
// Tests 1-3: CIG interop (require ORT 1.25+)
// =============================================================================
//
// Each test below reads the EP's negotiated ORT API version from EpDevice
// metadata (published by the EP under "nv_ep_ort_api_version") and skips if
// it's < 25. This is required because when the EP DLL was built with ORT < 1.25,
// its OrtEpFactory struct does not declare InitGraphicsInterop, so the host
// (1.25+) would read that callback at an offset past the EP's allocated struct
// and dispatch into garbage — a hard crash, not catchable by the post-call
// status check. Reading EpDevice_EpMetadata is safe at any version.
#if ORT_API_VERSION >= 25

// Test 1: Init with null command_queue should succeed gracefully
TEST(CigInteropTest, InitWithoutCommandQueue)
{
    ASSERT_NE(ort_env.get(), nullptr);

    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "TRT RTX EP device not found";
    const OrtEpDevice* ep_device = static_cast<const OrtEpDevice*>(devices[0]);

    const int ep_api_version = ep_negotiated_ort_api_version(ep_device);
    if (ep_api_version >= 0 && ep_api_version < 25)
    {
        GTEST_SKIP() << "EP DLL negotiated ORT API version " << ep_api_version
                     << "; graphics-interop API requires >= 25.";
    }

    const OrtApi& api = Ort::GetApi();
    const OrtInteropApi& interop_api = Ort::GetInteropApi();

    OrtGraphicsInteropConfig config = {};
    config.version = ORT_API_VERSION;
    config.graphics_api = ORT_GRAPHICS_API_D3D12;
    config.command_queue = nullptr;
    config.additional_options = nullptr;

    OrtStatus* status = interop_api.InitGraphicsInteropForEpDevice(ep_device, &config);
    if (status != nullptr)
    {
        std::string msg = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        FAIL() << "InitGraphicsInterop with null command_queue failed: " << msg;
    }

    status = interop_api.DeinitGraphicsInteropForEpDevice(ep_device);
    ASSERT_EQ(status, nullptr) << "DeinitGraphicsInterop should succeed";
}

// Test 2: Full D3D12 lifecycle — init, create stream, release, deinit
TEST(CigInteropTest, D3D12InitStreamDeinit)
{
    ASSERT_NE(ort_env.get(), nullptr);

    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "TRT RTX EP device not found";
    const OrtEpDevice* ep_device = static_cast<const OrtEpDevice*>(devices[0]);

    const int ep_api_version = ep_negotiated_ort_api_version(ep_device);
    if (ep_api_version >= 0 && ep_api_version < 25)
    {
        GTEST_SKIP() << "EP DLL negotiated ORT API version " << ep_api_version
                     << "; graphics-interop API requires >= 25.";
    }

    D3D12CreateDeviceLoadResult d3d12 = LoadD3D12CreateDevice();
    if (!d3d12.pfn)
    {
        GTEST_SKIP() << "d3d12.dll or D3D12CreateDevice not available";
    }

    ComPtr<ID3D12Device> pDevice;
    HRESULT hr = d3d12.pfn(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));
    if (FAILED(hr))
    {
        GTEST_SKIP() << "D3D12 device creation failed, HRESULT: 0x" << std::hex << hr;
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ComPtr<ID3D12CommandQueue> pCommandQueue;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));
    if (FAILED(hr))
    {
        GTEST_SKIP() << "D3D12 command queue creation failed, HRESULT: 0x" << std::hex << hr;
    }

    const OrtApi& api = Ort::GetApi();
    const OrtInteropApi& interop_api = Ort::GetInteropApi();

    OrtGraphicsInteropConfig config = {};
    config.version = ORT_API_VERSION;
    config.graphics_api = ORT_GRAPHICS_API_D3D12;
    config.command_queue = pCommandQueue.Get();
    config.additional_options = nullptr;

    OrtStatus* status = interop_api.InitGraphicsInteropForEpDevice(ep_device, &config);
    if (status != nullptr)
    {
        std::string msg = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        FAIL() << "InitGraphicsInterop failed: " << msg;
    }

    OrtSyncStream* stream = nullptr;
    status = api.CreateSyncStreamForEpDevice(ep_device, nullptr, &stream);
    ASSERT_EQ(status, nullptr) << "CreateSyncStreamForEpDevice should succeed with CIG context";
    ASSERT_NE(stream, nullptr);

    void* stream_handle = api.SyncStream_GetHandle(stream);
    EXPECT_NE(stream_handle, nullptr) << "Stream handle should be a valid CUDA stream";

    api.ReleaseSyncStream(stream);

    status = interop_api.DeinitGraphicsInteropForEpDevice(ep_device);
    ASSERT_EQ(status, nullptr) << "DeinitGraphicsInterop should succeed";
}

void RunD3D12ImportedResourceFullInference(bool use_cig)
{
    ASSERT_NE(ort_env.get(), nullptr);

    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "TRT RTX EP device not found";
    const OrtEpDevice* ep_device = static_cast<const OrtEpDevice*>(devices[0]);

    const int ep_api_version = ep_negotiated_ort_api_version(ep_device);
    if (ep_api_version >= 0 && ep_api_version < 25)
    {
        GTEST_SKIP() << "EP DLL negotiated ORT API version " << ep_api_version
                     << "; graphics-interop API requires >= 25.";
    }

    D3D12CreateDeviceLoadResult d3d12 = LoadD3D12CreateDevice();
    if (!d3d12.pfn)
    {
        GTEST_SKIP() << "d3d12.dll or D3D12CreateDevice not available";
    }

    ComPtr<ID3D12Device> pDevice;
    HRESULT hr = d3d12.pfn(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));
    if (FAILED(hr))
    {
        GTEST_SKIP() << "D3D12 device creation failed";
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ComPtr<ID3D12CommandQueue> pCommandQueue;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));
    if (FAILED(hr))
    {
        GTEST_SKIP() << "D3D12 command queue creation failed";
    }

    const OrtApi& api = Ort::GetApi();
    const OrtInteropApi& interop_api = Ort::GetInteropApi();

    // Init graphics interop
    OrtGraphicsInteropConfig config = {};
    config.version = ORT_API_VERSION;
    config.graphics_api = ORT_GRAPHICS_API_D3D12;
    config.command_queue = use_cig ? pCommandQueue.Get() : nullptr;
    config.additional_options = nullptr;

    OrtStatus* status = interop_api.InitGraphicsInteropForEpDevice(ep_device, &config);
    if (status != nullptr)
    {
        std::string msg = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        GTEST_FAIL() << "InitGraphicsInterop failed: " << msg;
    }

    // Create sync stream
    OrtSyncStream* stream = nullptr;
    status = api.CreateSyncStreamForEpDevice(ep_device, nullptr, &stream);
    ASSERT_EQ(status, nullptr) << "CreateSyncStreamForEpDevice failed";
    ASSERT_NE(stream, nullptr);

    // GPU-side D3D12<->CUDA synchronization via the external resource importer (ORT API v26+).
    // Because the run below sets disable_synchronize_execution_providers=1, the EP performs no
    // internal stream sync, so the test must order the work itself: import the D3D12 fence as a
    // CUDA semaphore and fence the upload->inference and inference->download hand-offs on the GPU
    // (the pattern used by the SimpleDXInterop CIG reference). CIG shares the context/queue but
    // does not auto-order CUDA-stream work against D3D12 command lists, so the fences are required.
    // On SDKs without the importer we fall back to CPU-blocking FlushAndWait.
#if ORT_API_VERSION >= 26
    bool use_semaphore_sync = false;
    OrtExternalResourceImporter* importer = nullptr;
    OrtExternalSemaphoreHandle* sync_sem = nullptr;
    ComPtr<ID3D12Fence> sync_fence;
    HANDLE shared_fence_handle = nullptr;
    {
        OrtStatus* imp_status = interop_api.CreateExternalResourceImporterForDevice(ep_device, &importer);
        if (imp_status != nullptr)
        {
            api.ReleaseStatus(imp_status);
        }
        else if (importer != nullptr &&
                 SUCCEEDED(pDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&sync_fence))) &&
                 SUCCEEDED(pDevice->CreateSharedHandle(sync_fence.Get(), nullptr, GENERIC_ALL, nullptr,
                                                       &shared_fence_handle)))
        {
            OrtExternalSemaphoreDescriptor sem_desc = {};
            sem_desc.version = ORT_API_VERSION;
            sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
            sem_desc.native_handle = shared_fence_handle;
            OrtStatus* sem_status = interop_api.ImportSemaphore(importer, &sem_desc, &sync_sem);
            if (sem_status != nullptr)
            {
                api.ReleaseStatus(sem_status);
            }
            else
            {
                use_semaphore_sync = (sync_sem != nullptr);
            }
        }
    }
#endif

    // Build model: CreateBaseModel produces O = ((X+Y)+Z)+S, all inputs 1.0f => output 4.0f.
    // X, Y, Z are shaped {1,64,64} (S is {1}). The tensors are sized large enough that the
    // deliberate stress upload below takes real GPU time, creating a window in which a missing
    // D3D12<->CUDA fence corrupts the result (so this test fails if the sync is dropped).
    const std::string model_name = "cig_interop_test_model.onnx";
    model_builder::CreateBaseModel(model_name, "cig_test", {1, 64, 64});

    // Create session with EP using the CIG user compute stream
    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.DisableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

    const auto stream_address = reinterpret_cast<size_t>(api.SyncStream_GetHandle(stream));
    const std::string stream_address_string = std::to_string(stream_address);
    Ort::KeyValuePairs ep_options;
    ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kUserComputeStream, stream_address_string.c_str());
    ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kHasUserComputeStream, "1");

    if (use_cig)
    {
        // For a simple unit test we pick 48K since this is the common lowest denominator
        // https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/compute-graphics.html#shared-memory-limitation
        const std::string max_shared_mem_string = std::to_string(48 * 1024);
        ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kMaxSharedMemSize,
                       max_shared_mem_string.c_str());
        // disable aux streams
        ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kLengthAuxStreamArray, "0");
        ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kCudaGraphEnable, "0");
    }
    ASSERT_NO_THROW(
        session_options.AppendExecutionProvider_V2(*ort_env, std::vector{Ort::ConstEpDevice(ep_device)}, ep_options));

    {
        auto ort_model_path = toOrtString(std::filesystem::path(model_name));
        Ort::Session session(*ort_env, ort_model_path.c_str(), session_options);

        // Query input/output shapes from session
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session.GetInputCount();

        // Prepare CPU data: all inputs = 1.0f
        // X, Y, Z: {1,64,64}; S: {1}
        std::vector<std::vector<float>> cpu_inputs;
        std::vector<size_t> input_byte_sizes;
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::string> input_names_str;

        for (size_t i = 0; i < num_inputs; ++i)
        {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names_str.push_back(name.get());

            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            for (auto& d : shape)
            {
                if (d < 0)
                    d = 1;
            }

            size_t elem_count = 1;
            for (auto d : shape)
                elem_count *= static_cast<size_t>(d);

            cpu_inputs.emplace_back(elem_count, 1.0f);
            input_byte_sizes.push_back(elem_count * sizeof(float));
            input_shapes.push_back(shape);
        }

        // Get output shape
        auto out_type_info = session.GetOutputTypeInfo(0);
        auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
        auto out_shape = out_tensor_info.GetShape();
        for (auto& d : out_shape)
        {
            if (d < 0)
                d = 1;
        }
        size_t out_elem_count = 1;
        for (auto d : out_shape)
            out_elem_count *= static_cast<size_t>(d);
        size_t out_byte_size = out_elem_count * sizeof(float);

        // Create D3D12 GPU buffers for inputs and output
        std::vector<ComPtr<ID3D12Resource>> gpu_inputs(num_inputs);
        std::vector<ComPtr<ID3D12Resource>> upload_buffers(num_inputs);
#if ORT_API_VERSION >= 26
        // "Corrupt" upload buffers (9.0f) consumed by the stress loop below. They only matter
        // when the importer fences are present (v26+): with a fence missing, inference reads
        // 9.0f instead of 1.0f and the output is no longer 4.0f.
        std::vector<ComPtr<ID3D12Resource>> corrupt_buffers(num_inputs);
#endif
        for (size_t i = 0; i < num_inputs; ++i)
        {
            CreateD3D12Buffer(pDevice.Get(), input_byte_sizes[i], gpu_inputs[i].GetAddressOf(),
                              D3D12_RESOURCE_STATE_COPY_DEST);
            CreateUploadBuffer(pDevice.Get(), input_byte_sizes[i], upload_buffers[i].GetAddressOf());

            // Fill upload buffer with the real input data (1.0f)
            void* pData = nullptr;
            upload_buffers[i]->Map(0, nullptr, &pData);
            memcpy(pData, cpu_inputs[i].data(), input_byte_sizes[i]);
            upload_buffers[i]->Unmap(0, nullptr);

#if ORT_API_VERSION >= 26
            CreateUploadBuffer(pDevice.Get(), input_byte_sizes[i], corrupt_buffers[i].GetAddressOf());
            void* pCorrupt = nullptr;
            corrupt_buffers[i]->Map(0, nullptr, &pCorrupt);
            std::vector<float> corrupt_vals(input_byte_sizes[i] / sizeof(float), 9.0f);
            memcpy(pCorrupt, corrupt_vals.data(), input_byte_sizes[i]);
            corrupt_buffers[i]->Unmap(0, nullptr);
#endif
        }

        ComPtr<ID3D12Resource> gpu_output;
        CreateD3D12Buffer(pDevice.Get(), out_byte_size, gpu_output.GetAddressOf(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        ComPtr<ID3D12Resource> readback_buffer;
        CreateReadBackBuffer(pDevice.Get(), out_byte_size, readback_buffer.GetAddressOf());

        // Upload inputs via D3D12 command list
        ComPtr<ID3D12CommandAllocator> pCmdAllocator;
        pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&pCmdAllocator));

        ComPtr<ID3D12GraphicsCommandList> pUploadCmdList;
        pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COMPUTE, pCmdAllocator.Get(), nullptr,
                                   IID_PPV_ARGS(&pUploadCmdList));
        for (size_t i = 0; i < num_inputs; ++i)
        {
#if ORT_API_VERSION >= 26
            // Stress the synchronization: enqueue many serialized copies of the corrupt data
            // (UAV barriers stop the driver coalescing them) so the upload takes real GPU time.
            // With the importer fences in place these finish before inference and the readback;
            // without a fence, inference (or the readback) races this load and the output != 4.0f.
            // Mirrors the SimpleDXInterop CIG reference reproducer.
            for (int k = 0; k < 1000; ++k)
            {
                pUploadCmdList->CopyResource(gpu_inputs[i].Get(), corrupt_buffers[i].Get());
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.UAV.pResource = nullptr;
                pUploadCmdList->ResourceBarrier(1, &barrier);
            }
#endif
            pUploadCmdList->CopyResource(gpu_inputs[i].Get(), upload_buffers[i].Get());
        }
        pUploadCmdList->Close();

        // Fence values for the two D3D12<->CUDA hand-offs.
        enum FenceState
        {
            FENCE_UPLOAD_DONE = 1,
            FENCE_KERNEL_DONE = 2
        };

        ID3D12CommandList* uploadList = pUploadCmdList.Get();
        pCommandQueue->ExecuteCommandLists(1, &uploadList);
#if ORT_API_VERSION >= 26
        if (use_semaphore_sync)
        {
            // Upload->inference hand-off: D3D12 signals the fence after the upload completes,
            // CUDA waits on it before the inference reads the inputs.
            pCommandQueue->Signal(sync_fence.Get(), FENCE_UPLOAD_DONE);
            status = interop_api.WaitSemaphore(importer, sync_sem, stream, FENCE_UPLOAD_DONE);
            ASSERT_EQ(status, nullptr) << "WaitSemaphore (upload) failed";
        }
        else
#endif
        {
            FlushAndWait(pDevice.Get(), pCommandQueue.Get());
        }

        // Create GPU tensors from D3D12 virtual addresses using Device_Agnostic memory info
        const OrtHardwareDevice* hw_device = api.EpDevice_Device(ep_device);
        uint32_t vendor_id = api.HardwareDevice_VendorId(hw_device);

        OrtMemoryInfo* gpu_mem_info = nullptr;
        status = api.CreateMemoryInfo_V2("Device_Agnostic", OrtMemoryInfoDeviceType_GPU, vendor_id, 0,
                                         OrtDeviceMemoryType_DEFAULT, 0, OrtArenaAllocator, &gpu_mem_info);
        ASSERT_EQ(status, nullptr) << "CreateMemoryInfo_V2 failed";

        // Bind GPU tensors and run inference
        Ort::IoBinding io_binding(session);

        std::vector<Ort::Value> gpu_input_tensors;
        for (size_t i = 0; i < num_inputs; ++i)
        {
            gpu_input_tensors.push_back(Ort::Value::CreateTensor(
                gpu_mem_info, reinterpret_cast<void*>(gpu_inputs[i]->GetGPUVirtualAddress()), input_byte_sizes[i],
                input_shapes[i].data(), input_shapes[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

            io_binding.BindInput(input_names_str[i].c_str(), gpu_input_tensors.back());
        }

        Ort::Value gpu_output_tensor = Ort::Value::CreateTensor(
            gpu_mem_info, reinterpret_cast<void*>(gpu_output->GetGPUVirtualAddress()), out_byte_size, out_shape.data(),
            out_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        auto output_name = session.GetOutputNameAllocated(0, allocator);
        io_binding.BindOutput(output_name.get(), gpu_output_tensor);

        Ort::RunOptions run_options;
        run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        session.Run(run_options, io_binding);

#if ORT_API_VERSION >= 26
        if (use_semaphore_sync)
        {
            // Inference->download hand-off: CUDA signals the fence after inference completes,
            // the D3D12 queue waits on it before the readback copy. This closes the
            // "CUDA finished before Dx downloads" gap that CPU FlushAndWait alone does not.
            status = interop_api.SignalSemaphore(importer, sync_sem, stream, FENCE_KERNEL_DONE);
            ASSERT_EQ(status, nullptr) << "SignalSemaphore (inference) failed";
            pCommandQueue->Wait(sync_fence.Get(), FENCE_KERNEL_DONE);
        }
        else
#endif
        {
            // No importer semaphore (pre-v26 or importer-creation failure). The EP did not
            // synchronize internally (disable_synchronize_execution_providers=1), so block on the
            // host until the bound outputs are ready before the D3D12 readback reads gpu_output.
            io_binding.SynchronizeOutputs();
        }

        // Download output via D3D12 command list
        ComPtr<ID3D12GraphicsCommandList> pDownloadCmdList;
        pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COMPUTE, pCmdAllocator.Get(), nullptr,
                                   IID_PPV_ARGS(&pDownloadCmdList));
        pDownloadCmdList->CopyResource(readback_buffer.Get(), gpu_output.Get());
        pDownloadCmdList->Close();

        ID3D12CommandList* downloadList = pDownloadCmdList.Get();
        pCommandQueue->ExecuteCommandLists(1, &downloadList);
        FlushAndWait(pDevice.Get(), pCommandQueue.Get());

        // Read back and validate: O = ((X+Y)+Z)+S = (1+1+1+1) = 4.0f
        void* pOutputData = nullptr;
        readback_buffer->Map(0, nullptr, &pOutputData);
        const float* result = static_cast<const float*>(pOutputData);
        for (size_t i = 0; i < out_elem_count; ++i)
        {
            EXPECT_NEAR(result[i], 4.0f, 1e-3f) << "Output mismatch at index " << i;
        }
        readback_buffer->Unmap(0, nullptr);

        api.ReleaseMemoryInfo(gpu_mem_info);
    }

    // Cleanup
    api.ReleaseSyncStream(stream);
#if ORT_API_VERSION >= 26
    if (sync_sem != nullptr)
    {
        interop_api.ReleaseExternalSemaphoreHandle(sync_sem);
    }
    if (importer != nullptr)
    {
        interop_api.ReleaseExternalResourceImporter(importer);
    }
    if (shared_fence_handle != nullptr)
    {
        CloseHandle(shared_fence_handle);
    }
#endif
    status = interop_api.DeinitGraphicsInteropForEpDevice(ep_device);
    ASSERT_EQ(status, nullptr) << "DeinitGraphicsInterop failed";

    clearFileIfExists(model_name);
}

TEST(CigInteropTest, D3D12ImportedResourceFullInferenceWithoutCig)
{
    RunD3D12ImportedResourceFullInference(false);
}

TEST(CigInteropTest, D3D12ImportedResourceFullInferenceWithCig)
{
    RunD3D12ImportedResourceFullInference(true);
}

#endif  // ORT_API_VERSION >= 25
#endif  // _WIN32
