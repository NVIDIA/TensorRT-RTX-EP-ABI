// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validates D3D12 <-> CUDA external resource import for the NvTensorRTRTX EP:
// importer creation, memory/semaphore capability checks, D3D12 shared resource
// import, tensor creation from imported memory, D3D12 timeline-fence import, and
// async wait/signal semaphore operations. Ported from microsoft/onnxruntime PR
// #26948 (nv_external_resource_importer_test.cc), adapted to this plugin's test
// harness (dynamic d3d12 load + get_trt_rtx_devices). Requires the external
// resource importer interop API added in ORT API v26.

#include <cuda_runtime.h>

#include <vector>

#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#if defined(_WIN32)
#include <d3d12.h>
#include <Windows.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

extern std::unique_ptr<Ort::Env> ort_env;

// All cases require the importer interop API (ORT API v26) and D3D12 (Windows).
#if defined(_WIN32) && (ORT_API_VERSION >= 26)

namespace
{

// d3d12.dll is loaded dynamically (not linked) so that linking DXGI/D3D12 into
// unittests.exe does not perturb CUDA state in unrelated tests.
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
    // Load DXGI first — D3D12 depends on it for adapter enumeration.
    r.dxgi_module = LoadLibraryW(L"dxgi.dll");
    r.module = LoadLibraryW(L"d3d12.dll");
    if (r.module)
    {
        r.pfn = reinterpret_cast<D3D12CreateDeviceLoadResult::PFN_D3D12CreateDevice>(
            GetProcAddress(r.module, "D3D12CreateDevice"));
    }
    return r;
}

// Creates a D3D12 default-heap buffer with the SHARED flag so it can be imported
// across APIs (D3D12 -> CUDA) via a shared NT handle.
void CreateSharedBuffer(ID3D12Device* device, size_t size, ID3D12Resource** out_resource,
                        D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COMMON)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    HRESULT hr = device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_SHARED, &desc, initial_state, nullptr,
                                                 IID_PPV_ARGS(out_resource));
    if (FAILED(hr))
    {
        GTEST_FAIL() << "Failed to create shared D3D12 buffer, HRESULT: 0x" << std::hex << hr;
    }
}

}  // namespace

// Test fixture: sets up a D3D12 device + compute command queue and resolves the
// registered NvTensorRTRTX EP device. Tests skip gracefully if either is absent.
class Direct3DExternalResourceFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ort_api_ = &Ort::GetApi();
        ort_interop_api_ = &Ort::GetInteropApi();

        D3D12CreateDeviceLoadResult d3d12 = LoadD3D12CreateDevice();
        if (!d3d12.pfn)
        {
            return;
        }

        if (FAILED(d3d12.pfn(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device_))))
        {
            return;
        }

        D3D12_COMMAND_QUEUE_DESC queue_desc = {};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        if (FAILED(d3d12_device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue_))))
        {
            return;
        }
        d3d12_available_ = true;

        if (ort_env.get() == nullptr)
        {
            return;
        }
        auto devices = get_trt_rtx_devices(*ort_env);
        if (devices.empty())
        {
            return;
        }
        ep_device_ = static_cast<const OrtEpDevice*>(devices[0]);
        ep_available_ = (ep_device_ != nullptr);
    }

    bool IsD3D12Available() const
    {
        return d3d12_available_;
    }

    ComPtr<ID3D12Device> d3d12_device_;
    ComPtr<ID3D12CommandQueue> command_queue_;
    const OrtApi* ort_api_ = nullptr;
    const OrtInteropApi* ort_interop_api_ = nullptr;
    const OrtEpDevice* ep_device_ = nullptr;
    bool d3d12_available_ = false;
    bool ep_available_ = false;
};

// Test: External resource importer creation and destruction.
TEST_F(Direct3DExternalResourceFixture, CreateExternalResourceImporter)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    if (status != nullptr)
    {
        std::string error = ort_api_->GetErrorMessage(status);
        ort_api_->ReleaseStatus(status);
        GTEST_FAIL() << "CreateExternalResourceImporterForDevice not supported: " << error;
    }
    ASSERT_NE(importer, nullptr) << "Importer should not be null";

    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Memory import capability check (D3D12 Resource & Heap).
TEST_F(Direct3DExternalResourceFixture, CanImportMemoryCapabilities)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    ASSERT_EQ(status, nullptr);
    ASSERT_NE(importer, nullptr);

    bool can_import_resource = false;
    status = ort_interop_api_->CanImportMemory(importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
                                               &can_import_resource);
    ASSERT_EQ(status, nullptr) << "CanImportMemory for D3D12_RESOURCE should succeed";
    EXPECT_TRUE(can_import_resource) << "Should support D3D12 Resource import";

    bool can_import_heap = false;
    status = ort_interop_api_->CanImportMemory(importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, &can_import_heap);
    ASSERT_EQ(status, nullptr) << "CanImportMemory for D3D12_HEAP should succeed";
    EXPECT_TRUE(can_import_heap) << "Should support D3D12 Heap import";

    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Semaphore import capability check (D3D12 Fence).
TEST_F(Direct3DExternalResourceFixture, CanImportSemaphoreCapabilities)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    ASSERT_EQ(status, nullptr);
    ASSERT_NE(importer, nullptr);

    bool can_import_fence = false;
    status = ort_interop_api_->CanImportSemaphore(importer, ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE, &can_import_fence);
    ASSERT_EQ(status, nullptr) << "CanImportSemaphore for D3D12_FENCE should succeed";
    EXPECT_TRUE(can_import_fence) << "Should support D3D12 Fence import";

    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Import a D3D12 shared resource into CUDA.
TEST_F(Direct3DExternalResourceFixture, ImportD3D12SharedResource)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    ASSERT_EQ(status, nullptr);
    ASSERT_NE(importer, nullptr);

    const size_t buffer_size = 1024 * sizeof(float);
    ComPtr<ID3D12Resource> d3d12_buffer;
    CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &d3d12_buffer);

    HANDLE shared_handle = nullptr;
    HRESULT hr = d3d12_device_->CreateSharedHandle(d3d12_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
    ASSERT_TRUE(SUCCEEDED(hr)) << "Failed to create shared handle";

    OrtExternalMemoryDescriptor mem_desc = {};
    mem_desc.version = ORT_API_VERSION;
    mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    mem_desc.native_handle = shared_handle;
    mem_desc.size_bytes = buffer_size;
    mem_desc.offset_bytes = 0;

    OrtExternalMemoryHandle* mem_handle = nullptr;
    status = ort_interop_api_->ImportMemory(importer, &mem_desc, &mem_handle);
    ASSERT_EQ(status, nullptr) << "ImportMemory should succeed (proves cuImportExternalMemory called)";
    ASSERT_NE(mem_handle, nullptr) << "Memory handle should not be null";

    ort_interop_api_->ReleaseExternalMemoryHandle(mem_handle);
    CloseHandle(shared_handle);
    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Create an ORT tensor from imported external memory and verify it is CUDA
// device memory (proving the D3D12 -> CUDA import actually happened).
TEST_F(Direct3DExternalResourceFixture, CreateTensorFromImportedMemory)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    ASSERT_EQ(status, nullptr);
    ASSERT_NE(importer, nullptr);

    const int64_t batch = 1, channels = 3, height = 32, width = 32;
    const int64_t shape[] = {batch, channels, height, width};
    const size_t num_elements = static_cast<size_t>(batch * channels * height * width);
    const size_t buffer_size = num_elements * sizeof(float);

    ComPtr<ID3D12Resource> d3d12_buffer;
    CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &d3d12_buffer);

    HANDLE shared_handle = nullptr;
    HRESULT hr = d3d12_device_->CreateSharedHandle(d3d12_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
    ASSERT_TRUE(SUCCEEDED(hr));

    OrtExternalMemoryDescriptor mem_desc = {};
    mem_desc.version = ORT_API_VERSION;
    mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    mem_desc.native_handle = shared_handle;
    mem_desc.size_bytes = buffer_size;
    mem_desc.offset_bytes = 0;

    OrtExternalMemoryHandle* mem_handle = nullptr;
    status = ort_interop_api_->ImportMemory(importer, &mem_desc, &mem_handle);
    ASSERT_EQ(status, nullptr);

    OrtExternalTensorDescriptor tensor_desc = {};
    tensor_desc.version = ORT_API_VERSION;
    tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    tensor_desc.shape = shape;
    tensor_desc.rank = 4;
    tensor_desc.offset_bytes = 0;

    OrtValue* tensor = nullptr;
    status = ort_interop_api_->CreateTensorFromMemory(importer, mem_handle, &tensor_desc, &tensor);
    ASSERT_EQ(status, nullptr) << "CreateTensorFromMemory should succeed";
    ASSERT_NE(tensor, nullptr) << "Tensor should not be null";

    OrtTensorTypeAndShapeInfo* type_info = nullptr;
    status = ort_api_->GetTensorTypeAndShape(tensor, &type_info);
    ASSERT_EQ(status, nullptr);

    size_t rank = 0;
    ort_api_->GetDimensionsCount(type_info, &rank);
    EXPECT_EQ(rank, 4u);

    std::vector<int64_t> actual_shape(rank);
    ort_api_->GetDimensions(type_info, actual_shape.data(), rank);
    EXPECT_EQ(actual_shape[0], batch);
    EXPECT_EQ(actual_shape[1], channels);
    EXPECT_EQ(actual_shape[2], height);
    EXPECT_EQ(actual_shape[3], width);

    ONNXTensorElementDataType elem_type;
    ort_api_->GetTensorElementType(type_info, &elem_type);
    EXPECT_EQ(elem_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    ort_api_->ReleaseTensorTypeAndShapeInfo(type_info);

    // The tensor's data pointer must be CUDA device memory — proves the import worked.
    void* tensor_data = nullptr;
    status = ort_api_->GetTensorMutableData(tensor, &tensor_data);
    ASSERT_EQ(status, nullptr) << "GetTensorMutableData should succeed";
    ASSERT_NE(tensor_data, nullptr) << "Tensor data pointer should not be null";

    cudaPointerAttributes attrs;
    cudaError_t cuda_err = cudaPointerGetAttributes(&attrs, tensor_data);
    ASSERT_EQ(cuda_err, cudaSuccess) << "cudaPointerGetAttributes failed: " << cudaGetErrorString(cuda_err);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice) << "Memory should be CUDA device memory, got type " << attrs.type;
    EXPECT_NE(attrs.device, -1) << "Device should be valid";

    ort_api_->ReleaseValue(tensor);
    ort_interop_api_->ReleaseExternalMemoryHandle(mem_handle);
    CloseHandle(shared_handle);
    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Import a D3D12 timeline fence as a CUDA external semaphore.
TEST_F(Direct3DExternalResourceFixture, ImportD3D12Fence)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    ASSERT_EQ(status, nullptr);
    ASSERT_NE(importer, nullptr);

    ComPtr<ID3D12Fence> d3d12_fence;
    HRESULT hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3d12_fence));
    ASSERT_TRUE(SUCCEEDED(hr)) << "Failed to create D3D12 fence";

    HANDLE shared_handle = nullptr;
    hr = d3d12_device_->CreateSharedHandle(d3d12_fence.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
    ASSERT_TRUE(SUCCEEDED(hr)) << "Failed to create shared fence handle";

    OrtExternalSemaphoreDescriptor sem_desc = {};
    sem_desc.version = ORT_API_VERSION;
    sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
    sem_desc.native_handle = shared_handle;

    OrtExternalSemaphoreHandle* sem_handle = nullptr;
    status = ort_interop_api_->ImportSemaphore(importer, &sem_desc, &sem_handle);
    ASSERT_EQ(status, nullptr) << "ImportSemaphore should succeed (proves cuImportExternalSemaphore called)";
    ASSERT_NE(sem_handle, nullptr) << "Semaphore handle should not be null";

    ort_interop_api_->ReleaseExternalSemaphoreHandle(sem_handle);
    CloseHandle(shared_handle);
    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Async wait/signal on an imported D3D12 fence across the D3D12 queue and the
// CUDA stream — exercises cuWaitExternalSemaphoresAsync / cuSignalExternalSemaphoresAsync.
TEST_F(Direct3DExternalResourceFixture, WaitAndSignalSemaphore)
{
    if (!IsD3D12Available())
    {
        GTEST_SKIP() << "D3D12 not available";
    }

    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    ASSERT_EQ(status, nullptr);
    ASSERT_NE(importer, nullptr);

    OrtSyncStream* ort_stream = nullptr;
    status = ort_api_->CreateSyncStreamForEpDevice(ep_device_, nullptr, &ort_stream);
    ASSERT_EQ(status, nullptr) << "CreateSyncStreamForEpDevice should succeed";

    ComPtr<ID3D12Fence> d3d12_fence;
    HRESULT hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3d12_fence));
    ASSERT_TRUE(SUCCEEDED(hr));

    HANDLE shared_handle = nullptr;
    hr = d3d12_device_->CreateSharedHandle(d3d12_fence.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
    ASSERT_TRUE(SUCCEEDED(hr));

    OrtExternalSemaphoreDescriptor sem_desc = {};
    sem_desc.version = ORT_API_VERSION;
    sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
    sem_desc.native_handle = shared_handle;

    OrtExternalSemaphoreHandle* sem_handle = nullptr;
    status = ort_interop_api_->ImportSemaphore(importer, &sem_desc, &sem_handle);
    ASSERT_EQ(status, nullptr);

    // Signal the fence from the D3D12 queue, then wait on it from the CUDA stream.
    const uint64_t signal_value = 1;
    command_queue_->Signal(d3d12_fence.Get(), signal_value);

    status = ort_interop_api_->WaitSemaphore(importer, sem_handle, ort_stream, signal_value);
    ASSERT_EQ(status, nullptr) << "WaitSemaphore should succeed";

    // Signal a new value from the CUDA stream; D3D12 should observe it.
    const uint64_t cuda_signal_value = 2;
    status = ort_interop_api_->SignalSemaphore(importer, sem_handle, ort_stream, cuda_signal_value);
    ASSERT_EQ(status, nullptr) << "SignalSemaphore should succeed";

    void* stream_handle = ort_api_->SyncStream_GetHandle(ort_stream);
    ASSERT_NE(stream_handle, nullptr);

    HANDLE wait_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ASSERT_NE(wait_event, nullptr);
    d3d12_fence->SetEventOnCompletion(cuda_signal_value, wait_event);
    DWORD wait_result = WaitForSingleObject(wait_event, 5000);  // 5s timeout
    CloseHandle(wait_event);
    EXPECT_EQ(wait_result, WAIT_OBJECT_0) << "D3D12 should see the fence value signaled by CUDA";

    ort_interop_api_->ReleaseExternalSemaphoreHandle(sem_handle);
    CloseHandle(shared_handle);
    ort_api_->ReleaseSyncStream(ort_stream);
    ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

#endif  // _WIN32 && ORT_API_VERSION >= 26
