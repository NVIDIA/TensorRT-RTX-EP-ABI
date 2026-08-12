// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validates Vulkan <-> CUDA external resource import for the NvTensorRTRTX EP:
// importer creation, memory/semaphore capability checks, Vulkan memory import,
// tensor creation from imported memory, timeline semaphore import, and async
// wait/signal semaphore operations.

#include <cuda_runtime.h>

#include "test_tensorrt_rtx_utils.h"
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_SMART_HANDLE
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <cstdint>
#include <cstring>
#include <optional>
#include <regex>
#include <vector>

#include <vulkan/vulkan.hpp>

#if defined(_WIN32)
#include <windows.h>
#endif

extern std::unique_ptr<Ort::Env> ort_env;

#if ORT_API_VERSION >= 26

namespace
{

struct NvVkDevice
{
    VkPhysicalDevice physical_device{};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceVulkan11Properties id_properties{};
    std::vector<Ort::ConstEpDevice> ep_devices;
};

struct ExportableBuffer
{
    VkBuffer buffer{};
    VkBufferView view{};
    VkDeviceMemory memory{};
    void* native_handle{};
    OrtExternalMemoryHandle* ort_handle{};
};

struct ExportableTimelineSemaphore
{
    VkSemaphore semaphore{};
    void* native_handle{};
    OrtExternalSemaphoreHandle* ort_handle{};
};

class VulkanExternalResourceFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (ort_env.get() == nullptr)
        {
            return;
        }

        interop_api_ = &Ort::GetInteropApi();
        loader_.init();

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "trt_rtx_ep_vk_external_resource_test";
        app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.pEngineName = "trt_rtx_ep_vk_external_resource_test";
        app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        if (loader_.vkCreateInstance(&instance_info, nullptr, &instance_) != VK_SUCCESS)
        {
            return;
        }
        loader_.init(vk::Instance(instance_));

        uint32_t physical_device_count = 0;
        if (loader_.vkEnumeratePhysicalDevices(instance_, &physical_device_count, nullptr) != VK_SUCCESS ||
            physical_device_count == 0)
        {
            return;
        }
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        if (loader_.vkEnumeratePhysicalDevices(instance_, &physical_device_count, physical_devices.data()) !=
            VK_SUCCESS)
        {
            return;
        }

        auto ep_devices = get_trt_rtx_devices(*ort_env);
#if !defined(_WIN32)
        std::regex pci_bus_id_pattern("([a-fA-F0-9]+):([a-fA-F0-9]+):([a-fA-F0-9]+)\\.([a-fA-F0-9]+)");
#endif

        for (VkPhysicalDevice physical_device : physical_devices)
        {
            NvVkDevice nv_device;
            nv_device.physical_device = physical_device;

            VkPhysicalDevicePCIBusInfoPropertiesEXT pci_properties{};
            pci_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
            VkPhysicalDeviceVulkan11Properties id_properties{};
            id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
            id_properties.pNext = &pci_properties;
            VkPhysicalDeviceProperties2 properties2{};
            properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties2.pNext = &id_properties;
            loader_.vkGetPhysicalDeviceProperties2(physical_device, &properties2);
            if (properties2.properties.vendorID != 0x10DE)
            {
                continue;
            }

            nv_device.properties = properties2.properties;
            nv_device.id_properties = id_properties;

            for (const auto& ep_device : ep_devices)
            {
                if (ep_device.Device().VendorId() != properties2.properties.vendorID ||
                    ep_device.Device().DeviceId() != properties2.properties.deviceID)
                {
                    continue;
                }
#if defined(_WIN32)
                auto luid = ep_device.Device().Metadata().GetValue("LUID");
                if (id_properties.deviceLUIDValid && luid)
                {
                    LUID vk_luid{};
                    std::memcpy(&vk_luid, id_properties.deviceLUID, sizeof(LUID));
                    const uint64_t ep_luid = std::stoull(luid);
                    const uint64_t vk_luid_u64 = (uint64_t(vk_luid.HighPart) << 32) | uint64_t(vk_luid.LowPart);
                    if (ep_luid != vk_luid_u64)
                    {
                        continue;
                    }
                }
#else
                auto pci_bus_id = ep_device.Device().Metadata().GetValue("pci_bus_id");
                if (pci_bus_id)
                {
                    std::cmatch matches;
                    if (std::regex_match(pci_bus_id, matches, pci_bus_id_pattern))
                    {
                        const auto domain = std::stoull(matches[1].str(), nullptr, 16);
                        const auto bus = std::stoull(matches[2].str(), nullptr, 16);
                        const auto device = std::stoull(matches[3].str(), nullptr, 16);
                        const auto function = std::stoull(matches[4].str(), nullptr, 16);
                        if (domain != pci_properties.pciDomain || bus != pci_properties.pciBus ||
                            device != pci_properties.pciDevice || function != pci_properties.pciFunction)
                        {
                            continue;
                        }
                    }
                }
#endif
                nv_device.ep_devices.push_back(ep_device);
            }

            if (!nv_device.ep_devices.empty())
            {
                nv_devices_.push_back(std::move(nv_device));
            }
        }

        if (nv_devices_.empty())
        {
            return;
        }

        physical_device_ = nv_devices_[0].physical_device;
        ep_device_ = nv_devices_[0].ep_devices[0];

        auto queue_family_index = FindQueueFamily();
        if (!queue_family_index)
        {
            return;
        }
        queue_family_index_ = *queue_family_index;

        VkPhysicalDeviceVulkan11Features vulkan11_features{};
        vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        VkPhysicalDeviceVulkan12Features vulkan12_features{};
        vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12_features.timelineSemaphore = true;
        vulkan12_features.pNext = &vulkan11_features;

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        std::vector<const char*> device_extensions;
#if defined(_WIN32)
        device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
        device_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        device_info.ppEnabledExtensionNames = device_extensions.data();
        device_info.pNext = &vulkan12_features;

        if (loader_.vkCreateDevice(physical_device_, &device_info, nullptr, &device_) != VK_SUCCESS)
        {
            return;
        }
        loader_.init(vk::Device(device_));
        loader_.vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);
        vulkan_available_ = true;

        Ort::Status status(interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer_));
        if (!status.IsOK() || importer_ == nullptr)
        {
            return;
        }
        importer_available_ = true;
    }

    void TearDown() override
    {
        for (auto& semaphore : semaphores_)
        {
            if (semaphore.ort_handle != nullptr)
            {
                interop_api_->ReleaseExternalSemaphoreHandle(semaphore.ort_handle);
            }
        }
        for (auto& buffer : buffers_)
        {
            if (buffer.ort_handle != nullptr)
            {
                interop_api_->ReleaseExternalMemoryHandle(buffer.ort_handle);
            }
        }
        if (importer_ != nullptr)
        {
            interop_api_->ReleaseExternalResourceImporter(importer_);
        }
        if (device_ != VK_NULL_HANDLE && loader_.vkDeviceWaitIdle)
        {
            loader_.vkDeviceWaitIdle(device_);
            for (auto& buffer : buffers_)
            {
                if (buffer.view != VK_NULL_HANDLE)
                {
                    loader_.vkDestroyBufferView(device_, buffer.view, nullptr);
                }
                if (buffer.buffer != VK_NULL_HANDLE)
                {
                    loader_.vkDestroyBuffer(device_, buffer.buffer, nullptr);
                }
                if (buffer.memory != VK_NULL_HANDLE)
                {
                    loader_.vkFreeMemory(device_, buffer.memory, nullptr);
                }
            }
            for (auto& semaphore : semaphores_)
            {
                if (semaphore.semaphore != VK_NULL_HANDLE)
                {
                    loader_.vkDestroySemaphore(device_, semaphore.semaphore, nullptr);
                }
            }
            loader_.vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE && loader_.vkDestroyInstance)
        {
            loader_.vkDestroyInstance(instance_, nullptr);
        }
    }

    bool IsVulkanAvailable() const
    {
        return vulkan_available_;
    }

    std::optional<uint32_t> FindQueueFamily()
    {
        uint32_t count = 0;
        loader_.vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(count);
        loader_.vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &count, properties.data());
        for (uint32_t i = 0; i < count; ++i)
        {
            if ((properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                (properties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0)
            {
                return i;
            }
        }
        return std::nullopt;
    }

    ExportableBuffer& CreateExportableBuffer(VkDeviceSize size)
    {
        auto& buffer = buffers_.emplace_back();

        VkPhysicalDeviceMemoryProperties memory_properties{};
        loader_.vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);

#if defined(_WIN32)
        const auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        const auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

        VkExternalMemoryBufferCreateInfo external_buffer_info{};
        external_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external_buffer_info.handleTypes = handle_type;
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.queueFamilyIndexCount = 1;
        buffer_info.pQueueFamilyIndices = &queue_family_index_;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        buffer_info.size = size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
        buffer_info.pNext = &external_buffer_info;
        EXPECT_EQ(VK_SUCCESS, loader_.vkCreateBuffer(device_, &buffer_info, nullptr, &buffer.buffer));

        VkMemoryRequirements memory_requirements{};
        loader_.vkGetBufferMemoryRequirements(device_, buffer.buffer, &memory_requirements);
        int memory_type_index = -1;
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
        {
            if ((memory_requirements.memoryTypeBits & (1u << i)) != 0 &&
                (memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
            {
                memory_type_index = static_cast<int>(i);
                break;
            }
        }
        EXPECT_NE(memory_type_index, -1);

        VkExportMemoryAllocateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_info.handleTypes = handle_type;
        VkMemoryAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.memoryTypeIndex = static_cast<uint32_t>(memory_type_index);
        allocate_info.allocationSize = memory_requirements.size;
        allocate_info.pNext = &export_info;
        EXPECT_EQ(VK_SUCCESS, loader_.vkAllocateMemory(device_, &allocate_info, nullptr, &buffer.memory));
        EXPECT_EQ(VK_SUCCESS, loader_.vkBindBufferMemory(device_, buffer.buffer, buffer.memory, 0));

        VkBufferViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
        view_info.buffer = buffer.buffer;
        view_info.format = VK_FORMAT_R32_SFLOAT;
        view_info.offset = 0;
        view_info.range = size;
        EXPECT_EQ(VK_SUCCESS, loader_.vkCreateBufferView(device_, &view_info, nullptr, &buffer.view));

#if defined(_WIN32)
        VkMemoryGetWin32HandleInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handle_info.memory = buffer.memory;
        handle_info.handleType = handle_type;
        HANDLE native_handle = nullptr;
        EXPECT_EQ(VK_SUCCESS, loader_.vkGetMemoryWin32HandleKHR(device_, &handle_info, &native_handle));
        buffer.native_handle = native_handle;
#else
        VkMemoryGetFdInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        handle_info.memory = buffer.memory;
        handle_info.handleType = handle_type;
        int native_handle = -1;
        EXPECT_EQ(VK_SUCCESS, loader_.vkGetMemoryFdKHR(device_, &handle_info, &native_handle));
        buffer.native_handle = reinterpret_cast<void*>(static_cast<intptr_t>(native_handle));
#endif

        OrtExternalMemoryDescriptor memory_desc{};
        memory_desc.version = ORT_API_VERSION;
#if defined(_WIN32)
        memory_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32;
#else
        memory_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD;
#endif
        memory_desc.native_handle = buffer.native_handle;
        memory_desc.size_bytes = static_cast<size_t>(size);
        memory_desc.offset_bytes = 0;

        Ort::Status status(interop_api_->ImportMemory(importer_, &memory_desc, &buffer.ort_handle));
        EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
        return buffer;
    }

    ExportableTimelineSemaphore& CreateExportableTimelineSemaphore()
    {
        auto& semaphore = semaphores_.emplace_back();

        VkSemaphoreTypeCreateInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_info.initialValue = 0;
        VkExportSemaphoreCreateInfo export_info{};
        export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#if defined(_WIN32)
        export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        timeline_info.pNext = &export_info;
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphore_info.pNext = &timeline_info;
        EXPECT_EQ(VK_SUCCESS, loader_.vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore.semaphore));

        OrtExternalSemaphoreDescriptor semaphore_desc{};
        semaphore_desc.version = ORT_API_VERSION;
#if defined(_WIN32)
        VkSemaphoreGetWin32HandleInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        handle_info.semaphore = semaphore.semaphore;
        handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        HANDLE native_handle = nullptr;
        EXPECT_EQ(VK_SUCCESS, loader_.vkGetSemaphoreWin32HandleKHR(device_, &handle_info, &native_handle));
        semaphore.native_handle = native_handle;
        semaphore_desc.type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;
#else
        VkSemaphoreGetFdInfoKHR handle_info{};
        handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        handle_info.semaphore = semaphore.semaphore;
        handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        int native_handle = -1;
        EXPECT_EQ(VK_SUCCESS, loader_.vkGetSemaphoreFdKHR(device_, &handle_info, &native_handle));
        semaphore.native_handle = reinterpret_cast<void*>(static_cast<intptr_t>(native_handle));
        semaphore_desc.type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;
#endif
        semaphore_desc.native_handle = semaphore.native_handle;

        Ort::Status status(interop_api_->ImportSemaphore(importer_, &semaphore_desc, &semaphore.ort_handle));
        EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
        return semaphore;
    }

    vk::detail::DispatchLoaderDynamic loader_;
    VkInstance instance_{};
    VkDevice device_{};
    VkPhysicalDevice physical_device_{};
    VkQueue queue_{};
    uint32_t queue_family_index_ = 0;
    std::vector<NvVkDevice> nv_devices_;
    Ort::ConstEpDevice ep_device_{nullptr};
    const OrtInteropApi* interop_api_ = nullptr;
    OrtExternalResourceImporter* importer_ = nullptr;
    std::vector<ExportableBuffer> buffers_;
    std::vector<ExportableTimelineSemaphore> semaphores_;
    bool vulkan_available_ = false;
    bool importer_available_ = false;
};

}  // namespace

TEST_F(VulkanExternalResourceFixture, CreateExternalResourceImporter)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }
    ASSERT_TRUE(importer_available_);
}

TEST_F(VulkanExternalResourceFixture, CanImportMemoryCapabilities)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }

    bool can_import = false;
#if defined(_WIN32)
    auto handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32;
#else
    auto handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD;
#endif
    Ort::Status status(interop_api_->CanImportMemory(importer_, handle_type, &can_import));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_TRUE(can_import);
}

TEST_F(VulkanExternalResourceFixture, CanImportSemaphoreCapabilities)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }

    bool can_import = false;
#if defined(_WIN32)
    auto semaphore_type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;
#else
    auto semaphore_type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;
#endif
    Ort::Status status(interop_api_->CanImportSemaphore(importer_, semaphore_type, &can_import));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_TRUE(can_import);
}

TEST_F(VulkanExternalResourceFixture, ImportVulkanMemory)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }

    auto& buffer = CreateExportableBuffer(1024 * sizeof(float));
    ASSERT_NE(buffer.ort_handle, nullptr);
}

TEST_F(VulkanExternalResourceFixture, CreateTensorFromImportedMemory)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }

    const int64_t shape[] = {1, 3, 32, 32};
    const size_t buffer_size = 1 * 3 * 32 * 32 * sizeof(float);
    auto& buffer = CreateExportableBuffer(buffer_size);
    ASSERT_NE(buffer.ort_handle, nullptr);

    OrtExternalTensorDescriptor tensor_desc{};
    tensor_desc.version = ORT_API_VERSION;
    tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    tensor_desc.shape = shape;
    tensor_desc.rank = 4;
    tensor_desc.offset_bytes = 0;

    OrtValue* tensor = nullptr;
    Ort::Status status(interop_api_->CreateTensorFromMemory(importer_, buffer.ort_handle, &tensor_desc, &tensor));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_NE(tensor, nullptr);

    void* tensor_data = nullptr;
    ASSERT_EQ(nullptr, Ort::GetApi().GetTensorMutableData(tensor, &tensor_data));
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, tensor_data));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    Ort::GetApi().ReleaseValue(tensor);
}

TEST_F(VulkanExternalResourceFixture, ImportVulkanTimelineSemaphore)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }

    auto& semaphore = CreateExportableTimelineSemaphore();
    ASSERT_NE(semaphore.ort_handle, nullptr);
}

TEST_F(VulkanExternalResourceFixture, WaitAndSignalSemaphore)
{
    if (!IsVulkanAvailable())
    {
        GTEST_SKIP() << "Vulkan not available";
    }

    auto& semaphore = CreateExportableTimelineSemaphore();
    ASSERT_NE(semaphore.ort_handle, nullptr);

    auto stream = ep_device_.CreateSyncStream();
    constexpr uint64_t vulkan_signal_value = 1;
    VkSemaphoreSignalInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = semaphore.semaphore;
    signal_info.value = vulkan_signal_value;
    ASSERT_EQ(VK_SUCCESS, loader_.vkSignalSemaphore(device_, &signal_info));

    Ort::Status status(interop_api_->WaitSemaphore(importer_, semaphore.ort_handle, stream, vulkan_signal_value));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    constexpr uint64_t cuda_signal_value = 2;
    status = Ort::Status(interop_api_->SignalSemaphore(importer_, semaphore.ort_handle, stream, cuda_signal_value));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &semaphore.semaphore;
    wait_info.pValues = &cuda_signal_value;
    EXPECT_EQ(VK_SUCCESS, loader_.vkWaitSemaphores(device_, &wait_info, 5'000'000'000ULL));
}

#endif  // ORT_API_VERSION >= 26
