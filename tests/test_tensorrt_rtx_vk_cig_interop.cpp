// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_rtx_provider_options.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "test_tensorrt_rtx_cuda_driver_loader.h"
#include "test_tensorrt_rtx_model_builder.h"
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

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
    bool has_cig_extension = false;
};

struct ExportableTimelineSemaphore
{
    VkSemaphore semaphore{};
    void* native_handle{};
    OrtExternalSemaphoreHandle* ort_handle{};
};

struct ExportableBuffer
{
    VkBuffer buffer{};
    VkBufferView view{};
    VkDeviceMemory memory{};
    void* native_handle{};
    OrtExternalMemoryHandle* ort_handle{};
};

struct TestParameters
{
    bool force_cig_if_supported = false;
    bool use_init_graphics_interop_call = false;
    bool allow_manual_cuda_ctx = true;
};

struct VkResources
{
    vk::detail::DispatchLoaderDynamic loader;
    VkInstance instance{};
    VkDevice device{};
    VkPhysicalDevice physical_device{};
    VkQueue queue{};
    uint32_t queue_family_index = 0;
    VkExternalComputeQueueNV external_compute_queue{};
    OrtExternalResourceImporter* importer{};
    VkCommandPool command_pool{};
    VkCommandBuffer upload_cmd_buffer{};
    VkCommandBuffer download_cmd_buffer{};
    std::vector<NvVkDevice> nv_devices;
    std::vector<ExportableTimelineSemaphore> semaphores;
    std::vector<ExportableBuffer> buffers;
    std::optional<Ort::ConstEpDevice> ep_device;
    bool graphics_interop_initialized = false;

    void ReleaseOrtResources()
    {
        if (device != VK_NULL_HANDLE && loader.vkDeviceWaitIdle)
        {
            loader.vkDeviceWaitIdle(device);
        }

        auto& interop_api = Ort::GetInteropApi();
        for (auto& semaphore : semaphores)
        {
            if (semaphore.ort_handle != nullptr)
            {
                interop_api.ReleaseExternalSemaphoreHandle(semaphore.ort_handle);
                semaphore.ort_handle = nullptr;
            }
        }
        for (auto& buffer : buffers)
        {
            if (buffer.ort_handle != nullptr)
            {
                interop_api.ReleaseExternalMemoryHandle(buffer.ort_handle);
                buffer.ort_handle = nullptr;
            }
        }
        if (importer != nullptr)
        {
            interop_api.ReleaseExternalResourceImporter(importer);
            importer = nullptr;
        }
    }

    ~VkResources()
    {
        ReleaseOrtResources();

        if (graphics_interop_initialized && ep_device)
        {
            auto& interop_api = Ort::GetInteropApi();
            Ort::Status status(interop_api.DeinitGraphicsInteropForEpDevice(*ep_device));
        }

        if (device != VK_NULL_HANDLE && loader.vkDeviceWaitIdle)
        {
            loader.vkDeviceWaitIdle(device);
            for (auto& buffer : buffers)
            {
                if (buffer.view != VK_NULL_HANDLE)
                {
                    loader.vkDestroyBufferView(device, buffer.view, nullptr);
                }
                if (buffer.buffer != VK_NULL_HANDLE)
                {
                    loader.vkDestroyBuffer(device, buffer.buffer, nullptr);
                }
                if (buffer.memory != VK_NULL_HANDLE)
                {
                    loader.vkFreeMemory(device, buffer.memory, nullptr);
                }
            }
            for (auto& semaphore : semaphores)
            {
                if (semaphore.semaphore != VK_NULL_HANDLE)
                {
                    loader.vkDestroySemaphore(device, semaphore.semaphore, nullptr);
                }
            }
            if (external_compute_queue != VK_NULL_HANDLE)
            {
                loader.vkDestroyExternalComputeQueueNV(device, external_compute_queue, nullptr);
            }
            if (command_pool != VK_NULL_HANDLE)
            {
                if (upload_cmd_buffer != VK_NULL_HANDLE || download_cmd_buffer != VK_NULL_HANDLE)
                {
                    std::vector<VkCommandBuffer> command_buffers;
                    if (upload_cmd_buffer != VK_NULL_HANDLE)
                    {
                        command_buffers.push_back(upload_cmd_buffer);
                    }
                    if (download_cmd_buffer != VK_NULL_HANDLE)
                    {
                        command_buffers.push_back(download_cmd_buffer);
                    }
                    loader.vkFreeCommandBuffers(device, command_pool, static_cast<uint32_t>(command_buffers.size()),
                                                command_buffers.data());
                }
                loader.vkDestroyCommandPool(device, command_pool, nullptr);
            }
            loader.vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE && loader.vkDestroyInstance)
        {
            loader.vkDestroyInstance(instance, nullptr);
        }
    }
};

void CreateAddOneModel(const std::filesystem::path& path, const std::vector<int64_t>& shape)
{
    onnx::ModelProto model;
    model.set_ir_version(7);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    auto* graph = model.mutable_graph();
    graph->set_name("vk_external_mem_add_one");
    std::vector<int> shape_int(shape.begin(), shape.end());
    model_builder::AddValueInfo(graph->mutable_input(), "X", onnx::TensorProto_DataType_FLOAT, shape_int);
    model_builder::AddValueInfo(graph->mutable_output(), "Y", onnx::TensorProto_DataType_FLOAT, shape_int);

    auto* one = graph->add_initializer();
    one->set_name("one");
    one->set_data_type(onnx::TensorProto_DataType_FLOAT);
    one->add_dims(1);
    one->add_float_data(1.0f);

    model_builder::AddNode(graph, "add_one", "Add", {"X", "one"}, {"Y"});
    model_builder::SaveModel(model, path.string());
}

std::optional<uint32_t> FindQueueFamily(VkResources& resources, VkPhysicalDevice physical_device)
{
    uint32_t count = 0;
    resources.loader.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    resources.loader.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, properties.data());
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

void InitVulkanInterop(VkResources& resources)
{
    resources.loader.init();

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "trt_rtx_ep_vk_test";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "ORT";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkResult result = resources.loader.vkCreateInstance(&instance_info, nullptr, &resources.instance);
    if (result != VK_SUCCESS)
    {
        GTEST_SKIP() << "Vulkan instance creation failed";
    }
    resources.loader.init(vk::Instance(resources.instance));

    uint32_t physical_device_count = 0;
    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkEnumeratePhysicalDevices(resources.instance, &physical_device_count, nullptr));
    if (physical_device_count == 0)
    {
        GTEST_SKIP() << "No Vulkan physical devices found";
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    ASSERT_EQ(VK_SUCCESS, resources.loader.vkEnumeratePhysicalDevices(resources.instance, &physical_device_count,
                                                                      physical_devices.data()));

    auto ep_devices = get_trt_rtx_devices(*ort_env);
#if !defined(_WIN32)
    std::regex pci_bus_id_pattern("([a-fA-F0-9]+):([a-fA-F0-9]+):([a-fA-F0-9]+)\\.([a-fA-F0-9]+)");
#endif

    for (auto& physical_device : physical_devices)
    {
        NvVkDevice nv_device;
        nv_device.physical_device = physical_device;

        uint32_t extension_count = 0;
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkEnumerateDeviceExtensionProperties(physical_device, nullptr,
                                                                                    &extension_count, nullptr));
        std::vector<VkExtensionProperties> extensions(extension_count);
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkEnumerateDeviceExtensionProperties(
                                  physical_device, nullptr, &extension_count, extensions.data()));
        bool has_pci_bus_info = false;
        for (const auto& extension : extensions)
        {
            if (std::strcmp(extension.extensionName, VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME) == 0)
            {
                nv_device.has_cig_extension = true;
            }
            if (std::strcmp(extension.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0)
            {
                has_pci_bus_info = true;
            }
        }

        VkPhysicalDevicePCIBusInfoPropertiesEXT pci_properties{};
        pci_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
        VkPhysicalDeviceVulkan11Properties id_properties{};
        id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        id_properties.pNext = has_pci_bus_info ? &pci_properties : nullptr;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &id_properties;
        resources.loader.vkGetPhysicalDeviceProperties2(physical_device, &properties2);
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
            if (has_pci_bus_info && pci_bus_id)
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
            resources.nv_devices.push_back(nv_device);
        }
    }

    if (resources.nv_devices.empty())
    {
        GTEST_SKIP() << "No NVIDIA Vulkan device matching a TRT RTX EP device was found";
    }

    resources.physical_device = resources.nv_devices[0].physical_device;
    auto queue_family_index = FindQueueFamily(resources, resources.physical_device);
    if (!queue_family_index)
    {
        GTEST_SKIP() << "No Vulkan compute/transfer queue family found";
    }
    resources.queue_family_index = *queue_family_index;
    const bool has_cig_extension = resources.nv_devices[0].has_cig_extension;

    VkPhysicalDeviceVulkan11Features vulkan11_features{};
    vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features vulkan12_features{};
    vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12_features.bufferDeviceAddress = true;
    vulkan12_features.timelineSemaphore = true;
    vulkan12_features.pNext = &vulkan11_features;

    VkExternalComputeQueueDeviceCreateInfoNV cig_create_info{};
    cig_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV;
    cig_create_info.reservedExternalQueues = 1;
    cig_create_info.pNext = &vulkan12_features;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = resources.queue_family_index;
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
    if (has_cig_extension)
    {
        device_extensions.push_back(VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME);
    }

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.data();
    device_info.pNext =
        has_cig_extension ? static_cast<void*>(&cig_create_info) : static_cast<void*>(&vulkan12_features);

    result = resources.loader.vkCreateDevice(resources.physical_device, &device_info, nullptr, &resources.device);
    if (result != VK_SUCCESS)
    {
        GTEST_SKIP() << "Vulkan device creation failed";
    }
    resources.loader.init(vk::Device(resources.device));
    resources.loader.vkGetDeviceQueue(resources.device, resources.queue_family_index, 0, &resources.queue);

    if (has_cig_extension)
    {
        VkExternalComputeQueueCreateInfoNV external_queue_info{};
        external_queue_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV;
        external_queue_info.preferredQueue = resources.queue;
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkCreateExternalComputeQueueNV(
                                  resources.device, &external_queue_info, nullptr, &resources.external_compute_queue));
    }
}

void CreateTimelineSemaphore(VkResources& resources, ExportableTimelineSemaphore& semaphore)
{
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
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
    semaphore_info.pNext = &timeline_info;
    timeline_info.pNext = &export_info;

    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkCreateSemaphore(resources.device, &semaphore_info, nullptr, &semaphore.semaphore));

    OrtExternalSemaphoreDescriptor semaphore_desc{};
    semaphore_desc.version = ORT_API_VERSION;
#if defined(_WIN32)
    VkSemaphoreGetWin32HandleInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.semaphore = semaphore.semaphore;
    handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE native_handle = nullptr;
    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkGetSemaphoreWin32HandleKHR(resources.device, &handle_info, &native_handle));
    semaphore_desc.type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;
    semaphore.native_handle = native_handle;
#else
    VkSemaphoreGetFdInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    handle_info.semaphore = semaphore.semaphore;
    handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int native_handle = -1;
    ASSERT_EQ(VK_SUCCESS, resources.loader.vkGetSemaphoreFdKHR(resources.device, &handle_info, &native_handle));
    semaphore_desc.type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;
    semaphore.native_handle = reinterpret_cast<void*>(static_cast<intptr_t>(native_handle));
#endif
    semaphore_desc.native_handle = semaphore.native_handle;

    Ort::Status status(
        Ort::GetInteropApi().ImportSemaphore(resources.importer, &semaphore_desc, &semaphore.ort_handle));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

void AllocateBuffer(VkResources& resources, ExportableBuffer& export_buffer, VkDeviceSize size,
                    VkMemoryPropertyFlags memory_property_flags, bool export_memory)
{
    VkPhysicalDeviceMemoryProperties memory_properties{};
    resources.loader.vkGetPhysicalDeviceMemoryProperties(resources.physical_device, &memory_properties);

#if defined(_WIN32)
    const auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    const auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.queueFamilyIndexCount = 1;
    buffer_info.pQueueFamilyIndices = &resources.queue_family_index;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    VkExternalMemoryBufferCreateInfo external_buffer_info{};
    external_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_buffer_info.handleTypes = handle_type;
    if (export_memory)
    {
        buffer_info.pNext = &external_buffer_info;
    }

    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkCreateBuffer(resources.device, &buffer_info, nullptr, &export_buffer.buffer));

    VkMemoryRequirements memory_requirements{};
    resources.loader.vkGetBufferMemoryRequirements(resources.device, export_buffer.buffer, &memory_requirements);
    int memory_type_index = -1;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
    {
        if ((memory_requirements.memoryTypeBits & (1u << i)) != 0 &&
            (memory_properties.memoryTypes[i].propertyFlags & memory_property_flags) == memory_property_flags)
        {
            memory_type_index = static_cast<int>(i);
            break;
        }
    }
    ASSERT_NE(memory_type_index, -1);

    VkExportMemoryAllocateInfo export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_info.handleTypes = handle_type;
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.memoryTypeIndex = static_cast<uint32_t>(memory_type_index);
    allocate_info.allocationSize = memory_requirements.size;
    if (export_memory)
    {
        allocate_info.pNext = &export_info;
    }

    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkAllocateMemory(resources.device, &allocate_info, nullptr, &export_buffer.memory));
    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkBindBufferMemory(resources.device, export_buffer.buffer, export_buffer.memory, 0));

    VkBufferViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    view_info.buffer = export_buffer.buffer;
    view_info.format = VK_FORMAT_R32_SFLOAT;
    view_info.offset = 0;
    view_info.range = size;
    ASSERT_EQ(VK_SUCCESS,
              resources.loader.vkCreateBufferView(resources.device, &view_info, nullptr, &export_buffer.view));

    if (!export_memory)
    {
        return;
    }

#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.memory = export_buffer.memory;
    handle_info.handleType = handle_type;
    HANDLE native_handle = nullptr;
    ASSERT_EQ(VK_SUCCESS, resources.loader.vkGetMemoryWin32HandleKHR(resources.device, &handle_info, &native_handle));
    export_buffer.native_handle = native_handle;
#else
    VkMemoryGetFdInfoKHR handle_info{};
    handle_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handle_info.memory = export_buffer.memory;
    handle_info.handleType = handle_type;
    int native_handle = -1;
    ASSERT_EQ(VK_SUCCESS, resources.loader.vkGetMemoryFdKHR(resources.device, &handle_info, &native_handle));
    export_buffer.native_handle = reinterpret_cast<void*>(static_cast<intptr_t>(native_handle));
#endif

    OrtExternalMemoryDescriptor memory_desc{};
    memory_desc.version = ORT_API_VERSION;
#if defined(_WIN32)
    memory_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32;
#else
    memory_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD;
#endif
    memory_desc.native_handle = export_buffer.native_handle;
    memory_desc.size_bytes = static_cast<size_t>(size);
    memory_desc.offset_bytes = 0;

    Ort::Status status(Ort::GetInteropApi().ImportMemory(resources.importer, &memory_desc, &export_buffer.ort_handle));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

Ort::Session ConfigureSession(const std::filesystem::path& model_path, Ort::SyncStream& stream,
                              const Ort::ConstEpDevice& ep_device, bool use_cig)
{
    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.DisableMemPattern();
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

    Ort::KeyValuePairs ep_options;
    ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kUserComputeStream,
                   std::to_string(reinterpret_cast<size_t>(stream.GetHandle())).c_str());
    ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kHasUserComputeStream, "1");
    if (use_cig)
    {
        // For a simple unit test we pick 48K since this is the common lowest denominator
        // https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/compute-graphics.html#shared-memory-limitation
        ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kMaxSharedMemSize,
                       std::to_string(48 * 1024).c_str());
        // disable aux streams
        ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kLengthAuxStreamArray, "0");
        ep_options.Add(onnxruntime::tensorrt_rtx::provider_option_names::kCudaGraphEnable, "0");
    }
    session_options.AppendExecutionProvider_V2(*ort_env, std::vector{ep_device}, ep_options);

    auto model_path_string = toOrtString(model_path);
    return Ort::Session(*ort_env, model_path_string.c_str(), session_options);
}

void TestVulkanInterop(const TestParameters& test_parameters)
{
    ASSERT_NE(ort_env.get(), nullptr);

    VkResources resources;
    InitVulkanInterop(resources);

    ASSERT_FALSE(resources.nv_devices.empty());
    ASSERT_FALSE(resources.nv_devices[0].ep_devices.empty());
    const auto& ep_device = resources.nv_devices[0].ep_devices[0];
    resources.ep_device = ep_device;
    const OrtEpDevice* raw_ep_device = static_cast<const OrtEpDevice*>(ep_device);

    const int ep_api_version = ep_negotiated_ort_api_version(raw_ep_device);
    if (ep_api_version >= 0 && ep_api_version < 26)
    {
        GTEST_SKIP() << "EP DLL negotiated ORT API version " << ep_api_version
                     << "; external resource import API requires >= 26.";
    }

    auto& interop_api = Ort::GetInteropApi();
    Ort::Status status(interop_api.CreateExternalResourceImporterForDevice(ep_device, &resources.importer));
    if (!status.IsOK() || resources.importer == nullptr)
    {
        GTEST_FAIL() << "External resource import not supported: "
                     << (status.IsOK() ? "<null importer>" : status.GetErrorMessage());
    }

#if defined(_WIN32)
    constexpr auto vk_memory_handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_WIN32;
    constexpr auto vk_semaphore_type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_WIN32;
#else
    constexpr auto vk_memory_handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD;
    constexpr auto vk_semaphore_type = ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD;
#endif

    bool can_import_memory = false;
    status = Ort::Status(interop_api.CanImportMemory(resources.importer, vk_memory_handle_type, &can_import_memory));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_TRUE(can_import_memory) << "VK external buffer import not supported";

    bool can_import_semaphore = false;
    status = Ort::Status(interop_api.CanImportSemaphore(resources.importer, vk_semaphore_type, &can_import_semaphore));
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    ASSERT_TRUE(can_import_semaphore) << "VK external timeline semaphore import not supported";

    const bool has_cig_extension = resources.nv_devices[0].has_cig_extension;
    int cuda_device_count = 0;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&cuda_device_count));
    ASSERT_GT(cuda_device_count, 0);

    int selected_device = -1;
    for (int i = 0; i < cuda_device_count; ++i)
    {
        cudaDeviceProp properties{};
        ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&properties, i));
        if (std::memcmp(&properties.uuid, resources.nv_devices[0].id_properties.deviceUUID,
                        sizeof(resources.nv_devices[0].id_properties.deviceUUID)) == 0)
        {
            selected_device = i;
            break;
        }
    }

    int cig_supported = 0;
    ASSERT_EQ(cudaSuccess, cudaDeviceGetAttribute(&cig_supported, cudaDevAttrVulkanCigSupported, selected_device));

    std::vector<uint8_t> external_compute_queue_data(64);
    if (has_cig_extension)
    {
        VkExternalComputeQueueDataParamsNV external_queue_params{};
        external_queue_params.sType = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV;
        external_queue_params.deviceIndex = 0;
        resources.loader.vkGetExternalComputeQueueDataNV(resources.external_compute_queue, &external_queue_params,
                                                         external_compute_queue_data.data());
    }

    CUcontext manual_cig_context = nullptr;
    CudaDriverLoader driver;
    ASSERT_TRUE(driver.IsLoaded()) << "CUDA driver API was not available";

    if (test_parameters.force_cig_if_supported)
    {
        if (!has_cig_extension)
        {
            GTEST_SKIP() << "VK_NV_external_compute_queue is not available";
        }
        if (cig_supported == 0)
        {
            GTEST_SKIP() << "CUDA device does not support Vulkan CIG";
        }
        if (test_parameters.use_init_graphics_interop_call)
        {

            Ort::KeyValuePairs key_values;
            if (test_parameters.force_cig_if_supported)
            {
                key_values.Add(onnxruntime::tensorrt_rtx::provider_option_names::kExternalComputeQueueDataParamNV_data,
                               std::to_string(reinterpret_cast<uintptr_t>(external_compute_queue_data.data())).c_str());
            }

            OrtGraphicsInteropConfig interop_config{};
            interop_config.version = ORT_API_VERSION;
            interop_config.graphics_api = ORT_GRAPHICS_API_VULKAN;
            interop_config.command_queue = nullptr;
            interop_config.additional_options = key_values.GetConst();

            status = Ort::Status(interop_api.InitGraphicsInteropForEpDevice(ep_device, &interop_config));
            ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
            resources.graphics_interop_initialized = true;
        }
        else if (test_parameters.allow_manual_cuda_ctx)
        {
            CUctxCigParam cig_params{};
            cig_params.sharedDataType = CIG_DATA_TYPE_NV_BLOB;
            cig_params.sharedData = external_compute_queue_data.data();
            CUctxCreateParams context_params{};
            context_params.cigParams = test_parameters.force_cig_if_supported ? &cig_params : nullptr;
            ASSERT_EQ(CUDA_SUCCESS, driver.cuCtxCreate_v4_fn(&manual_cig_context, &context_params, 0, selected_device));
            ASSERT_EQ(CUDA_SUCCESS, driver.cuCtxSetCurrent_fn(manual_cig_context));
        }
    }

    const std::filesystem::path model_path =
        std::filesystem::temp_directory_path() / "trt_rtx_vk_external_mem_add_one.onnx";
    clearFileIfExists(model_path);
    constexpr int64_t batch = 1;
    constexpr int64_t channels = 3;
    constexpr int64_t dim = 64;
    std::vector<int64_t> shape = {batch, channels, dim, dim};
    CreateAddOneModel(model_path, shape);
    {

        const size_t num_elements = static_cast<size_t>(batch * channels * dim * dim);
        const size_t buffer_size = num_elements * sizeof(float);

        ExportableTimelineSemaphore input_ready{};
        CreateTimelineSemaphore(resources, input_ready);
        resources.semaphores.push_back(input_ready);
        ExportableTimelineSemaphore inference_done{};
        CreateTimelineSemaphore(resources, inference_done);
        resources.semaphores.push_back(inference_done);
        ExportableTimelineSemaphore download_done{};
        CreateTimelineSemaphore(resources, download_done);
        resources.semaphores.push_back(download_done);

        ExportableBuffer upload_buffer{};
        // This buffer is mapped for CPU upload/readback without explicit
        // vkFlushMappedMemoryRanges/vkInvalidateMappedMemoryRanges, calls.
        // For simplicity coherent host memory is used so that sequential CPU<->GPU hand-offs are memory-visible.
        AllocateBuffer(resources, upload_buffer, buffer_size,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
        resources.buffers.push_back(upload_buffer);
        ExportableBuffer input_buffer{};
        AllocateBuffer(resources, input_buffer, buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
        resources.buffers.push_back(input_buffer);
        ExportableBuffer output_buffer{};
        AllocateBuffer(resources, output_buffer, buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
        resources.buffers.push_back(output_buffer);

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = resources.queue_family_index;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ASSERT_EQ(VK_SUCCESS,
                  resources.loader.vkCreateCommandPool(resources.device, &pool_info, nullptr, &resources.command_pool));

        VkCommandBufferAllocateInfo command_buffer_info{};
        command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_info.commandPool = resources.command_pool;
        command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_info.commandBufferCount = 2;
        VkCommandBuffer command_buffers[2]{};
        ASSERT_EQ(VK_SUCCESS,
                  resources.loader.vkAllocateCommandBuffers(resources.device, &command_buffer_info, command_buffers));
        resources.upload_cmd_buffer = command_buffers[0];
        resources.download_cmd_buffer = command_buffers[1];

        auto& upload_buffer_ref = resources.buffers[0];
        auto& input_buffer_ref = resources.buffers[1];
        auto& output_buffer_ref = resources.buffers[2];
        auto& input_ready_semaphore = resources.semaphores[0];
        auto& inference_done_semaphore = resources.semaphores[1];
        auto& download_done_semaphore = resources.semaphores[2];

        void* mapped = nullptr;
        ASSERT_EQ(VK_SUCCESS,
                  resources.loader.vkMapMemory(resources.device, upload_buffer_ref.memory, 0, buffer_size, 0, &mapped));
        float* host_data = static_cast<float*>(mapped);
        for (size_t i = 0; i < num_elements; ++i)
        {
            host_data[i] = static_cast<float>(i);
        }
        resources.loader.vkUnmapMemory(resources.device, upload_buffer_ref.memory);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkBeginCommandBuffer(resources.upload_cmd_buffer, &begin_info));
        VkBufferCopy copy_region{};
        copy_region.size = buffer_size;
        resources.loader.vkCmdCopyBuffer(resources.upload_cmd_buffer, upload_buffer_ref.buffer, input_buffer_ref.buffer,
                                         1, &copy_region);
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkEndCommandBuffer(resources.upload_cmd_buffer));

        OrtExternalTensorDescriptor tensor_desc{};
        tensor_desc.version = ORT_API_VERSION;
        tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        tensor_desc.shape = shape.data();
        tensor_desc.rank = 4;
        tensor_desc.offset_bytes = 0;

        OrtValue* input_tensor = nullptr;
        status = Ort::Status(interop_api.CreateTensorFromMemory(resources.importer, input_buffer_ref.ort_handle,
                                                                &tensor_desc, &input_tensor));
        ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
        OrtValue* output_tensor = nullptr;
        status = Ort::Status(interop_api.CreateTensorFromMemory(resources.importer, output_buffer_ref.ort_handle,
                                                                &tensor_desc, &output_tensor));
        ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

        void* input_data = nullptr;
        void* output_data = nullptr;
        ASSERT_EQ(nullptr, Ort::GetApi().GetTensorMutableData(input_tensor, &input_data));
        ASSERT_EQ(nullptr, Ort::GetApi().GetTensorMutableData(output_tensor, &output_data));
        cudaPointerAttributes input_attributes{};
        cudaPointerAttributes output_attributes{};
        ASSERT_EQ(cudaSuccess, cudaPointerGetAttributes(&input_attributes, input_data));
        ASSERT_EQ(cudaSuccess, cudaPointerGetAttributes(&output_attributes, output_data));
        ASSERT_EQ(input_attributes.type, cudaMemoryTypeDevice);
        ASSERT_EQ(output_attributes.type, cudaMemoryTypeDevice);

        auto stream = ep_device.CreateSyncStream();
        auto session = ConfigureSession(model_path, stream, ep_device, test_parameters.force_cig_if_supported);

        Ort::IoBinding io_binding(session);
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        io_binding.BindInput(input_name.get(), Ort::Value(input_tensor));
        io_binding.BindOutput(output_name.get(), Ort::Value(output_tensor));
        io_binding.SynchronizeInputs();

        constexpr uint64_t input_ready_value = 1;
        constexpr uint64_t inference_done_value = 1;
        constexpr uint64_t download_done_value = 1;

        ASSERT_EQ(VK_SUCCESS, resources.loader.vkBeginCommandBuffer(resources.download_cmd_buffer, &begin_info));
        resources.loader.vkCmdCopyBuffer(resources.download_cmd_buffer, output_buffer_ref.buffer,
                                         upload_buffer_ref.buffer, 1, &copy_region);
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkEndCommandBuffer(resources.download_cmd_buffer));

        uint64_t signal_input_ready = input_ready_value;
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_input_ready;

        VkSubmitInfo upload_submit{};
        upload_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        upload_submit.pNext = &timeline_info;
        upload_submit.commandBufferCount = 1;
        upload_submit.pCommandBuffers = &resources.upload_cmd_buffer;
        upload_submit.signalSemaphoreCount = 1;
        upload_submit.pSignalSemaphores = &input_ready_semaphore.semaphore;
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkQueueSubmit(resources.queue, 1, &upload_submit, VK_NULL_HANDLE));

        status = Ort::Status(
            interop_api.WaitSemaphore(resources.importer, input_ready_semaphore.ort_handle, stream, input_ready_value));
        ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

        Ort::RunOptions run_options;
        run_options.SetSyncStream(stream);
        run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        session.Run(run_options, io_binding);

        status = Ort::Status(interop_api.SignalSemaphore(resources.importer, inference_done_semaphore.ort_handle,
                                                         stream, inference_done_value));
        ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

        uint64_t wait_inference_done = inference_done_value;
        uint64_t signal_download_done = download_done_value;
        timeline_info.waitSemaphoreValueCount = 1;
        timeline_info.pWaitSemaphoreValues = &wait_inference_done;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_download_done;

        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo download_submit{};
        download_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        download_submit.pNext = &timeline_info;
        download_submit.waitSemaphoreCount = 1;
        download_submit.pWaitDstStageMask = &wait_stage;
        download_submit.pWaitSemaphores = &inference_done_semaphore.semaphore;
        download_submit.commandBufferCount = 1;
        download_submit.pCommandBuffers = &resources.download_cmd_buffer;
        download_submit.signalSemaphoreCount = 1;
        download_submit.pSignalSemaphores = &download_done_semaphore.semaphore;
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkQueueSubmit(resources.queue, 1, &download_submit, VK_NULL_HANDLE));

        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &download_done_semaphore.semaphore;
        wait_info.pValues = &download_done_value;
        ASSERT_EQ(VK_SUCCESS, resources.loader.vkWaitSemaphores(resources.device, &wait_info, UINT64_MAX));

        mapped = nullptr;
        ASSERT_EQ(VK_SUCCESS,
                  resources.loader.vkMapMemory(resources.device, upload_buffer_ref.memory, 0, buffer_size, 0, &mapped));
        host_data = static_cast<float*>(mapped);
        for (size_t i = 0; i < num_elements; ++i)
        {
            EXPECT_FLOAT_EQ(host_data[i], static_cast<float>(i) + 1.0f) << "index " << i;
        }
        resources.loader.vkUnmapMemory(resources.device, upload_buffer_ref.memory);
    }
    clearFileIfExists(model_path);
    // Teardown order matters for CIG/imported external resources:
    // - Ort::Session/IoBinding/Ort::Value
    // - pending Vulkan queue work must be idle
    // - destroy ORT-imported semaphores/memory/importer
    // - CUDA (CIG) context can be destroyed
    // - VK objects devices can be destroyued
    resources.ReleaseOrtResources();
    if (manual_cig_context != nullptr)
    {
        ASSERT_EQ(CUDA_SUCCESS, driver.cuCtxSetCurrent_fn(nullptr));
        ASSERT_EQ(CUDA_SUCCESS, driver.cuCtxDestroy_fn(manual_cig_context));
    }
}

}  // namespace

TEST(VkCigInteropTest, VkCigDisabled)
{
    TestParameters parameters;
    parameters.force_cig_if_supported = false;
    parameters.use_init_graphics_interop_call = false;
    TestVulkanInterop(parameters);
}

TEST(VkCigInteropTest, VkCigEnabled)
{
    TestParameters parameters;
    parameters.force_cig_if_supported = true;
    parameters.use_init_graphics_interop_call = false;
    TestVulkanInterop(parameters);
}

TEST(VkCigInteropTest, VkInitGraphicsInterop)
{
    TestParameters parameters;
    parameters.force_cig_if_supported = false;
    parameters.use_init_graphics_interop_call = true;
    TestVulkanInterop(parameters);
}

TEST(VkCigInteropTest, VkInitGraphicsInteropNoManualCudaCtx)
{
    TestParameters parameters;
    parameters.force_cig_if_supported = false;
    parameters.use_init_graphics_interop_call = true;
    parameters.allow_manual_cuda_ctx = false;
    TestVulkanInterop(parameters);
}

TEST(VkCigInteropTest, VkInitGraphicsInteropCig)
{
    TestParameters parameters;
    parameters.force_cig_if_supported = true;
    parameters.use_init_graphics_interop_call = true;
    TestVulkanInterop(parameters);
}

#endif  // ORT_API_VERSION >= 26
