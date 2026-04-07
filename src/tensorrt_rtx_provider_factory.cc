// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_rtx_provider_factory.h"

#include "tensorrt_rtx_execution_provider.h"
#include "tensorrt_rtx_execution_provider_data_transfer.h"
#include "tensorrt_rtx_execution_provider_stream_support.h"
#include "tensorrt_rtx_execution_provider_custom_ops.h"
#include "tensorrt_rtx_allocator.h"
#include "tensorrt_rtx_provider_options.h"
#include "cuda_mempool_arena.h"
#include "ep_arena.h"
#include "utils/cuda/cuda_common.h"
#include "kernel_registration.h"

#include "onnxruntime_cxx_api.h"

#include <cuda_runtime.h>

// NVML for driver version checking
#include <nvml.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace trt_rtx_ep
{

namespace
{
bool TryHexStringToBinary(const char* hex, std::vector<uint8_t>& out, std::string& error_message)
{
    if (hex == nullptr)
    {
        error_message = "null engine header string";
        return false;
    }

    size_t len = std::strlen(hex);
    if ((len % 2) != 0)
    {
        error_message = "hex string must have even length";
        return false;
    }

    out.clear();
    out.reserve(len / 2);

    auto nibble = [](char c, uint8_t& value) -> bool
    {
        if (c >= '0' && c <= '9')
        {
            value = static_cast<uint8_t>(c - '0');
            return true;
        }
        if (c >= 'a' && c <= 'f')
        {
            value = static_cast<uint8_t>(c - 'a' + 10);
            return true;
        }
        if (c >= 'A' && c <= 'F')
        {
            value = static_cast<uint8_t>(c - 'A' + 10);
            return true;
        }
        return false;
    };

    for (size_t i = 0; i < len; i += 2)
    {
        uint8_t high = 0;
        uint8_t low = 0;
        if (!nibble(hex[i], high) || !nibble(hex[i + 1], low))
        {
            error_message = "hex string contains non-hex characters";
            return false;
        }
        out.push_back(static_cast<uint8_t>((high << 4) | low));
    }

    return true;
}
}  // namespace

//
// Factory Constructor - Sets up OrtEpFactory interface function pointers
//
TensorrtRtxExecutionProviderFactory::TensorrtRtxExecutionProviderFactory(const char* ep_name,
                                                                         const OrtLogger& default_logger,
                                                                         ApiPtrs apis)
    : OrtEpFactory{}, ApiPtrs(apis), default_logger_{default_logger}, ep_name_{ep_name}
{
   // Set OrtEpFactory interface function pointers
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
    GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
    GetCustomOpDomains = GetCustomOpDomainsImpl;
    ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
    GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

    // Register custom operations (FP4/FP8 quantization) with ONNX Runtime.
    // These operations are recognized by ONNX Runtime but executed by TensorRT's inference engine.
    // Registration must succeed for the EP to function properly with quantized models.
    auto status = InitializeCustomOpDomains();
    if (status != nullptr)
    {
        // Extract error details and log before releasing the status object.
        const char* error_msg = ort_api.GetErrorMessage(status);
        ort_api.Logger_LogMessage(&default_logger, ORT_LOGGING_LEVEL_ERROR,
                                   error_msg ? error_msg : "Failed to initialize custom op domains",
                                   ORT_FILE, __LINE__, __FUNCTION__);
        ort_api.ReleaseStatus(status);
        // Factory creation must fail if custom operations cannot be registered.
        throw std::runtime_error("Failed to initialize custom op domains for TensorRT RTX EP");
    }
}

TensorrtRtxExecutionProviderFactory::~TensorrtRtxExecutionProviderFactory() noexcept
{
    // Release custom op domains
    for (auto* domain : custom_op_domains_)
    {
        if (domain != nullptr)
        {
            try
            {
                Ort::GetApi().ReleaseCustomOpDomain(domain);
            }
            catch (...)
            {

            }
        }
    }
    custom_op_domains_.clear();
    
    if (kernel_registry_ != nullptr)
    {
        try
        {
            Ort::GetEpApi().ReleaseKernelRegistry(kernel_registry_);
        }
        catch (...)
        {
            // Cannot propagate exceptions from destructor
        }
    }
}

OrtStatus* TensorrtRtxExecutionProviderFactory::ShrinkCudaMempoolAllocators(uint32_t device_id)
{
    auto it_mempool = device_mempool_allocators.find(device_id);
    if (it_mempool != device_mempool_allocators.end())
    {
        RETURN_IF_ERROR(it_mempool->second->Shrink());
    }

    return nullptr;
}

OrtStatus* TensorrtRtxExecutionProviderFactory::GetKernelRegistryForEp(
    const OrtKernelRegistry** out_kernel_registry)
{
    *out_kernel_registry = nullptr;

    if (GetNumKernels() == 0)
    {
        return nullptr;
    }

    if (kernel_registry_ == nullptr)
    {
        void* op_kernel_state = nullptr;  // Optional state that is provided to kernels on creation (can be null).
        const char* ep_name = ep_name_.c_str();

        // This statement creates the kernel registry and caches it in the OrtEpFactory instance.
        // We assume that all EPs created by this factory can use the same kernel registry.
        RETURN_IF_ERROR(CreateKernelRegistry(ep_name, op_kernel_state, &kernel_registry_));
    }

    *out_kernel_registry = kernel_registry_;
    return nullptr;
}

const char* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return nullptr;
    }
    const auto* factory = static_cast<const TensorrtRtxExecutionProviderFactory*>(this_ptr);
    return factory->ep_name_.c_str();
}

const char* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return nullptr;
    }
    const auto* factory = static_cast<const TensorrtRtxExecutionProviderFactory*>(this_ptr);
    return factory->vendor_.c_str();
}

const char* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return nullptr;
    }
    const auto* factory = static_cast<const TensorrtRtxExecutionProviderFactory*>(this_ptr);
    return factory->ep_version_.c_str();
}

uint32_t ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return 0;
    }
    const auto* factory = static_cast<const TensorrtRtxExecutionProviderFactory*>(this_ptr);
    return factory->vendor_id_;
}

OrtStatus* TensorrtRtxExecutionProviderFactory::CreateMemoryInfoForDevices(int num_devices)
{
    // Create memory info objects for each device
    for (int device_id = 0; device_id < num_devices; ++device_id)
    {
        // Create device memory info (OrtDeviceMemoryType_DEFAULT)
        OrtMemoryInfo* device_memory_info = nullptr;
        RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2(
            "TensorRTRTX",
            OrtMemoryInfoDeviceType_GPU,
            vendor_id_,
            device_id,
            OrtDeviceMemoryType_DEFAULT,
            0,  // alignment
            OrtAllocatorType::OrtDeviceAllocator,
            &device_memory_info));

        device_memory_infos[device_id] = MemoryInfoUniquePtr(
            device_memory_info,
            [this](OrtMemoryInfo* ptr)
            {
                ort_api.ReleaseMemoryInfo(ptr);
            });

        // Create pinned/host-accessible memory info (OrtDeviceMemoryType_HOST_ACCESSIBLE)
        OrtMemoryInfo* pinned_memory_info = nullptr;
        RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2(
            "TensorRTRTX host accessible",
            OrtMemoryInfoDeviceType_GPU,
            vendor_id_,
            device_id,
            OrtDeviceMemoryType_HOST_ACCESSIBLE,
            0,  // alignment
            OrtAllocatorType::OrtDeviceAllocator,
            &pinned_memory_info));

        pinned_memory_infos[device_id] = MemoryInfoUniquePtr(
            pinned_memory_info,
            [this](OrtMemoryInfo* ptr)
            {
                ort_api.ReleaseMemoryInfo(ptr);
            });
    }

    return nullptr;
}

//!
//! \brief Checks if a given OrtHardwareDevice is a supported NVIDIA GPU.
//!
//! This function verifies if the provided hardware device corresponds to a physical
//! NVIDIA GPU that meets the minimum compute capability requirements for this execution provider.
//!
//! The check is performed by:
//! 1. Extracting the LUID (Locally Unique Identifier) from the device's metadata.
//! 2. Converting the string LUID to a 64-bit integer.
//! 3. Iterating through all available CUDA devices on the system.
//! 4. For each CUDA device, constructing its 64-bit LUID from its properties.
//! 5. Comparing the LUIDs. If a match is found, it checks if the device's
//!    compute capability is at least 8.0 (Ampere) or newer.
//!
//! \param device The OrtHardwareDevice to check.
//! \param ort_api The ORT API interface for accessing device metadata.
//! \param major Optional output parameter for the major compute capability version.
//! \param minor Optional output parameter for the minor compute capability version.
//! \return True if the device is a supported NVIDIA GPU, false otherwise.
//!
static bool IsOrtHardwareDeviceSupported(const OrtHardwareDevice* device, const OrtApi& ort_api, int* major = nullptr, int* minor = nullptr)
{
#if defined(_WIN32)
    // Get device metadata using API
    const OrtKeyValuePairs* metadata = ort_api.HardwareDevice_Metadata(device);
    const char* luid_str = ort_api.GetKeyValue(metadata, "LUID");

    if (luid_str == nullptr)
    {
        return false;
    }

    uint64_t target_luid;
    try
    {
        target_luid = std::stoull(luid_str);
    }
    catch (const std::exception&)
    {
        return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess)
    {
        return false;
    }

    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
        {
            continue;
        }

        // The LUID is an 8-byte value, valid on Windows when luidDeviceNodeMask is non-zero.
        // We reconstruct the 64-bit integer representation from the raw bytes.
        if (prop.luidDeviceNodeMask == 0)
        {
            continue;
        }

        // Ensure the LUID is 8 bytes and reinterpret it directly as a uint64_t for comparison.
        static_assert(sizeof(prop.luid) == sizeof(uint64_t), "cudaDeviceProp::luid should be 8 bytes");
        uint64_t current_luid = *reinterpret_cast<const uint64_t*>(prop.luid);

        if (current_luid == target_luid)
        {
            // Set output parameters if provided
            if (major != nullptr)
            {
                *major = prop.major;
            }
            if (minor != nullptr)
            {
                *minor = prop.minor;
            }
            // Ampere architecture or newer is required.
            return prop.major >= 8;
        }
    }

    return false;
#else
    // Linux: Use PCI bus ID
    const OrtKeyValuePairs* metadata = ort_api.HardwareDevice_Metadata(device);
    const char* pci_bus_id = ort_api.GetKeyValue(metadata, "pci_bus_id");

    if (pci_bus_id == nullptr)
    {
        return false;
    }

    int cuda_device_idx = 0;
    if (cudaDeviceGetByPCIBusId(&cuda_device_idx, pci_bus_id) != cudaSuccess)
    {
        return false;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, cuda_device_idx) != cudaSuccess)
    {
        return false;
    }

    // Set output parameters if provided
    if (major != nullptr)
    {
        *major = prop.major;
    }
    if (minor != nullptr)
    {
        *minor = prop.minor;
    }
    // Ampere architecture or newer is required.
    return prop.major >= 8;
#endif
}

//!
//! \brief Parses an NVML driver version string and compares it with a minimum required version.
//!
//! NVML returns "major.minor" on Windows (e.g., "581.80") and "major.minor.patch" on Linux
//! (e.g., "550.54.14"). The patch component is ignored for comparison purposes since the
//! minimum required versions are specified as "major.minor" only.
//!
//! If either version string cannot be parsed, the function returns false (driver treated as
//! incompatible) to fail safely. A lexicographic string-comparison fallback would be
//! incorrect for version numbers (e.g., "9.0" >= "555.85" is lexicographically true but
//! numerically false).
//!
//! \param driver_version_str NVML driver version string (e.g., "581.80" or "550.54.14")
//! \param min_version_str Minimum required version in "major.minor" format (e.g., "570.00")
//! \return true if driver_version >= min_version, false otherwise or if parsing fails
//!
static bool CompareNVMLDriverVersion(const std::string& driver_version_str, const std::string& min_version_str)
{
    // Helper lambda to parse a version string of the form "major.minor[.patch]".
    // Only major and minor are extracted; an optional patch component is ignored.
    // Returns false if the string does not contain at least a valid "major.minor" prefix.
    auto parseVersion = [](const std::string& version_str, int& major, int& minor) -> bool
    {
        // Locate the first dot separating major from minor.
        size_t first_dot = version_str.find('.');
        if (first_dot == std::string::npos || first_dot == 0 || first_dot == version_str.length() - 1)
        {
            return false;
        }

        // Locate an optional second dot that begins the patch component.
        size_t second_dot = version_str.find('.', first_dot + 1);

        // Bound the minor field explicitly between the two dots (or end of string).
        // Do NOT rely on std::stoi silently truncating "54.14" to 54 — extract the
        // substring first so the intent is clear and stoi can validate the field.
        std::string minor_str = (second_dot == std::string::npos)
                                    ? version_str.substr(first_dot + 1)
                                    : version_str.substr(first_dot + 1, second_dot - (first_dot + 1));
        if (minor_str.empty())
        {
            return false;
        }

        try
        {
            major = std::stoi(version_str.substr(0, first_dot));
            minor = std::stoi(minor_str);
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    };

    int driver_major = 0, driver_minor = 0;
    int min_major = 0, min_minor = 0;

    // If either version string cannot be parsed, fail safely by reporting the driver as
    // incompatible. A lexicographic fallback is incorrect for version numbers and could
    // allow an unsupported driver through.
    if (!parseVersion(driver_version_str, driver_major, driver_minor) ||
        !parseVersion(min_version_str, min_major, min_minor))
    {
        return false;
    }

    // Compare versions numerically: major first, then minor.
    if (driver_major > min_major) return true;
    if (driver_major < min_major) return false;
    return driver_minor >= min_minor;
}

//!
//! \brief Checks for hardware device incompatibility reasons with TensorRT RTX EP.
//!
//! This function is called by ORT's GetHardwareDeviceEpIncompatibilityDetails() API
//! to provide diagnostic information about why a device may be incompatible with
//! this execution provider.
//!
//! Currently checks:
//! - Compute capability: Requires Ampere (8.0) or newer GPU architecture
//! - Driver version: Uses NVML to check NVIDIA graphics driver version
//!   - Ampere/Ada (CC 8.x): Requires >= 555.85
//!   - Blackwell (CC 12.x): Requires >= 570.00
//!
//! \param this_ptr The OrtEpFactory instance.
//! \param hw The hardware device to check for incompatibility.
//! \param details Pre-allocated incompatibility details object initialized by ORT.
//! \return nullptr on success (compatible or details set), OrtStatus on error.
//!
OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetHardwareDeviceIncompatibilityDetailsImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* hw,
    OrtDeviceEpIncompatibilityDetails* details) noexcept
{
    auto* factory = static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    if (hw == nullptr || details == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "[NvTensorRTRTX EP] Invalid arguments: hw or details is null");
    }

    // Check if the device is a GPU from NVIDIA vendor
    OrtHardwareDeviceType device_type = factory->ort_api.HardwareDevice_Type(hw);
    uint32_t vendor_id = factory->ort_api.HardwareDevice_VendorId(hw);

    if (device_type != OrtHardwareDeviceType::OrtHardwareDeviceType_GPU ||
        vendor_id != factory->vendor_id_)
    {
        // Not a NVIDIA GPU - device type/vendor incompatible
        uint32_t reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
        return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
            details,
            reasons,
            0,  // error_code
            "NvTensorRTRTX EP only supports NVIDIA GPU devices");
    }

    // Check compute capability and get major/minor for driver version check
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    if (!IsOrtHardwareDeviceSupported(hw, factory->ort_api, &compute_capability_major, &compute_capability_minor))
    {
        uint32_t reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;

        // CC remains 0.0 only when the device was never matched in the CUDA device list
        // (missing LUID/PCI-bus-ID metadata, CUDA enumeration failure, etc.).
        // This is a lookup failure, not a compute-capability issue, so report it separately
        // to avoid the confusing message "does not support GPU with Compute Capability 0.0".
        if (compute_capability_major == 0 && compute_capability_minor == 0)
        {
            return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
                details,
                reasons,
                0,  // error_code
                "NvTensorRTRTX EP could not resolve the GPU device properties via CUDA. "
                "Ensure the NVIDIA driver and CUDA runtime are correctly installed.");
        }

        // Device was found but its architecture is below the minimum requirement.
        std::string cc_string = std::to_string(compute_capability_major) + "." + std::to_string(compute_capability_minor);
        std::string msg = "NvTensorRTRTX EP does not support GPU with Compute Capability " + cc_string +
                          ". Minimum required: Compute Capability 8.0 (Ampere architecture or newer).";
        return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
            details,
            reasons,
            0,  // error_code
            msg.c_str());
    }

    // Determine minimum driver version based on GPU architecture
    // Note: NVML returns driver version in standard format (e.g., "581.80"), not "R570_00" format
    // So we use standard format for minimum versions
    std::string min_driver_version;
    if (compute_capability_major >= 12)
    {
        // Blackwell architecture (CC 12.x) - requires driver 570.00 or higher
        // (R570_00 in documentation corresponds to 570.00 in NVML format)
        min_driver_version = "570.00";
    }
    else if (compute_capability_major >= 8)
    {
        // Ampere and Ada architectures (CC 8.x) - requires driver 555.85 or higher
        min_driver_version = "555.85";
    }
    else
    {
        // Should not reach here (already checked compute capability above)
        min_driver_version = "555.85";
    }

    // Initialize NVML and get driver version
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS)
    {
        uint32_t reasons = OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE;
        std::string msg = "Failed to initialize NVML: " + std::string(nvmlErrorString(nvml_result)) +
                          ". NVIDIA driver may not be properly installed.";
        return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
            details,
            reasons,
            0,  // error_code
            msg.c_str());
    }

    char driver_version_str[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {0};
    nvml_result = nvmlSystemGetDriverVersion(driver_version_str, sizeof(driver_version_str));

    // Shutdown NVML before returning
    nvmlShutdown();

    if (nvml_result != NVML_SUCCESS)
    {
        uint32_t reasons = OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE;
        std::string msg = "Failed to query NVIDIA driver version: " + std::string(nvmlErrorString(nvml_result));
        return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
            details,
            reasons,
            0,  // error_code
            msg.c_str());
    }

    // Compare driver version with minimum required
    if (!CompareNVMLDriverVersion(driver_version_str, min_driver_version))
    {
        uint32_t reasons = OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE;
        std::string msg = "NVIDIA driver version " + std::string(driver_version_str) +
                          " is too old. Minimum required: " + min_driver_version + " or higher";

        return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
            details,
            reasons,
            0,  // error_code (could store parsed version if needed)
            msg.c_str());
    }

    // Device is compatible - details are already initialized with default values by ORT
    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] GetSupportedDevicesImpl: this_ptr is null");
    }

    auto* factory = static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    // Security check: validate remaining input parameters
    if (num_devices > 0 && devices == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] GetSupportedDevicesImpl: devices array is null");
    }
    if (ep_devices == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] GetSupportedDevicesImpl: ep_devices output is null");
    }
    if (p_num_ep_devices == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] GetSupportedDevicesImpl: p_num_ep_devices output is null");
    }

    size_t& num_ep_devices = *p_num_ep_devices;

    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    RETURN_IF_ERROR(factory->CreateMemoryInfoForDevices(num_cuda_devices));

    int16_t device_id = 0;

    // std::to_string may throw std::bad_alloc or std::bad_array_new_length on memory allocation failure
    try
    {
        for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i)
        {
            const OrtHardwareDevice* device = devices[i];

            if (factory->ort_api.HardwareDevice_Type(device) == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
                factory->ort_api.HardwareDevice_VendorId(device) == factory->vendor_id_ &&
                IsOrtHardwareDeviceSupported(device, factory->ort_api, nullptr, nullptr))
            {
                OrtKeyValuePairs* ep_options = nullptr;
                OrtKeyValuePairs* ep_metadata = nullptr;
                factory->ort_api.CreateKeyValuePairs(&ep_options);
                factory->ort_api.CreateKeyValuePairs(&ep_metadata);
                factory->ort_api.AddKeyValuePair(ep_options, "device_id", std::to_string(device_id).c_str());

                RETURN_IF_ERROR(factory->ort_api.GetEpApi()->CreateEpDevice(factory, device, ep_metadata, ep_options,
                                                                            &ep_devices[num_ep_devices]));
                factory->ort_api.ReleaseKeyValuePairs(ep_options);
                factory->ort_api.ReleaseKeyValuePairs(ep_metadata);

                const OrtMemoryInfo* gpu_mem_info = factory->device_memory_infos[device_id].get();
                const OrtMemoryInfo* host_accessible_mem_info = factory->pinned_memory_infos[device_id].get();

                RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_devices[num_ep_devices], gpu_mem_info));
                RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_devices[num_ep_devices], host_accessible_mem_info));
                num_ep_devices++;
                device_id++;
            }
        }
    }
    catch (const std::exception& e)
    {
        return factory->ort_api.CreateStatus(ORT_FAIL, e.what());
    }
    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
    _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata,
    _In_ size_t num_devices,
    _In_ const OrtSessionOptions* session_options,
    _In_ const OrtLogger* logger,
    _Out_ OrtEp** ep) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpImpl: this_ptr is null");
    }

    auto* factory = static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    // Security check: validate remaining input parameters
    if (session_options == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpImpl: session_options is null");
    }
    if (logger == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpImpl: logger is null");
    }
    if (ep == nullptr)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpImpl: ep output is null");
    }
    *ep = nullptr;

    if (num_devices != 1)
    {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                             "EP only supports selection for one device.");
    }

    // Log creation
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Execution Provider",
                                                       ORT_FILE, __LINE__, __FUNCTION__));

    auto execution_provider = std::make_unique<TensorrtRtxExecutionProvider>(
        *factory, factory->ep_name_, *session_options, *logger);

    *ep = execution_provider.release();
    return nullptr;
}

void ORT_API_CALL TensorrtRtxExecutionProviderFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept
{
    // Security check: validate ep is not null before deleting
    if (ep == nullptr)
    {
        return;
    }
    TensorrtRtxExecutionProvider* execution_provider = static_cast<TensorrtRtxExecutionProvider*>(ep);
    delete execution_provider;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options,
    OrtAllocator** allocator) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateAllocatorImpl: this_ptr is null");
    }

    auto& factory = *static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    // Security check: validate remaining input parameters
    if (memory_info == nullptr)
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateAllocatorImpl: memory_info is null");
    }
    if (allocator == nullptr)
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateAllocatorImpl: allocator output is null");
    }
    // Note: allocator_options can be null, so we don't check it

    const OrtMemoryDevice* mem_device = factory.ep_api.MemoryInfo_GetMemoryDevice(memory_info);
    uint32_t device_id = factory.ep_api.MemoryDevice_GetDeviceId(mem_device);

    if (factory.ep_api.MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_DEFAULT)
    {
        // Use the one that was previously created
        if (factory.device_allocators.find(device_id) != factory.device_allocators.end())
        {
            *allocator = factory.device_allocators[device_id].get();
            return nullptr;
        }

       // Create CUDA mempool allocator for shared activation memory allocation.
        {
            std::unique_ptr<CudaMempoolAllocator> mempool_allocator;
            RETURN_IF_ERROR(CudaMempoolAllocator::Create(
                memory_info,
                static_cast<DeviceId>(device_id),
                factory.ort_api,
                factory.default_logger_,
                mempool_allocator));

            factory.device_mempool_allocators[device_id] = std::move(mempool_allocator);
        }
        
        // Create BFC arena allocator for non-shared activation memory allocation.
        {
            // Fall back to BFC arena
            auto cuda_allocator_raw = new TensorrtRtxAllocator(memory_info, static_cast<DeviceId>(device_id));
            AllocatorUniquePtr<OrtAllocator> cuda_allocator(
                static_cast<OrtAllocator*>(cuda_allocator_raw),
                [](OrtAllocator* p)
                {
                    delete static_cast<TensorrtRtxAllocator*>(p);
                });

            std::unique_ptr<ArenaAllocator> arena_allocator = nullptr;
            RETURN_IF_ERROR(ArenaAllocator::CreateOrtArenaAllocator(std::move(cuda_allocator), allocator_options, factory.ort_api, factory.default_logger_, arena_allocator));

            *allocator = arena_allocator.get();
            factory.device_allocators[device_id] = std::move(arena_allocator);
        }
    }
    else if (factory.ep_api.MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE)
    {
        // Use the one that was previously created
        if (factory.pinned_allocators.find(device_id) != factory.pinned_allocators.end())
        {
            *allocator = factory.pinned_allocators[device_id].get();
            return nullptr;
        }

        // Create a CUDA pinned allocator
        auto cuda_pinned_allocator_raw = new TensorrtRtxPinnedAllocator(memory_info);
        AllocatorUniquePtr<OrtAllocator> cuda_pinned_allocator(
            static_cast<OrtAllocator*>(cuda_pinned_allocator_raw),
            [](OrtAllocator* p)
            {
                delete static_cast<TensorrtRtxPinnedAllocator*>(p);
            });

        std::unique_ptr<ArenaAllocator> arena_allocator = nullptr;
        RETURN_IF_ERROR(ArenaAllocator::CreateOrtArenaAllocator(std::move(cuda_pinned_allocator), allocator_options, factory.ort_api, factory.default_logger_, arena_allocator));

        *allocator = arena_allocator.get();
        factory.pinned_allocators[device_id] = std::move(arena_allocator);
    }
    else
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                            "Value did not come directly from an OrtEpDevice returned by this factory.");
    }

    return nullptr;
}

void ORT_API_CALL TensorrtRtxExecutionProviderFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this_ptr*/,
    OrtAllocator* allocator) noexcept
{
    // TODO: Release allocator if it's not shared.
    // If using shared allocators across sessions, this can be a no-op.
    // delete static_cast<YourAllocator*>(allocator);
    (void)allocator;  // Suppress unused parameter warning
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateDataTransferImpl: this_ptr is null");
    }

    auto& factory = *static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    // Security check: validate output parameter is not null
    if (data_transfer == nullptr)
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateDataTransferImpl: data_transfer output is null");
    }

    if (factory.data_transfer_impl == nullptr)
    {
        factory.data_transfer_impl = std::make_unique<TensorrtRtxDataTransfer>(
            static_cast<const ApiPtrs&>(factory),
            factory.device_mem_devices,
            factory.pinned_mem_devices,
            factory.vendor_id_);
    }
    *data_transfer = factory.data_transfer_impl.get();
    return nullptr;
}

bool ORT_API_CALL TensorrtRtxExecutionProviderFactory::IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return false;
    }
    // TODO: Return true if your EP supports stream-based execution.
    return true;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                                           const OrtMemoryDevice* memory_device,
                                                                                           const OrtKeyValuePairs* stream_options,
                                                                                           OrtSyncStreamImpl** ort_stream) noexcept
{
    // Security check: validate this_ptr is not null
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] Factory CreateSyncStreamForDeviceImpl: this_ptr is null");
    }

    auto& factory = *static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    // Security check: validate remaining input parameters
    if (memory_device == nullptr)
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] Factory CreateSyncStreamForDeviceImpl: memory_device is null");
    }
    if (ort_stream == nullptr)
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] Factory CreateSyncStreamForDeviceImpl: ort_stream output is null");
    }
    // Note: stream_options can be null, so we don't check it
    auto device_id = factory.ep_api.MemoryDevice_GetDeviceId(memory_device);

    std::unique_ptr<TensorrtRtxSyncStreamImpl> impl;
    RETURN_IF_ERROR(TensorrtRtxSyncStreamImpl::Create(factory, nullptr, device_id, stream_options, impl));
    *ort_stream = impl.release();
    return nullptr;
}

//
// Custom Op Domain Support 
//

//!
//! \brief Registers TensorRT-specific custom operations with ONNX Runtime.
//!
//! \details This method creates a custom op domain for TensorRT operations (currently FP4/FP8
//!          quantization operations) and registers them with ONNX Runtime's infrastructure.
//!          When ONNX Runtime encounters these operations in a model, it routes execution
//!          to this execution provider rather than attempting to execute them on the CPU.
//!
//!
//! \return nullptr on success, or an OrtStatus pointer describing the error on failure.
//!
OrtStatus* TensorrtRtxExecutionProviderFactory::InitializeCustomOpDomains()
{
    // Create a custom operation domain named "trt" to namespace TensorRT-specific operations.
    OrtCustomOpDomain* trt_domain = nullptr;
    RETURN_IF_ERROR(ort_api.CreateCustomOpDomain(kTrtCustomOpDomain, &trt_domain));
    
    // Static storage ensures custom operation objects remain valid for the application lifetime.
    // ONNX Runtime holds pointers to these objects, so they must not be destroyed prematurely.
    static std::vector<std::unique_ptr<TensorRTRtxCustomOp>> custom_ops;
    
    // Register each TensorRT custom operation (e.g., TRT_FP4DynamicQuantize, TRT_FP8QuantizeLinear).
    for (const char* name : kTrtCustomOpNames) {
        auto op = std::make_unique<TensorRTRtxCustomOp>(ep_name_.c_str(), /* compute_stream = */ nullptr);
        op->SetName(name);
        
        // Add the operation to the domain so ONNX Runtime can route it to this execution provider.
        RETURN_IF_ERROR(ort_api.CustomOpDomain_Add(trt_domain, op.get()));
        custom_ops.push_back(std::move(op));
    }
    
    custom_op_domains_.push_back(trt_domain);
    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetNumCustomOpDomainsImpl(
    OrtEpFactory* this_ptr,
    size_t* num_domains) noexcept
{
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, 
                                         "[NvTensorRTRTX EP] GetNumCustomOpDomainsImpl: this_ptr is null");
    }
    
    if (num_domains == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, 
                                         "[NvTensorRTRTX EP] GetNumCustomOpDomainsImpl: num_domains is null");
    }
    
    auto* factory = static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);
    *num_domains = factory->custom_op_domains_.size();
    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::GetCustomOpDomainsImpl(
    OrtEpFactory* this_ptr,
    OrtCustomOpDomain** domains,
    size_t num_domains) noexcept
{
    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, 
                                         "[NvTensorRTRTX EP] GetCustomOpDomainsImpl: this_ptr is null");
    }
    
    if (domains == nullptr && num_domains > 0)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, 
                                         "[NvTensorRTRTX EP] GetCustomOpDomainsImpl: domains is null");
    }
    
    auto* factory = static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);
    
    for (size_t i = 0; i < num_domains && i < factory->custom_op_domains_.size(); i++)
    {
        domains[i] = factory->custom_op_domains_[i];
    }
    
    return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtRtxExecutionProviderFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    const char* compatibility_info,
    OrtCompiledModelCompatibility* model_compatibility) noexcept
{
    auto log_message = [](const OrtApi& ort_api, const OrtLogger& logger,
                          OrtLoggingLevel level, const std::string& message) noexcept
    {
        OrtStatus* status = ort_api.Logger_LogMessage(&logger, level, message.c_str(),
                                                      ORT_FILE, __LINE__, __FUNCTION__);
        if (status != nullptr)
        {
            ort_api.ReleaseStatus(status);
        }
    };

    if (this_ptr == nullptr)
    {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                          "[NvTensorRTRTX EP] ValidateCompiledModelCompatibilityInfoImpl: null OrtEpFactory");
    }
    auto& factory = *static_cast<TensorrtRtxExecutionProviderFactory*>(this_ptr);

    if (compatibility_info == nullptr || model_compatibility == nullptr)
    {
        return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "[NvTensorRTRTX EP] Invalid arguments: compatibility_info or model_compatibility is null");
    }

    (void)devices;
    (void)num_devices;

    try
    {
        if (compatibility_info[0] == '\0')
        {
            *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
            return nullptr;
        }

        std::vector<uint8_t> engine_header;
        std::string decode_error;
        if (!TryHexStringToBinary(compatibility_info, engine_header, decode_error))
        {
            std::string message = "[NvTensorRTRTX EP] Failed to decode engine header: " + decode_error;
            log_message(factory.ort_api, factory.default_logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, message);
            *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
            return nullptr;
        }

        if (engine_header.size() != kTensorRTEngineHeaderSize)
        {
            std::string message = "[NvTensorRTRTX EP] Invalid header size: " +
                                  std::to_string(engine_header.size()) + " bytes (expected 64)";
            log_message(factory.ort_api, factory.default_logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, message);
            *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
            return nullptr;
        }

        static std::mutex runtime_creation_mutex;
        std::unique_ptr<nvinfer1::IRuntime> runtime;
        {
            std::lock_guard<std::mutex> lock(runtime_creation_mutex);
            TensorrtRtxLogger& trt_logger = GetTensorrtRtxLogger(false);
            trt_logger.set_ort_logger(&factory.default_logger_, &factory.ort_api);
            runtime.reset(nvinfer1::createInferRuntime(trt_logger));
        }

        if (!runtime)
        {
            std::string message = "[NvTensorRTRTX EP] Failed to create TensorRT runtime";
            log_message(factory.ort_api, factory.default_logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, message);
            return factory.ort_api.CreateStatus(ORT_FAIL, message.c_str());
        }

        uint64_t diagnostics = 0;
        nvinfer1::EngineValidity validity = runtime->getEngineValidity(engine_header.data(), engine_header.size(),
                                                                       &diagnostics);

        switch (validity)
        {
        case nvinfer1::EngineValidity::kVALID:
            *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
            break;
        case nvinfer1::EngineValidity::kSUBOPTIMAL:
        {
            std::ostringstream message;
            message << "[NvTensorRTRTX EP] Engine compatible but recompilation recommended (diagnostics: 0x" << std::hex
                    << diagnostics << std::dec << ")";
            log_message(factory.ort_api, factory.default_logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                        message.str());
            *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
            break;
        }
        case nvinfer1::EngineValidity::kINVALID:
        {
            std::ostringstream message;
            message << "[NvTensorRTRTX EP] Engine incompatible with this system (diagnostics: 0x" << std::hex
                    << diagnostics << std::dec << ")";
            log_message(factory.ort_api, factory.default_logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                        message.str());
            *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
            break;
        }
        default:
        {
            std::ostringstream message;
            message << "[NvTensorRTRTX EP] Unknown validity status: " << static_cast<int>(validity);
            log_message(factory.ort_api, factory.default_logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                        message.str());
            *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
            break;
        }
        }

        return nullptr;
    }
    catch (const std::exception& ex)
    {
        std::string error_msg = std::string("[NvTensorRTRTX EP] Exception during validation: ") + ex.what();
        (void)factory.ort_api.Logger_LogMessage(&factory.default_logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                                error_msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
        return factory.ort_api.CreateStatus(ORT_FAIL, error_msg.c_str());
    }
    catch (...)
    {
        std::string error_msg = "[NvTensorRTRTX EP] Unknown exception during validation";
        (void)factory.ort_api.Logger_LogMessage(&factory.default_logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                                error_msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
        return factory.ort_api.CreateStatus(ORT_FAIL, error_msg.c_str());
    }
}
}  // namespace trt_rtx_ep

// To make symbols visible on macOS/iOS
#if defined(__APPLE__)
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C"
{
    //
    // Public C API - Entry point for ORT to create the factory
    //
    EXPORT_SYMBOL OrtStatus* CreateEpFactories(
        const char* registration_name,
        const OrtApiBase* ort_api_base,
        const OrtLogger* default_logger,
        OrtEpFactory** factories,
        size_t max_factories,
        size_t* num_factories)
    {
        // Security check: validate critical input parameters are not null
        // Note: We must check ort_api_base first before we can use OrtApi to create error statuses
        if (ort_api_base == nullptr)
        {
            // Cannot create proper OrtStatus without OrtApi, return nullptr to indicate failure
            // ORT will handle this as an error condition
            return nullptr;
        }

        const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
        if (ort_api == nullptr)
        {
            return nullptr;  // Cannot create status without OrtApi
        }

        // Now we can create proper error statuses
        if (registration_name == nullptr)
        {
            return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpFactories: registration_name is null");
        }
        if (default_logger == nullptr)
        {
            return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpFactories: default_logger is null");
        }
        if (factories == nullptr)
        {
            return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpFactories: factories output array is null");
        }
        if (num_factories == nullptr)
        {
            return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "[NvTensorRTRTX EP] CreateEpFactories: num_factories output is null");
        }

        const OrtEpApi* ort_ep_api = ort_api->GetEpApi();
        if (ort_ep_api == nullptr)
        {
            return ort_api->CreateStatus(ORT_FAIL, "[NvTensorRTRTX EP] CreateEpFactories: Failed to get OrtEpApi");
        }

        const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
        if (model_editor_api == nullptr)
        {
            return ort_api->CreateStatus(ORT_FAIL, "[NvTensorRTRTX EP] CreateEpFactories: Failed to get OrtModelEditorApi");
        }

        // Load external tensorrt_plugins library from EP directory
        // This library contains GroupQueryAttention and RotaryEmbedding plugins for transformer models
        static std::once_flag plugins_load_flag;
        std::call_once(plugins_load_flag, [&]()
                       {
            try
            {
#if defined(_WIN32)
                // Get EP DLL path using GetModuleHandleExW (wide string version)
                HMODULE hModule = NULL;
                if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | 
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                       (LPCWSTR)&CreateEpFactories, &hModule))
                {
                    wchar_t path[MAX_PATH];
                    GetModuleFileNameW(hModule, path, MAX_PATH);
                    std::filesystem::path ep_dir = std::filesystem::path(path).parent_path();
                    auto plugin_path = ep_dir / L"tensorrt_plugins.dll";
                    
                    //Use LoadLibraryExW to use search path control flags
                    HMODULE plugin_dll = LoadLibraryExW(plugin_path.wstring().c_str(), nullptr,
                                                       LOAD_WITH_ALTERED_SEARCH_PATH);
                    if (plugin_dll)
                    {
                        //Log success
                        std::string msg = "[NvTensorRTRTX EP] External plugins loaded: tensorrt_plugins";
                        ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_INFO,
                                                   msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                    }
                    else
                    {
                        // Log failure
                        DWORD error_code = GetLastError();
                        LPWSTR error_msg = nullptr;
                        FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                       nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                       (LPWSTR)&error_msg, 0, nullptr);
                        
                        std::wstring wide_path = plugin_path.wstring();
                        std::string path_str(wide_path.begin(), wide_path.end());
                        std::string msg = "[NvTensorRTRTX EP] Failed to load tensorrt_plugins " + path_str +
                                          " (Error " + std::to_string(error_code) + ")";
                        if (error_msg)
                        {
                            std::wstring wide_err(error_msg);
                            std::string err_str(wide_err.begin(), wide_err.end());
                            msg += ": " + err_str;
                            LocalFree(error_msg);
                        }
                        ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_WARNING,
                                                   msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                    }
                }
#else
                // Linux: Use dladdr to get .so path
                Dl_info dl_info;
                if (dladdr((void*)CreateEpFactories, &dl_info))
                {
                    std::filesystem::path ep_dir = std::filesystem::path(dl_info.dli_fname).parent_path();
                    auto plugin_path = ep_dir / "libtensorrt_plugins.so";
                    
                    void* plugin_handle = dlopen(plugin_path.string().c_str(), RTLD_LAZY);
                    if (plugin_handle)
                    {
                        // Log success
                        std::string msg = "[NvTensorRTRTX EP] External plugins loaded: tensorrt_plugins";
                        ort_api->Logger_LogMessage(default_logger, ORT_LOGGING_LEVEL_INFO,
                                                   msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__);
                    }
                }
#endif
            }
            catch (...)
            {
                // Silently ignore - plugin loading is optional
            } });

        // Create factory instance
        std::unique_ptr<OrtEpFactory> factory = std::make_unique<trt_rtx_ep::TensorrtRtxExecutionProviderFactory>(
            registration_name, *default_logger, ApiPtrs{*ort_api, *ort_ep_api, *model_editor_api});

        if (max_factories < 1)
        {
            return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                         "Not enough space to return EP factory. Need at least one.");
        }

        factories[0] = factory.release();
        *num_factories = 1;

        return nullptr;
    }

    EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory)
    {
        // Security check: validate factory is not null before deleting
        if (factory == nullptr)
        {
            return nullptr;  // Nothing to release, not an error
        }
        delete static_cast<trt_rtx_ep::TensorrtRtxExecutionProviderFactory*>(factory);
        return nullptr;
    }

}  // extern "C"
