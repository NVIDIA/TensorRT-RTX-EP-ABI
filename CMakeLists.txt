# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.28)

# Enable MSBuild multi-tool task scheduling: each source file becomes an
# independent MSBuild task, allowing the scheduler to interleave compilations
# across project boundaries rather than waiting for a whole .vcxproj to finish.
# EnforceProcessCountAcrossBuilds prevents oversubscription when --parallel N
# is also passed to cmake --build.
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    list(APPEND CMAKE_VS_GLOBALS
        "UseMultiToolTask=true"
        "EnforceProcessCountAcrossBuilds=true"
    )
endif()

project(trt-rtx-ep LANGUAGES CXX)

# =============================================================================
# Target Architecture
# =============================================================================
# Visual Studio generators expose the selected target through
# CMAKE_GENERATOR_PLATFORM. Ninja does not, so it uses the MSVC developer
# environment or an explicit TRT_RTX_EP_TARGET_ARCH. ARM64EC must not be
# inferred from C++ compiler macros because it has x64-compatible ABI traits.
set(TRT_RTX_EP_TARGET_ARCH "" CACHE STRING
    "Target architecture (x64, arm64ec, or arm64). Inferred from the generator or MSVC environment.")
set_property(CACHE TRT_RTX_EP_TARGET_ARCH PROPERTY STRINGS x64 arm64ec arm64)
set(TRT_RTX_EP_OUTPUT_NAME "onnxruntime_providers_nv_tensorrt_rtx")
set(TRT_RTX_EP_DEPENDENCY_SUBDIR "")

if(WIN32)
    string(TOLOWER "${TRT_RTX_EP_TARGET_ARCH}" TRT_RTX_EP_TARGET_ARCH)
    set(_trt_rtx_detected_arch "")

    # Visual Studio provides -A through CMAKE_GENERATOR_PLATFORM. Ninja's
    # selected MSVC target comes from the developer environment instead.
    if(CMAKE_GENERATOR_PLATFORM)
        string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _trt_rtx_detected_arch)
    elseif(NOT TRT_RTX_EP_TARGET_ARCH AND
           DEFINED ENV{VSCMD_ARG_TGT_ARCH} AND
           NOT "$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "")
        string(TOLOWER "$ENV{VSCMD_ARG_TGT_ARCH}" _trt_rtx_detected_arch)
    elseif(NOT TRT_RTX_EP_TARGET_ARCH)
        # CLion can configure Ninja with an absolute cl.exe path without the
        # developer-prompt environment.
        get_filename_component(_trt_rtx_compiler_dir
            "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_trt_rtx_detected_arch
            "${_trt_rtx_compiler_dir}" NAME)
        string(TOLOWER "${_trt_rtx_detected_arch}" _trt_rtx_detected_arch)
    endif()

    if(_trt_rtx_detected_arch STREQUAL "amd64")
        set(_trt_rtx_detected_arch "x64")
    endif()
    if(_trt_rtx_detected_arch AND
       NOT _trt_rtx_detected_arch MATCHES "^(x64|arm64ec|arm64)$")
        message(FATAL_ERROR
            "Unable to infer a supported Windows target architecture from "
            "'${_trt_rtx_detected_arch}'. Set TRT_RTX_EP_TARGET_ARCH to x64, arm64ec, or arm64.")
    endif()

    if(NOT TRT_RTX_EP_TARGET_ARCH)
        set(TRT_RTX_EP_TARGET_ARCH "${_trt_rtx_detected_arch}")
    elseif(CMAKE_GENERATOR_PLATFORM AND
           NOT TRT_RTX_EP_TARGET_ARCH STREQUAL _trt_rtx_detected_arch)
        message(FATAL_ERROR
            "TRT_RTX_EP_TARGET_ARCH='${TRT_RTX_EP_TARGET_ARCH}' conflicts with "
            "CMAKE_GENERATOR_PLATFORM='${CMAKE_GENERATOR_PLATFORM}'.")
    endif()

    if(NOT TRT_RTX_EP_TARGET_ARCH MATCHES "^(x64|arm64ec|arm64)$")
        message(FATAL_ERROR
            "Unsupported TRT_RTX_EP_TARGET_ARCH='${TRT_RTX_EP_TARGET_ARCH}'. "
            "Expected x64, arm64ec, or arm64.")
    endif()

    # Standalone x64 and ARM64 builds both retain the public unsuffixed DLL
    # name and flat dependency layout.

    message(STATUS "TensorRT RTX EP target architecture: ${TRT_RTX_EP_TARGET_ARCH}")
    message(STATUS "TensorRT RTX EP output name: ${TRT_RTX_EP_OUTPUT_NAME}")
endif()

# =============================================================================
# Project Version Configuration
# =============================================================================
# Default 0.0.0 for local/dev builds. CI overrides via:
#   cmake -DTRT_RTX_EP_VERSION=1.2.3 ..
set(TRT_RTX_EP_VERSION "0.0.0" CACHE STRING "EP version (set by CI)")
if(TRT_RTX_EP_VERSION STREQUAL "")
    set(TRT_RTX_EP_VERSION "0.0.0")
endif()
if(NOT TRT_RTX_EP_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
        "TRT_RTX_EP_VERSION must use M.m.p format, got '${TRT_RTX_EP_VERSION}'.")
endif()

message(STATUS "TensorRT RTX EP version: ${TRT_RTX_EP_VERSION}")

# Split version string into components for Windows VERSIONINFO resource
string(REPLACE "." ";" TRT_RTX_EP_VERSION_LIST "${TRT_RTX_EP_VERSION}")
list(GET TRT_RTX_EP_VERSION_LIST 0 TRT_RTX_EP_VERSION_MAJOR)
list(GET TRT_RTX_EP_VERSION_LIST 1 TRT_RTX_EP_VERSION_MINOR)
list(GET TRT_RTX_EP_VERSION_LIST 2 TRT_RTX_EP_VERSION_PATCH)

# Windows version resource — embeds FILEVERSION / PRODUCTVERSION into the DLL
if(WIN32)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/version.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/version.rc"
        @ONLY
    )
    set(TRT_RTX_EP_VERSION_RC "${CMAKE_CURRENT_BINARY_DIR}/generated/version.rc")
endif()

# =============================================================================
# C++ Standard Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Disable compiler-specific extensions for strict C++20 compliance

# Generate compile_commands.json for IDE integration and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position-independent code (required for shared libraries on some platforms)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRT_RTX_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
# Keep build-tree executables, the EP, and all runtime DLLs together. Install
# rules below create the separately packaged executable/test/EP layouts.
set(TRT_RTX_EP_BIN_DIR "${TRT_RTX_BIN_DIR}")
set(TRT_RTX_ARCHIVE_DIR "${CMAKE_BINARY_DIR}/lib")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${TRT_RTX_ARCHIVE_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${TRT_RTX_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${TRT_RTX_BIN_DIR}")

function(trt_rtx_set_target_output_directory target directory)
    set_target_properties(${target} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${TRT_RTX_ARCHIVE_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${directory}"
        RUNTIME_OUTPUT_DIRECTORY "${directory}"
    )
endfunction()

# =============================================================================
# Compiler Warning Flags
# =============================================================================
if(MSVC)
    # MSVC warning flags
    add_compile_options(/W4 /WX-)  # Warning level 4, don't treat warnings as errors
    add_compile_options(/wd4100)   # Unreferenced formal parameter
    add_compile_options(/wd4996)   # Deprecated functions
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    # GCC/Clang warning flags
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-missing-field-initializers)
endif()

# =============================================================================
# Find Required Packages
# =============================================================================

# Security-related compiler flags for MSVC (Windows)
if(MSVC)
    if(WIN32 AND TRT_RTX_EP_TARGET_ARCH STREQUAL "x64")
        set(_trt_rtx_spectre_default ON)
    else()
        # ARM64/ARM64EC Spectre-mitigated CRT libraries are an optional Visual
        # Studio component and are not present in the standard cross toolchain.
        set(_trt_rtx_spectre_default OFF)
    endif()
    option(TRT_RTX_EP_ENABLE_SPECTRE_MITIGATIONS
        "Compile with /Qspectre (requires architecture-specific Spectre libraries)"
        ${_trt_rtx_spectre_default})

    # /GS - Buffer security check (enabled by default, but explicit)
    # /guard:cf - Control Flow Guard
    # /sdl - Security Development Lifecycle checks
    add_compile_options(/GS /guard:cf /sdl)
    if(TRT_RTX_EP_ENABLE_SPECTRE_MITIGATIONS)
        # /Qspectre - Spectre mitigation
        add_compile_options(/Qspectre)
    elseif(WIN32 AND NOT TRT_RTX_EP_TARGET_ARCH STREQUAL "x64")
        message(STATUS
            "ARM Spectre mitigation disabled. In Visual Studio Installer, select "
            "Individual components > MSVC v143 - VS 2022 C++ ARM64/ARM64EC "
            "Spectre-mitigated libs (Latest), then set "
            "TRT_RTX_EP_ENABLE_SPECTRE_MITIGATIONS=ON for production builds.")
    endif()
    # /sdl can promote deprecation diagnostics after the earlier /wd4996.
    # Repeat it here so MSVC keeps third-party deprecated API uses as warnings. Added due to protobuf
    add_compile_options(/wd4996)
    
    # Enable Control Flow Guard for linker as well
    add_link_options(/guard:cf)

    # RelWithDebInfo: disable incremental linking and enable dead-code stripping
    # to keep DLL size close to Release while retaining debug symbols in the PDB.
    string(REPLACE "/INCREMENTAL" "/INCREMENTAL:NO"
        CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO
        "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF /OPT:ICF")
endif()

# Find CUDA Toolkit (required)
#
# NVIDIA's combined Windows ARM64 toolkit layout contains both lib/x64 and
# lib/arm64. FindCUDAToolkit currently searches the host ABI first, so an
# ARM64 cross-build can otherwise be configured with x64 import libraries and
# fail only at the final link step. Seed the architecture-specific libraries
# before the imported CUDA targets are created.
set(TRT_RTX_EP_CUDA_LIBRARY_DIR "" CACHE PATH
    "CUDA import-library directory. Inferred from CUDAToolkit_ROOT on Windows.")
if(WIN32 AND NOT TRT_RTX_EP_CUDA_LIBRARY_DIR AND CUDAToolkit_ROOT)
    if(TRT_RTX_EP_TARGET_ARCH STREQUAL "arm64")
        set(_trt_rtx_cuda_library_arch "arm64")
    else()
        # ARM64EC applications use the x64 ABI surface of CUDA.
        set(_trt_rtx_cuda_library_arch "x64")
    endif()

    set(_trt_rtx_cuda_library_candidate
        "${CUDAToolkit_ROOT}/lib/${_trt_rtx_cuda_library_arch}")
    if(EXISTS "${_trt_rtx_cuda_library_candidate}/cudart.lib" AND
       EXISTS "${_trt_rtx_cuda_library_candidate}/cuda.lib" AND
       EXISTS "${_trt_rtx_cuda_library_candidate}/nvml.lib")
        set(TRT_RTX_EP_CUDA_LIBRARY_DIR
            "${_trt_rtx_cuda_library_candidate}" CACHE PATH
            "CUDA import-library directory. Inferred from CUDAToolkit_ROOT on Windows."
            FORCE)
    endif()
endif()

if(WIN32 AND TRT_RTX_EP_CUDA_LIBRARY_DIR)
    foreach(_cuda_library IN ITEMS cudart cuda nvml)
        if(NOT EXISTS "${TRT_RTX_EP_CUDA_LIBRARY_DIR}/${_cuda_library}.lib")
            message(FATAL_ERROR
                "TRT_RTX_EP_CUDA_LIBRARY_DIR is missing ${_cuda_library}.lib: "
                "${TRT_RTX_EP_CUDA_LIBRARY_DIR}")
        endif()
    endforeach()

    # These are the cache variables consumed by FindCUDAToolkit when it
    # constructs CUDA::cudart, CUDA::cuda_driver, and CUDA::nvml.
    set(CUDA_CUDART "${TRT_RTX_EP_CUDA_LIBRARY_DIR}/cudart.lib"
        CACHE FILEPATH "CUDA runtime import library" FORCE)
    set(CUDA_cudart_LIBRARY "${TRT_RTX_EP_CUDA_LIBRARY_DIR}/cudart.lib"
        CACHE FILEPATH "CUDA runtime import library" FORCE)
    set(CUDA_cuda_driver_LIBRARY "${TRT_RTX_EP_CUDA_LIBRARY_DIR}/cuda.lib"
        CACHE FILEPATH "CUDA driver import library" FORCE)
    set(CUDA_nvml_LIBRARY "${TRT_RTX_EP_CUDA_LIBRARY_DIR}/nvml.lib"
        CACHE FILEPATH "NVML import library" FORCE)
    message(STATUS
        "CUDA import libraries for ${TRT_RTX_EP_TARGET_ARCH}: "
        "${TRT_RTX_EP_CUDA_LIBRARY_DIR}")
endif()

find_package(CUDAToolkit 12.9 REQUIRED CUDA::cudart CUDA::cuda_driver CUDA::nvml)

# Some redistributable CUDA SDK layouts include nvcc and version.json but do
# not expose a version through FindCUDAToolkit. Recover it from NVIDIA's
# version manifest so delay-load names and staging remain deterministic.
if(NOT CUDAToolkit_VERSION_MAJOR AND
   EXISTS "${CUDAToolkit_ROOT}/version.json")
    file(READ "${CUDAToolkit_ROOT}/version.json" _cuda_version_manifest)
    string(JSON CUDAToolkit_VERSION ERROR_VARIABLE _cuda_version_error
        GET "${_cuda_version_manifest}" cuda version)
    if(NOT _cuda_version_error)
        string(REPLACE "." ";" _cuda_version_parts "${CUDAToolkit_VERSION}")
        list(GET _cuda_version_parts 0 CUDAToolkit_VERSION_MAJOR)
        list(GET _cuda_version_parts 1 CUDAToolkit_VERSION_MINOR)
        list(GET _cuda_version_parts 2 CUDAToolkit_VERSION_PATCH)
        message(STATUS
            "Recovered CUDA Toolkit version from version.json: ${CUDAToolkit_VERSION}")
    endif()
endif()
if(NOT CUDAToolkit_VERSION_MAJOR)
    message(FATAL_ERROR
        "Could not determine the CUDA Toolkit major version from FindCUDAToolkit "
        "or ${CUDAToolkit_ROOT}/version.json.")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorrt_rtx.cmake")

set(_onnxruntime_runtime_files "")
if(WIN32)
    if(ONNXRUNTIME_DLL)
        list(APPEND _onnxruntime_runtime_files "${ONNXRUNTIME_DLL}")
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_DLL)
        list(APPEND _onnxruntime_runtime_files "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}")
    endif()
else()
    if(ONNXRUNTIME_LIB)
        list(APPEND _onnxruntime_runtime_files "${ONNXRUNTIME_LIB}")
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
        list(APPEND _onnxruntime_runtime_files "${ONNXRUNTIME_PROVIDERS_SHARED_LIB}")
    endif()
endif()

if(_onnxruntime_runtime_files)
    add_custom_target(onnxruntime_runtime_dependencies
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TRT_RTX_BIN_DIR}/$<CONFIG>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_onnxruntime_runtime_files}
            "${TRT_RTX_BIN_DIR}/$<CONFIG>"
        COMMENT "Copying ONNX Runtime runtime dependencies"
        VERBATIM
    )
    add_dependencies(onnxruntime::onnxruntime onnxruntime_runtime_dependencies)
    add_dependencies(onnxruntime_interface onnxruntime_runtime_dependencies)
endif()
set(TRT_RTX_ORT_RUNTIME_FILES ${_onnxruntime_runtime_files})
unset(_onnxruntime_runtime_files)

# =============================================================================
# Source Files
# =============================================================================
# Note: CMake documentation recommends explicitly listing source files,
# but GLOB is acceptable for smaller projects with careful regeneration.
file(GLOB TRT_RTX_EP_SOURCES
    "src/*.cc"
    "src/utils/*.cc"
    "src/kernels/*.cc"
)

file(GLOB TRT_RTX_EP_HEADERS
    "src/*.h"
    "src/utils/*.h"
    "src/kernels/*.h"
)

# secure_load.cc is Windows-only (delay load hooks)
# dll_main.cc is Windows-only (DllMain module pin)
if(NOT WIN32)
    list(FILTER TRT_RTX_EP_SOURCES EXCLUDE REGEX ".*secure_load\\.cc$")
    list(FILTER TRT_RTX_EP_SOURCES EXCLUDE REGEX ".*dll_main\\.cc$")
endif()

# =============================================================================
# External Dependencies (FetchContent)
# =============================================================================
option(USE_VCPKG "use vcpkg package manager" OFF)
option(USE_PRECOMPILED_HOST_PROTOC "Use a precompiled win-x64 host protoc when building protobuf/ONNX" OFF)
option(TRT_RTX_EP_ENABLE_NVTX3 "Enable NVTX3 instrumentation in the TensorRT RTX EP" OFF)

if(USE_VCPKG MATCHES "OFF")
    include(FetchContent)

    set(_ORTTRTRTX_HOST_PROTOC "")
    if(USE_PRECOMPILED_HOST_PROTOC)
        if(NOT CMAKE_HOST_WIN32)
            message(FATAL_ERROR
                "USE_PRECOMPILED_HOST_PROTOC currently downloads a Windows x64 protoc "
                "binary and is only supported when CMake is running on a Windows host."
            )
        endif()

        set(_PROTOC_VERSION "31.1")
        set(_PROTOC_HOST_DIR "${CMAKE_BINARY_DIR}/_protoc_host")
        set(_HOST_PROTOC_EXE "${_PROTOC_HOST_DIR}/bin/protoc.exe")
        if(NOT EXISTS "${_HOST_PROTOC_EXE}")
            message(STATUS "Downloading host protoc v${_PROTOC_VERSION}...")
            file(MAKE_DIRECTORY "${_PROTOC_HOST_DIR}")
            file(DOWNLOAD
                "https://github.com/protocolbuffers/protobuf/releases/download/v${_PROTOC_VERSION}/protoc-${_PROTOC_VERSION}-win64.zip"
                "${_PROTOC_HOST_DIR}/protoc-${_PROTOC_VERSION}-win64.zip"
                SHOW_PROGRESS STATUS _PROTOC_DL_STATUS TLS_VERIFY ON
            )
            list(GET _PROTOC_DL_STATUS 0 _PROTOC_DL_ERR)
            if(NOT _PROTOC_DL_ERR EQUAL 0)
                list(GET _PROTOC_DL_STATUS 1 _PROTOC_DL_MSG)
                message(FATAL_ERROR
                    "Failed to download protoc: ${_PROTOC_DL_MSG}\n"
                    "For offline builds, place protoc.exe at: ${_HOST_PROTOC_EXE}"
                )
            endif()
            file(ARCHIVE_EXTRACT
                INPUT "${_PROTOC_HOST_DIR}/protoc-${_PROTOC_VERSION}-win64.zip"
                DESTINATION "${_PROTOC_HOST_DIR}"
            )
        endif()
        if(NOT EXISTS "${_HOST_PROTOC_EXE}")
            message(FATAL_ERROR "protoc not found after extraction: ${_HOST_PROTOC_EXE}")
        endif()
        set(_ORTTRTRTX_HOST_PROTOC "${_HOST_PROTOC_EXE}")
        message(STATUS "Using host protoc: ${_ORTTRTRTX_HOST_PROTOC}")
    endif()
    if(WIN32)
        set(BUILD_SHARED_LIBS OFF)
    endif()

    if(_ORTTRTRTX_HOST_PROTOC)
        # protobuf
        FetchContent_Declare(
            protobuf
            GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
            GIT_TAG        v31.1
            EXCLUDE_FROM_ALL
        )
        set(protobuf_BUILD_TESTS         OFF CACHE INTERNAL "")
        set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE INTERNAL "")
        set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(protobuf)
    endif()

    # Add ONNX
    FetchContent_Declare(
        onnx
        GIT_REPOSITORY https://github.com/onnx/onnx.git
        GIT_TAG        v1.21.0
        EXCLUDE_FROM_ALL
    )

    if(_ORTTRTRTX_HOST_PROTOC)
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE "${_ORTTRTRTX_HOST_PROTOC}" CACHE INTERNAL "")
        set(Protobuf_PROTOC_EXECUTABLE    "${_ORTTRTRTX_HOST_PROTOC}" CACHE FILEPATH "" FORCE)
    endif()

    FetchContent_MakeAvailable(onnx)

    if(MSVC)
        # Protobuf builds as third-party code under our MSVC flags. Disable /sdl
        # and warnings target-locally so promoted protobuf diagnostics do not
        # fail our build.
        foreach(_protobuf_target
            libprotobuf
            libprotobuf-lite
            libprotoc
            protoc
            protoc-gen-upb
            protoc-gen-upbdefs)
            if(TARGET ${_protobuf_target})
                target_compile_options(${_protobuf_target} PRIVATE /sdl- /w /wd4146 /wd4996)
            endif()
        endforeach()
    endif()

    set(DEPS_PATH "${CMAKE_BINARY_DIR}/_deps")
else()
    find_package(protobuf CONFIG REQUIRED)
    find_package(ONNX CONFIG REQUIRED)
endif()

# =============================================================================
# Build Configuration Options
# =============================================================================
option(TRT_RTX_EP_PRODUCTION_BUILD "Enable production build with signature verification" OFF)
set(TRT_RTX_EP_ADDITIONAL_DELAYLOAD_DLLS "" CACHE STRING
    "Additional architecture-specific NVIDIA DLL names to delay-load from the EP dependency directory")
set(_TRT_RTX_EP_EFFECTIVE_DELAYLOAD_DLLS ${TRT_RTX_EP_ADDITIONAL_DELAYLOAD_DLLS})
if(WIN32 AND NOT TRT_RTX_EP_TARGET_ARCH STREQUAL "x64")
    # CUDA 12+ uses a major-versioned runtime DLL. If the selected CUDA import
    # library instead targets NVIDIA's ARM64X hybrid runtime, LINK emits LNK4199
    # and ignores this unused /DELAYLOAD entry; the system hybrid stays a normal
    # import. If it targets cudart64_N.dll, this keeps the two architectures
    # isolated in their dependency directories.
    list(APPEND _TRT_RTX_EP_EFFECTIVE_DELAYLOAD_DLLS
        "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
endif()
list(REMOVE_DUPLICATES _TRT_RTX_EP_EFFECTIVE_DELAYLOAD_DLLS)

# =============================================================================
# Platform-Specific Configuration
# =============================================================================
if(WIN32)
    if(USE_VCPKG MATCHES "OFF")
        # # Use generator expressions for multi-config generators (e.g., Visual Studio)
        # # where CMAKE_BUILD_TYPE is empty at configure time
        set(DEPS_LIBS
            "${DEPS_PATH}/onnx-build/$<CONFIG>/onnx.lib"
            "${DEPS_PATH}/onnx-build/$<CONFIG>/onnx_proto.lib"
        )
    endif()
    set(TRT_EP_LIB_LINK_FLAG "-DEF:${CMAKE_SOURCE_DIR}/src/tensorrt_rtx_execution_provider.def")

    message(STATUS "Delay load DLL: ${TRT_RTX_DLL_NAME}")
    message(STATUS "Delay load DLL: ${TRT_ONNX_PARSER_DLL_NAME}")

    # Add delay load linker flags
    set(TRT_EP_LIB_LINK_FLAG "${TRT_EP_LIB_LINK_FLAG} /DELAYLOAD:${TRT_RTX_DLL_NAME} /DELAYLOAD:${TRT_ONNX_PARSER_DLL_NAME} /DELAYLOAD:d3d12.dll /DELAYLOAD:dxgi.dll /DELAYLOAD:nvcuda.dll")
    foreach(_delay_dll IN LISTS _TRT_RTX_EP_EFFECTIVE_DELAYLOAD_DLLS)
        if(_delay_dll)
            message(STATUS "Additional delay load DLL: ${_delay_dll}")
            set(TRT_EP_LIB_LINK_FLAG "${TRT_EP_LIB_LINK_FLAG} /DELAYLOAD:${_delay_dll}")
        endif()
    endforeach()
else()
    if(USE_VCPKG MATCHES "OFF")
        # # Linux
        set(DEPS_LIBS
            "${DEPS_PATH}/onnx-build/libonnx.a"
            "${DEPS_PATH}/onnx-build/libonnx_proto.a"
        )
    endif()
endif()

# =============================================================================
# Create Shared Library
# =============================================================================
if(WIN32)
    add_library(TensorRTRtxEp SHARED ${TRT_RTX_EP_SOURCES} ${TRT_RTX_EP_HEADERS} ${TRT_RTX_EP_VERSION_RC})
else()
    add_library(TensorRTRtxEp SHARED ${TRT_RTX_EP_SOURCES} ${TRT_RTX_EP_HEADERS})
endif()

# LTCG (Link-Time Code Generation) for whole-program optimization, scoped to
# TensorRTRtxEp only so FetchContent dependencies (protobuf, onnx, ...) are not
# affected. /GL at compile time must be paired with /LTCG at link time.
if(MSVC)
    target_compile_options(TensorRTRtxEp PRIVATE
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:RelWithDebInfo>:/GL>
    )
    target_link_options(TensorRTRtxEp PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:RelWithDebInfo>:/LTCG>
    )
endif()

# Target compile definitions (modern CMake - replaces add_definitions)
target_compile_definitions(TensorRTRtxEp PRIVATE
    ONNX_NAMESPACE=onnx
    ONNX_ML
    TRT_RTX_EP_VERSION="${TRT_RTX_EP_VERSION}"
    # Force the Ort:: C++ wrapper into manual-init mode across all TUs in this
    # target. We negotiate the ORT API version with the host at load time (see
    # src/utils/ort_api_init.h) and route the wrapper through it via
    # Ort::InitApi(). Mixing manual and default init across TUs is rejected by
    # ORT's #pragma detect_mismatch, so the definition must be consistent.
    ORT_API_MANUAL_INIT
        NOMINMAX=1
)

if(NV_TRT_MAJOR_RTX)
    target_compile_definitions(TensorRTRtxEp PRIVATE NV_TRT_MAJOR_RTX=${NV_TRT_MAJOR_RTX})
endif()

if(NV_TRT_MINOR_RTX)
    target_compile_definitions(TensorRTRtxEp PRIVATE NV_TRT_MINOR_RTX=${NV_TRT_MINOR_RTX})
endif()

# Pass DLL names to source code for delay load verification (Windows only)
if(WIN32)
    string(JOIN "|" _trt_rtx_additional_delayload_dlls ${_TRT_RTX_EP_EFFECTIVE_DELAYLOAD_DLLS})
    target_compile_definitions(TensorRTRtxEp PRIVATE
        TRT_RTX_DLL_NAME="${TRT_RTX_DLL_NAME}"
        TRT_ONNX_PARSER_DLL_NAME="${TRT_ONNX_PARSER_DLL_NAME}"
        TRT_RTX_EP_TARGET_ARCH="${TRT_RTX_EP_TARGET_ARCH}"
        TRT_RTX_EP_DEPENDENCY_SUBDIR="${TRT_RTX_EP_DEPENDENCY_SUBDIR}"
        TRT_RTX_EP_ADDITIONAL_DELAYLOAD_DLLS="${_trt_rtx_additional_delayload_dlls}"
    )
    
    # Production build flag for signature verification
    if(TRT_RTX_EP_PRODUCTION_BUILD)
        target_compile_definitions(TensorRTRtxEp PRIVATE TRT_RTX_EP_PRODUCTION_BUILD=1)
        message(STATUS "Production build: Signature verification ENABLED")
    else()
        message(STATUS "Test build: Signature verification DISABLED")
    endif()
endif()

# Include directories
target_include_directories(TensorRTRtxEp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}   
)

if (USE_VCPKG MATCHES "OFF")
    target_include_directories(TensorRTRtxEp PRIVATE
        "${DEPS_PATH}/onnx-src"
        "${DEPS_PATH}/onnx-build"
        "${DEPS_PATH}/protobuf-src/src"
        "${DEPS_PATH}/abseil-src"
    )
endif()

# Link libraries
target_link_libraries(TensorRTRtxEp PRIVATE
    onnxruntime::headers
    tensorrt_rtx::tensorrt_rtx
    protobuf::libprotobuf
    onnx
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvml
)

if(TRT_RTX_EP_ENABLE_NVTX3)
    find_package(CUDAToolkit 12.9 REQUIRED CUDA::nvtx3)
    target_link_libraries(TensorRTRtxEp PRIVATE CUDA::nvtx3)
    target_compile_definitions(TensorRTRtxEp PRIVATE TRT_RTX_EP_ENABLE_NVTX3=1)
else()
    target_compile_definitions(TensorRTRtxEp PRIVATE TRT_RTX_EP_ENABLE_NVTX3=0)
endif()

# Link delay load support library (Windows only)
if(WIN32)
    target_link_libraries(TensorRTRtxEp PRIVATE delayimp d3d12 dxgi)
endif()

# =============================================================================
# Target Properties
# =============================================================================
set_target_properties(TensorRTRtxEp PROPERTIES
    OUTPUT_NAME "${TRT_RTX_EP_OUTPUT_NAME}"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
trt_rtx_set_target_output_directory(TensorRTRtxEp "${TRT_RTX_EP_BIN_DIR}")

# Apply DEF file on Windows to export symbols
if(WIN32)
    set_target_properties(TensorRTRtxEp PROPERTIES
        LINK_FLAGS ${TRT_EP_LIB_LINK_FLAG}
    )
endif()

# The provider and all of its delay-loaded runtime dependencies share bin with
# the host executables in the build tree.
set(_trt_rtx_ep_runtime_files "")
if(WIN32)
    if(TRT_RTX_DLL)
        list(APPEND _trt_rtx_ep_runtime_files "${TRT_RTX_DLL}")
    endif()
    if(TRT_ONNX_PARSER_DLL)
        list(APPEND _trt_rtx_ep_runtime_files "${TRT_ONNX_PARSER_DLL}")
    endif()

    get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION_RELEASE)
    if(NOT _cudart_loc)
        get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION)
    endif()
    if(_cudart_loc)
        get_filename_component(_cudart_dir "${_cudart_loc}" DIRECTORY)
        get_filename_component(_cudart_name "${_cudart_loc}" NAME_WE)
        # CUDA 12 toolkits ship the runtime DLL in bin/; CUDA 13 Windows toolkits (installer and redist
        # layouts alike) moved it to bin/x64/, so search both.
        find_file(CUDART_DLL
            NAMES "${_cudart_name}.dll" "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll" "cudart64_12.dll" "cudart64.dll"
            HINTS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_BIN_DIR}/x64"
                  "${_cudart_dir}" "${_cudart_dir}/../bin" "${_cudart_dir}/../bin/x64"
            REQUIRED)
        list(APPEND _trt_rtx_ep_runtime_files "${CUDART_DLL}")
    endif()
else()
    list(APPEND _trt_rtx_ep_runtime_files
        "${TRT_RTX_LIB}"
        "${TRT_ONNX_PARSER_LIB}"
    )

    get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION_RELEASE)
    if(NOT _cudart_loc)
        get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION)
    endif()
    if(_cudart_loc)
        list(APPEND _trt_rtx_ep_runtime_files "${_cudart_loc}")
    endif()
endif()

if(_trt_rtx_ep_runtime_files)
    add_custom_target(trt_rtx_ep_runtime_dependencies
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TRT_RTX_BIN_DIR}/$<CONFIG>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_trt_rtx_ep_runtime_files}
            "${TRT_RTX_BIN_DIR}/$<CONFIG>"
        COMMENT "Copying TensorRT RTX EP runtime dependencies"
        VERBATIM
    )
    add_dependencies(TensorRTRtxEp trt_rtx_ep_runtime_dependencies)
endif()
if(TARGET onnxruntime_runtime_dependencies)
    # Building the provider also produces a runnable bin/<configuration> layout.
    add_dependencies(TensorRTRtxEp onnxruntime_runtime_dependencies)
endif()
set(TRT_RTX_EP_RUNTIME_FILES ${_trt_rtx_ep_runtime_files})
unset(_trt_rtx_ep_runtime_files)

# =============================================================================
# Tests
# =============================================================================
# enable_testing() must be called unconditionally at the root level so CMake
# always generates CTestTestfile.cmake here (required for `ctest` to find tests).
enable_testing()
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# =============================================================================
# Benchmark
# =============================================================================
option(BUILD_BENCHMARK "Build cached EP-context benchmark tools" ON)
if(BUILD_BENCHMARK)
    add_subdirectory(tools/benchmark)
endif()

# =============================================================================
# Examples
# =============================================================================
option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
    # onnxruntimesetup.cmake (included by every example subdirectory) checks for
    # this interface target first. Pre-creating it here lets the examples piggy-back
    # on the SDK already located by the root project instead of requiring a
    # separate ONNX_RUNTIME_PATH variable.
    if(NOT TARGET onnxruntime_interface)
        find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB
            NAMES onnxruntime_providers_shared
            PATHS ${ONNXRUNTIME_LIB_DIR}
            NO_DEFAULT_PATH
        )
        add_library(onnxruntime_interface INTERFACE)
        target_include_directories(onnxruntime_interface SYSTEM INTERFACE
            ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(onnxruntime_interface INTERFACE
            ${ONNXRUNTIME_LIB}
            $<$<BOOL:${ONNXRUNTIME_PROVIDERS_SHARED_LIB}>:${ONNXRUNTIME_PROVIDERS_SHARED_LIB}>)
    endif()
    add_subdirectory(examples/cxx)
endif()

# =============================================================================
# Install layout
# =============================================================================
# The build tree intentionally keeps every runtime dependency in bin/<config>.
# Installation separates the deliverables so each can be shipped independently:
# regular examples and unittests carry their own ORT runtime, while the EP
# package contains the provider, TensorRT/CUDA dependencies, and its public
# provider-options header.
if(TARGET TensorRTRtxEp)
    install(TARGETS TensorRTRtxEp
        RUNTIME DESTINATION lib COMPONENT ep
        LIBRARY DESTINATION lib COMPONENT ep
        ARCHIVE DESTINATION lib COMPONENT ep)
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/tensorrt_rtx_provider_options.h"
        DESTINATION include
        COMPONENT ep)
    if(TRT_RTX_EP_RUNTIME_FILES)
        install(FILES ${TRT_RTX_EP_RUNTIME_FILES}
            DESTINATION lib
            COMPONENT ep)
    endif()
endif()

if(TARGET unittests)
    install(TARGETS unittests
        RUNTIME DESTINATION bin/tests COMPONENT tests
        LIBRARY DESTINATION bin/tests COMPONENT tests
        ARCHIVE DESTINATION lib COMPONENT tests)
    if(TRT_RTX_ORT_RUNTIME_FILES)
        install(FILES ${TRT_RTX_ORT_RUNTIME_FILES}
            DESTINATION bin/tests
            COMPONENT tests)
    endif()
endif()

get_property(_trt_rtx_cxx_examples GLOBAL PROPERTY TRT_RTX_CXX_EXAMPLE_TARGETS)
set(_trt_rtx_regular_examples)
foreach(_trt_rtx_example IN LISTS _trt_rtx_cxx_examples)
    if(NOT TARGET ${_trt_rtx_example})
        continue()
    endif()
    get_property(_trt_rtx_component TARGET ${_trt_rtx_example} PROPERTY TRT_RTX_INSTALL_COMPONENT)
    list(APPEND _trt_rtx_regular_examples ${_trt_rtx_example})
endforeach()

set(_trt_rtx_sample_assets
    "${CMAKE_BINARY_DIR}/examples/cxx/candy.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/cxx/assets/Input.png")
if(_trt_rtx_regular_examples)
    install(TARGETS ${_trt_rtx_regular_examples}
        RUNTIME DESTINATION bin/examples COMPONENT examples
        LIBRARY DESTINATION bin/examples COMPONENT examples
        ARCHIVE DESTINATION lib COMPONENT examples)
    install(FILES ${_trt_rtx_sample_assets}
        DESTINATION bin/examples
        OPTIONAL
        COMPONENT examples)
    if(TRT_RTX_ORT_RUNTIME_FILES)
        install(FILES ${TRT_RTX_ORT_RUNTIME_FILES}
            DESTINATION bin/examples
            COMPONENT examples)
    endif()
endif()
