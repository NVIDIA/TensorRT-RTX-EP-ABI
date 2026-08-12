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

cmake_minimum_required(VERSION 3.20)

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
# Project Version Configuration
# =============================================================================
# Default 0.0.0 for local/dev builds. CI overrides via:
#   cmake -DTRT_RTX_EP_VERSION=1.2.3 ..
set(TRT_RTX_EP_VERSION "0.0.0" CACHE STRING "EP version (set by CI)")

message(STATUS "TensorRT RTX EP version: ${TRT_RTX_EP_VERSION}")

# Split version string into components for Windows VERSIONINFO resource
string(REPLACE "." ";" TRT_RTX_EP_VERSION_LIST ${TRT_RTX_EP_VERSION})
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
    # /GS - Buffer security check (enabled by default, but explicit)
    # /Qspectre - Spectre mitigation
    # /guard:cf - Control Flow Guard
    # /sdl - Security Development Lifecycle checks
    add_compile_options(/GS /Qspectre /guard:cf /sdl)
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
find_package(CUDAToolkit 12.9 REQUIRED CUDA::cudart CUDA::cuda_driver CUDA::nvml)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorrt_rtx.cmake")

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

if(USE_VCPKG MATCHES "OFF")
    include(FetchContent)

    function(fetch_content_make_available_exclude_from_all)
        foreach(_name IN LISTS ARGN)
            string(TOLOWER "${_name}" _name_lower)
            FetchContent_GetProperties(${_name})
            if(NOT ${_name_lower}_POPULATED)
                FetchContent_Populate(${_name})
                if(EXISTS "${${_name_lower}_SOURCE_DIR}/CMakeLists.txt")
                    add_subdirectory("${${_name_lower}_SOURCE_DIR}" "${${_name_lower}_BINARY_DIR}" EXCLUDE_FROM_ALL)
                endif()
            endif()
        endforeach()
    endfunction()

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
        )
        set(protobuf_BUILD_TESTS         OFF CACHE INTERNAL "")
        set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE INTERNAL "")
        set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
        fetch_content_make_available_exclude_from_all(protobuf)
    endif()

    # Add ONNX
    FetchContent_Declare(
        onnx
        GIT_REPOSITORY https://github.com/onnx/onnx.git
        GIT_TAG        v1.21.0
    )

    if(_ORTTRTRTX_HOST_PROTOC)
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE "${_ORTTRTRTX_HOST_PROTOC}" CACHE INTERNAL "")
        set(Protobuf_PROTOC_EXECUTABLE    "${_ORTTRTRTX_HOST_PROTOC}" CACHE FILEPATH "" FORCE)
    endif()

    fetch_content_make_available_exclude_from_all(onnx)

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
)

if(NV_TRT_MAJOR_RTX)
    target_compile_definitions(TensorRTRtxEp PRIVATE NV_TRT_MAJOR_RTX=${NV_TRT_MAJOR_RTX})
endif()

if(NV_TRT_MINOR_RTX)
    target_compile_definitions(TensorRTRtxEp PRIVATE NV_TRT_MINOR_RTX=${NV_TRT_MINOR_RTX})
endif()

# Pass DLL names to source code for delay load verification (Windows only)
if(WIN32)
    target_compile_definitions(TensorRTRtxEp PRIVATE
        TRT_RTX_DLL_NAME="${TRT_RTX_DLL_NAME}"
        TRT_ONNX_PARSER_DLL_NAME="${TRT_ONNX_PARSER_DLL_NAME}"
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

# Link delay load support library (Windows only)
if(WIN32)
    target_link_libraries(TensorRTRtxEp PRIVATE delayimp d3d12 dxgi)
endif()

# =============================================================================
# Target Properties
# =============================================================================
set_target_properties(TensorRTRtxEp PROPERTIES
    OUTPUT_NAME "onnxruntime_providers_nv_tensorrt_rtx"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Apply DEF file on Windows to export symbols
if(WIN32)
    set_target_properties(TensorRTRtxEp PROPERTIES
        LINK_FLAGS ${TRT_EP_LIB_LINK_FLAG}
    )
endif()

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

    foreach(_example_target
        ep-device-selection
        devicetensors-datatransfer
        devicetensors-datatransfer-async
        syncstreams_cuda
        sample
        sample_buffer
        sample_weightless
        sample_weightless_buffer
    )
        if(TARGET ${_example_target})
            # DLL copy is handled by copy_ep_dll_post_build() in each example's own
            # CMakeLists.txt (add_custom_command(TARGET ...) must run in the same
            # directory scope the target was created in, so it can't be done here).
            add_dependencies(${_example_target} TensorRTRtxEp)
        endif()
    endforeach()
endif()
