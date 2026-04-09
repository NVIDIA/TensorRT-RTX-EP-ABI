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

cmake_minimum_required(VERSION 3.15)
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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Disable compiler-specific extensions for strict C++17 compliance

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
    
    # Enable Control Flow Guard for linker as well
    add_link_options(/guard:cf)
endif()

# Find CUDA Toolkit (required)
find_package(CUDAToolkit 12.9 REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime SDK root directory (must contain include/ and lib/)")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must be set. "
                        "Example: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime -B build")
endif()
message(STATUS "Using ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")

set(TRT_RTX_ROOT "" CACHE PATH "Path to TensorRT RTX SDK root directory (must contain include/ and lib/)")
if(NOT TRT_RTX_ROOT)
    message(FATAL_ERROR "TRT_RTX_ROOT must be set. "
                        "Example: cmake -DTRT_RTX_ROOT=/path/to/tensorrt-rtx -B build")
endif()
message(STATUS "Using TRT_RTX_ROOT: ${TRT_RTX_ROOT}")

# =============================================================================
# ONNX Runtime Configuration
# =============================================================================
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

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
if(NOT WIN32)
    list(FILTER TRT_RTX_EP_SOURCES EXCLUDE REGEX ".*secure_load\\.cc$")
endif()

# =============================================================================
# TensorRT RTX Configuration
# =============================================================================
set(TRT_RTX_INCLUDE_DIR "${TRT_RTX_ROOT}/include")
set(TRT_RTX_LIB_DIR "${TRT_RTX_ROOT}/lib")

# Detect TensorRT RTX version from NvInferVersion.h
find_path(TENSORRT_RTX_INCLUDE_DIR NvInfer.h
    HINTS ${TRT_RTX_ROOT}
    PATH_SUFFIXES include
)

if(TENSORRT_RTX_INCLUDE_DIR)
    file(READ ${TENSORRT_RTX_INCLUDE_DIR}/NvInferVersion.h NVINFER_VER_CONTENT)
    string(REGEX MATCH "define TRT_MAJOR_RTX * +([0-9]+)" NV_TRT_MAJOR_RTX "${NVINFER_VER_CONTENT}")
    string(REGEX REPLACE "define TRT_MAJOR_RTX * +([0-9]+)" "\\1" NV_TRT_MAJOR_RTX "${NV_TRT_MAJOR_RTX}")
    string(REGEX MATCH "define TRT_MINOR_RTX * +([0-9]+)" NV_TRT_MINOR_RTX "${NVINFER_VER_CONTENT}")
    string(REGEX REPLACE "define TRT_MINOR_RTX * +([0-9]+)" "\\1" NV_TRT_MINOR_RTX "${NV_TRT_MINOR_RTX}")

    if(NV_TRT_MAJOR_RTX)
        message(STATUS "NV_TRT_MAJOR_RTX is ${NV_TRT_MAJOR_RTX}")
        message(STATUS "NV_TRT_MINOR_RTX is ${NV_TRT_MINOR_RTX}")
    else()
        message(STATUS "Can't find NV_TRT_MAJOR_RTX macro")
    endif()

    # Construct versioned library names (Windows style)
    if(WIN32 AND NV_TRT_MAJOR_RTX)
        set(TRT_RTX_LIB_NAME "tensorrt_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}")
        set(TRT_ONNX_PARSER_LIB_NAME "tensorrt_onnxparser_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}")
    endif()
endif()

# Fallback to default library names if version detection failed
if(NOT TRT_RTX_LIB_NAME)
    set(TRT_RTX_LIB_NAME "tensorrt_rtx")
endif()

if(NOT TRT_ONNX_PARSER_LIB_NAME)
    set(TRT_ONNX_PARSER_LIB_NAME "tensorrt_onnxparser_rtx")
endif()

message(STATUS "Looking for ${TRT_RTX_LIB_NAME}")
message(STATUS "Looking for ${TRT_ONNX_PARSER_LIB_NAME}")

# Find TensorRT RTX libraries
find_library(TRT_RTX_LIB
    NAMES ${TRT_RTX_LIB_NAME}
    PATHS ${TRT_RTX_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(TRT_ONNX_PARSER_LIB
    NAMES ${TRT_ONNX_PARSER_LIB_NAME}
    PATHS ${TRT_RTX_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TRT_RTX_LIB)
    message(STATUS "Can't find ${TRT_RTX_LIB_NAME}")
endif()

if(NOT TRT_ONNX_PARSER_LIB)
    message(STATUS "Can't find ${TRT_ONNX_PARSER_LIB_NAME}")
endif()

message(STATUS "TensorRT RTX library: ${TRT_RTX_LIB}")
message(STATUS "TensorRT ONNX Parser library: ${TRT_ONNX_PARSER_LIB}")

# =============================================================================
# External Dependencies (FetchContent)
# =============================================================================
include(FetchContent)

# Add ONNX
FetchContent_Declare(
    onnx
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG        v1.18.0
)
FetchContent_MakeAvailable(onnx)

set(DEPS_PATH "${CMAKE_BINARY_DIR}/_deps")

# =============================================================================
# Build Configuration Options
# =============================================================================
option(TRT_RTX_EP_PRODUCTION_BUILD "Enable production build with signature verification" OFF)

# =============================================================================
# Platform-Specific Configuration
# =============================================================================
if(WIN32)
    # Use generator expressions for multi-config generators (e.g., Visual Studio)
    # where CMAKE_BUILD_TYPE is empty at configure time
    set(DEPS_LIBS
        "${DEPS_PATH}/onnx-build/$<CONFIG>/onnx.lib"
        "${DEPS_PATH}/onnx-build/$<CONFIG>/onnx_proto.lib"
    )

    set(TRT_EP_LIB_LINK_FLAG "-DEF:${CMAKE_SOURCE_DIR}/src/tensorrt_rtx_execution_provider.def")

    # =============================================================================
    # Delay Load Configuration for TensorRT RTX DLLs
    # =============================================================================
    # Construct DLL names for delay loading
    if(NV_TRT_MAJOR_RTX AND NV_TRT_MINOR_RTX)
        set(TRT_RTX_DLL_NAME "tensorrt_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}.dll")
        set(TRT_ONNX_PARSER_DLL_NAME "tensorrt_onnxparser_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}.dll")
    else()
        set(TRT_RTX_DLL_NAME "tensorrt_rtx.dll")
        set(TRT_ONNX_PARSER_DLL_NAME "tensorrt_onnxparser_rtx.dll")
    endif()

    message(STATUS "Delay load DLL: ${TRT_RTX_DLL_NAME}")
    message(STATUS "Delay load DLL: ${TRT_ONNX_PARSER_DLL_NAME}")

    # Add delay load linker flags
    set(TRT_EP_LIB_LINK_FLAG "${TRT_EP_LIB_LINK_FLAG} /DELAYLOAD:${TRT_RTX_DLL_NAME} /DELAYLOAD:${TRT_ONNX_PARSER_DLL_NAME}")
else()
    # Linux
    set(DEPS_LIBS
        "${DEPS_PATH}/onnx-build/libonnx.a"
        "${DEPS_PATH}/onnx-build/libonnx_proto.a"
    )
endif()

# =============================================================================
# Create Shared Library
# =============================================================================
if(WIN32)
    add_library(TensorRTRtxEp SHARED ${TRT_RTX_EP_SOURCES} ${TRT_RTX_EP_HEADERS} ${TRT_RTX_EP_VERSION_RC})
else()
    add_library(TensorRTRtxEp SHARED ${TRT_RTX_EP_SOURCES} ${TRT_RTX_EP_HEADERS})
endif()

# Target compile definitions (modern CMake - replaces add_definitions)
target_compile_definitions(TensorRTRtxEp PRIVATE
    ONNX_NAMESPACE=onnx
    ONNX_ML
    TRT_RTX_EP_VERSION="${TRT_RTX_EP_VERSION}"
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
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${TENSORRT_RTX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${DEPS_PATH}/onnx-src"
    "${DEPS_PATH}/onnx-build"
    "${DEPS_PATH}/protobuf-src/src"
    "${DEPS_PATH}/abseil-src"
)

# Link libraries
target_link_libraries(TensorRTRtxEp PRIVATE
    ${ONNXRUNTIME_LIB}
    ${TRT_RTX_LIB}
    ${TRT_ONNX_PARSER_LIB}
    protobuf::libprotobuf
    onnx
)

# Link CUDA runtime and NVML using standardized namespaced targets
if(CUDAToolkit_FOUND)
    target_link_libraries(TensorRTRtxEp PRIVATE 
        CUDA::cudart 
        CUDA::nvml
    )
    message(STATUS "CUDA found - linking cudart and nvml")
else()
    message(STATUS "CUDA not found - building without CUDA support")
endif()

# Link delay load support library (Windows only)
if(WIN32)
    target_link_libraries(TensorRTRtxEp PRIVATE delayimp)
endif()

# =============================================================================
# Target Properties
# =============================================================================
set_target_properties(TensorRTRtxEp PROPERTIES
    OUTPUT_NAME "onnxruntime_providers_nv_tensorrt_rtx"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Apply DEF file on Windows to export symbols
if(WIN32)
    set_target_properties(TensorRTRtxEp PROPERTIES
        LINK_FLAGS ${TRT_EP_LIB_LINK_FLAG}
    )
endif()
