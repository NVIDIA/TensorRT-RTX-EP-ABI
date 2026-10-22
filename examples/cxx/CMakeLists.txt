# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(trt_rtx_ep_examples)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)
FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(lodepng)
add_library(lodepng_lib STATIC ${lodepng_SOURCE_DIR}/lodepng.cpp ${lodepng_SOURCE_DIR}/lodepng.h)
target_include_directories(lodepng_lib PUBLIC ${lodepng_SOURCE_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(onnxruntimesetup)

add_library(example_utils STATIC
        common/utils.cpp
)
target_include_directories(example_utils PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/common"
)
target_link_libraries(example_utils PUBLIC
        onnxruntime_interface
        lodepng_lib
)
target_compile_features(example_utils PUBLIC cxx_std_20)
set_target_properties(example_utils PROPERTIES
        CXX_EXTENSIONS OFF
)

# Download candy.onnx (neural style transfer) used by all samples.
set(CANDY_ONNX_URL "https://huggingface.co/onnxmodelzoo/candy-9/resolve/main/candy-9.onnx")
set(CANDY_ONNX_PATH "${CMAKE_CURRENT_BINARY_DIR}/candy.onnx")
if(NOT EXISTS "${CANDY_ONNX_PATH}")
    message(STATUS "Downloading candy.onnx ...")
    file(DOWNLOAD "${CANDY_ONNX_URL}" "${CANDY_ONNX_PATH}"
         STATUS _download_status
         SHOW_PROGRESS)
    list(GET _download_status 0 _download_code)
    if(NOT _download_code EQUAL 0)
        message(WARNING "Failed to download candy.onnx (status ${_download_status}). "
                        "Samples will not have the model available.")
        file(REMOVE "${CANDY_ONNX_PATH}")
    endif()
endif()

# Shared assets directory for Input.png used by samples 10, 20, and 30.
set(ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")

# Stage the shared model and input image next to every sample executable.
# Target-local post-build copies automatically follow each multi-config output
# directory, so individual sample CMake files only need to name their targets.
function(stage_candy_sample_assets)
    if(NOT EXISTS "${CANDY_ONNX_PATH}")
        message(WARNING "candy.onnx was not downloaded; samples will need it at runtime")
        return()
    endif()
    foreach(_sample_target IN LISTS ARGN)
        copy_file_to_target_runtime_dir(${_sample_target} "${CANDY_ONNX_PATH}")
        if(EXISTS "${ASSETS_DIR}/Input.png")
            copy_file_to_target_runtime_dir(${_sample_target} "${ASSETS_DIR}/Input.png")
        endif()
    endforeach()
endfunction()

# Add a Candy-based example with the common ORT links, EP dependency, shared
# assets, model compile definition, and configuration-aware output layout.
function(add_cxx_example target)
    cmake_parse_arguments(ARG
        "NO_COMMON_LIBRARIES;NO_EP_RUNTIME_DEPENDENCY"
        "RUNTIME_DIRECTORY;MODEL_DEFINITION;TEST_NAME;INSTALL_COMPONENT"
        "SOURCES;LIBRARIES"
        ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_cxx_example(${target}) requires SOURCES")
    endif()
    if(NOT ARG_RUNTIME_DIRECTORY)
        set(ARG_RUNTIME_DIRECTORY "${RUNTIME_DIRECTORY}")
    endif()
    if(NOT ARG_INSTALL_COMPONENT)
        set(ARG_INSTALL_COMPONENT examples)
    endif()

    add_executable(${target} ${ARG_SOURCES})
    if(NOT ARG_NO_COMMON_LIBRARIES)
        target_link_libraries(${target} PRIVATE onnxruntime_interface example_utils)
    endif()
    if(ARG_LIBRARIES)
        target_link_libraries(${target} PRIVATE ${ARG_LIBRARIES})
    endif()
    if(NOT ARG_NO_EP_RUNTIME_DEPENDENCY)
        add_ep_runtime_dependency(${target})
    endif()
    if(ARG_MODEL_DEFINITION)
        target_compile_definitions(${target} PRIVATE ${ARG_MODEL_DEFINITION}="candy.onnx")
    endif()

    stage_candy_sample_assets(${target})
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
        ARCHIVE_OUTPUT_DIRECTORY "${TRT_RTX_ARCHIVE_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${ARG_RUNTIME_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${ARG_RUNTIME_DIRECTORY}"
        TRT_RTX_INSTALL_COMPONENT "${ARG_INSTALL_COMPONENT}"
    )
    set_property(GLOBAL APPEND PROPERTY TRT_RTX_CXX_EXAMPLE_TARGETS ${target})
    if(ARG_TEST_NAME)
        add_test(
            NAME "${ARG_TEST_NAME}"
            COMMAND $<TARGET_FILE:${target}>
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${target}>
        )
        set_tests_properties("${ARG_TEST_NAME}" PROPERTIES
            LABELS "examples"
            TIMEOUT 300
        )
    endif()
endfunction()

add_subdirectory(10_ep-device-selection)
add_subdirectory(20_devicetensors-datatransfer)
add_subdirectory(21_devicetensors-datatransfer-async)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    add_subdirectory(30_syncstreams-cuda)
else()
    message(STATUS "CUDA Toolkit not found -- skipping 30_syncstreams-cuda")
endif()

add_subdirectory(40_ep-context)

# Allow `ctest` to discover example tests when the examples are built standalone
# (when built via the root CMakeLists.txt the root's enable_testing() already covers this).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
endif()
