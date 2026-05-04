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

# =============================================================================
# Google Test
# =============================================================================
include(FetchContent)
# Prevent gtest from overriding compiler/CRT settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# =============================================================================
# Helper: copy a file into the test binary output directory.
# Handles multi-config generators (e.g. Visual Studio) by copying into every
# per-configuration subdirectory so the test executable always finds its DLLs.
# =============================================================================
macro(copy_to_test_bin file)
    get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_is_multi_config)
        foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${_config}")
            file(COPY "${file}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/${_config}"
                 FOLLOW_SYMLINK_CHAIN)
        endforeach()
    else()
        file(COPY "${file}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
             FOLLOW_SYMLINK_CHAIN)
    endif()
endmacro()

# =============================================================================
# Copy ORT runtime binaries into the test binary directory so the test
# executable loads the same ORT build it was compiled against.
# =============================================================================
if(WIN32)
    find_file(ORT_RUNTIME_DLL
        onnxruntime.dll
        HINTS ${ONNXRUNTIME_LIB_DIR}
        REQUIRED)
    find_file(ORT_PROVIDERS_SHARED_DLL
        onnxruntime_providers_shared.dll
        HINTS ${ONNXRUNTIME_LIB_DIR}
        REQUIRED)
    copy_to_test_bin("${ORT_RUNTIME_DLL}")
    copy_to_test_bin("${ORT_PROVIDERS_SHARED_DLL}")
else()
    find_library(ORT_PROVIDERS_SHARED_LIB
        onnxruntime_providers_shared
        HINTS ${ONNXRUNTIME_LIB_DIR}
        REQUIRED)
    copy_to_test_bin("${ONNXRUNTIME_LIB}")
    copy_to_test_bin("${ORT_PROVIDERS_SHARED_LIB}")
endif()

# =============================================================================
# Copy TensorRT RTX binaries into the test binary directory.
# =============================================================================
if(WIN32)
    find_file(TRT_RTX_DLL
        NAMES ${TRT_RTX_DLL_NAME}
        HINTS ${TRT_RTX_ROOT} ${TRT_RTX_LIB_DIR} "${TRT_RTX_ROOT}/bin" "${TRT_RTX_ROOT}/../bin"
        REQUIRED)
    find_file(TRT_ONNX_PARSER_DLL
        NAMES ${TRT_ONNX_PARSER_DLL_NAME}
        HINTS ${TRT_RTX_ROOT} ${TRT_RTX_LIB_DIR} "${TRT_RTX_ROOT}/bin" "${TRT_RTX_ROOT}/../bin"
        REQUIRED)
    copy_to_test_bin("${TRT_RTX_DLL}")
    copy_to_test_bin("${TRT_ONNX_PARSER_DLL}")
else()
    copy_to_test_bin("${TRT_RTX_LIB}")
    copy_to_test_bin("${TRT_ONNX_PARSER_LIB}")
endif()

# =============================================================================
# Copy CUDA runtime into the test binary directory.
# =============================================================================
if(WIN32)
    get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION_RELEASE)
    if(NOT _cudart_loc)
        get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION)
    endif()
    if(_cudart_loc)
        get_filename_component(_cudart_dir "${_cudart_loc}" DIRECTORY)
        get_filename_component(_cudart_name "${_cudart_loc}" NAME_WE)
        find_file(CUDART_DLL
            NAMES "${_cudart_name}.dll" "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll" "cudart64_12.dll" "cudart64.dll"
            HINTS "${CUDAToolkit_BIN_DIR}" "${_cudart_dir}" "${_cudart_dir}/../bin"
            REQUIRED)
        copy_to_test_bin("${CUDART_DLL}")
    endif()
else()
    get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION_RELEASE)
    if(NOT _cudart_loc)
        get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION)
    endif()
    if(_cudart_loc)
        copy_to_test_bin("${_cudart_loc}")
    endif()
endif()

# =============================================================================
# Model download
# =============================================================================
set(RESNET18_MODEL_NAME "resnet18_Opset18_timm.onnx")
set(RESNET18_MODEL_URL
    "https://huggingface.co/onnxmodelzoo/resnet18_Opset18_timm/resolve/main/resnet18_Opset18_timm.onnx")
set(RESNET18_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${RESNET18_MODEL_NAME}")

if(NOT EXISTS "${RESNET18_MODEL_PATH}")
    message(STATUS "Downloading ${RESNET18_MODEL_NAME} ...")
    file(DOWNLOAD "${RESNET18_MODEL_URL}" "${RESNET18_MODEL_PATH}"
         STATUS _dl_status
         SHOW_PROGRESS)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(WARNING "Failed to download ${RESNET18_MODEL_NAME} (${_dl_status}). "
                        "test_load_model will fail at runtime.")
        file(REMOVE "${RESNET18_MODEL_PATH}")
    endif()
endif()

# =============================================================================
# unittests
# =============================================================================
file(GLOB UNITTEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set(PROTO_PREPROCESSING_TEST_SOURCES
    "${CMAKE_SOURCE_DIR}/src/clip_bound_compatibility.cc"
    "${CMAKE_SOURCE_DIR}/src/logical_output_compatibility.cc"
    "${CMAKE_SOURCE_DIR}/src/pooling_dilation_compatibility.cc"
    "${CMAKE_SOURCE_DIR}/src/qdq_lowering.cc"
    "${CMAKE_SOURCE_DIR}/src/trt_proto_preprocessing.cc"
)

add_executable(unittests ${UNITTEST_SOURCES} ${PROTO_PREPROCESSING_TEST_SOURCES})

set_target_properties(unittests PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

target_include_directories(unittests PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}
)

if(USE_VCPKG MATCHES "OFF")
    target_include_directories(unittests PRIVATE
        "${DEPS_PATH}/onnx-src"
        "${DEPS_PATH}/onnx-build"
        "${DEPS_PATH}/protobuf-src/src"
        "${DEPS_PATH}/abseil-src"
    )
endif()

# GTest::gtest (not gtest_main) — test_main.cpp provides main()
target_link_libraries(unittests PRIVATE
    ${ONNXRUNTIME_LIB}
    GTest::gtest
    CUDA::cudart
    CUDA::cuda_driver
    protobuf::libprotobuf
    onnx
)

target_compile_definitions(unittests PRIVATE
    ONNX_NAMESPACE=onnx
    ONNX_ML
    MODEL_FILE="${RESNET18_MODEL_PATH}"
    EP_LIB_PATH="$<TARGET_FILE:TensorRTRtxEp>"
)

# Copy the built EP DLL next to the test executable after each build
add_custom_command(TARGET unittests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:TensorRTRtxEp>
        $<TARGET_FILE_DIR:unittests>
    COMMENT "Copying TRT RTX EP DLL to test directory"
)

# Registers each TEST()/TEST_F() as a separate CTest test case
gtest_discover_tests(unittests)
