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
# GoogleTest is only a build dependency; do not add its headers and libraries
# to this project's runtime install packages.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    URL_HASH SHA256=40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

if(NOT TARGET Vulkan::Headers)
    FetchContent_Declare(
        vulkan_headers
        URL https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/refs/tags/v1.4.344
        URL_HASH SHA256=88b86b441eed931fb4cba595aff46fbb14177d72274c1554411d6bbd2117e86f
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(vulkan_headers)
endif()

# The root project's runtime staging targets place core ONNX Runtime binaries in
# bin/<config> and TensorRT/CUDA dependencies only in bin/ep/<config>. Tests
# depend on TensorRTRtxEp below, so no test-local configure-time copies are
# needed (or permitted) here.

# =============================================================================
# Model download
# =============================================================================
set(RESNET18_MODEL_NAME "resnet18_Opset18_timm.onnx")
set(RESNET18_MODEL_URL
    "https://huggingface.co/onnxmodelzoo/resnet18_Opset18_timm/resolve/main/resnet18_Opset18_timm.onnx")
set(RESNET18_MODEL_PATH "${TRT_RTX_BIN_DIR}/${RESNET18_MODEL_NAME}")

file(MAKE_DIRECTORY "${TRT_RTX_BIN_DIR}")

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
    "${CMAKE_SOURCE_DIR}/src/multi_rotary_cache_concat_offset.cc"
    "${CMAKE_SOURCE_DIR}/src/pooling_dilation_compatibility.cc"
    "${CMAKE_SOURCE_DIR}/src/qdq_lowering.cc"
    "${CMAKE_SOURCE_DIR}/src/trt_proto_preprocessing.cc"
)
set(PROVIDER_OPTIONS_TEST_SOURCES
    "${CMAKE_SOURCE_DIR}/src/tensorrt_rtx_execution_provider_info.cc"
    "${CMAKE_SOURCE_DIR}/src/utils/path_string.cc"
)
set(ORT_MODEL_DUMP_SOURCE
    "${CMAKE_SOURCE_DIR}/src/utils/ort_model_dump.cc"
)

add_executable(unittests
    ${UNITTEST_SOURCES}
    ${PROTO_PREPROCESSING_TEST_SOURCES}
    ${PROVIDER_OPTIONS_TEST_SOURCES}
    ${ORT_MODEL_DUMP_SOURCE}
    # EP source under test directly (no TRT symbols linked -- weightless_refit.cc only uses the
    # nvonnxparser RefitTransformKind enum constants at compile time). See test_tensorrt_rtx_weightless_refit.cpp.
    ${CMAKE_SOURCE_DIR}/src/weightless_refit.cc
    ${CMAKE_SOURCE_DIR}/src/direct_cuda_stream_reader.cc
    ${CMAKE_SOURCE_DIR}/src/file_stream_writer.cc
)
add_dependencies(unittests TensorRTRtxEp)

set_target_properties(unittests PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${TRT_RTX_BIN_DIR}"
)

target_include_directories(unittests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}
    ${TRT_RTX_INCLUDE_DIR}   # NvOnnxParser.h etc. for weightless_refit.cc (enum constants only)
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
    onnxruntime::onnxruntime
    GTest::gtest
    Vulkan::Headers
    CUDA::cudart
    CUDA::cuda_driver
    protobuf::libprotobuf
    onnx
)

if (TRT_RTX_EP_ENABLE_NVTX3)
    target_link_libraries(unittests PRIVATE CUDA::nvtx3)
    target_compile_definitions(unittests PRIVATE TRT_RTX_EP_ENABLE_NVTX3=1)
else ()
    target_compile_definitions(unittests PRIVATE TRT_RTX_EP_ENABLE_NVTX3=0)
endif ()

option(TRT_RTX_ENABLE_CUPTI_TESTS "Enable CUPTI-based memcpy tests" ON)

if(TRT_RTX_ENABLE_CUPTI_TESTS)
    find_path(TRT_RTX_CUPTI_INCLUDE_DIR
        NAMES cupti.h
        HINTS
            "${CUDAToolkit_ROOT}/extras/CUPTI/include"
            "${CUDAToolkit_TARGET_DIR}/extras/CUPTI/include"
    )
endif()

if(TRT_RTX_ENABLE_CUPTI_TESTS AND TRT_RTX_CUPTI_INCLUDE_DIR)
    target_compile_definitions(unittests PRIVATE TRT_RTX_ENABLE_CUPTI_MEMCPY_TEST=1)
    target_include_directories(unittests PRIVATE "${TRT_RTX_CUPTI_INCLUDE_DIR}")
else()
    target_compile_definitions(unittests PRIVATE TRT_RTX_ENABLE_CUPTI_MEMCPY_TEST=0)
    message(STATUS "CUPTI headers were not found. The CUPTI memcpy test will be compiled as a skip.")
endif()

target_compile_definitions(unittests PRIVATE
    ONNX_NAMESPACE=onnx
    ONNX_ML
    ORT_EP_UTILS_TEST_MAX_INLINE_PROTO_BYTES=2048U
    # Prefer the provider library name in the current runtime directory; this
    # generated target path remains a fallback for tests run from elsewhere.
    EP_LIB_PATH="$<TARGET_FILE:TensorRTRtxEp>"
        NOMINMAX=1
)

install(FILES "${RESNET18_MODEL_PATH}"
        DESTINATION bin/tests
        COMPONENT tests)

# gtest_discover_tests executes the target binary after linking. That cannot be
# done reliably while cross-building ARM64/ARM64EC on an x64 host. Parse the
# test sources during configuration for those targets. Native Linux CI builds
# in a container without the NVIDIA driver library, so defer discovery until
# test time on a GPU runner with libcuda.so.1 available.
if(WIN32 AND
   (TRT_RTX_EP_TARGET_ARCH STREQUAL "arm64ec" OR
    TRT_RTX_EP_TARGET_ARCH STREQUAL "arm64"))
    gtest_add_tests(
        TARGET unittests
        SOURCES
            ${UNITTEST_SOURCES}
            ${PROTO_PREPROCESSING_TEST_SOURCES}
            ${PROVIDER_OPTIONS_TEST_SOURCES}
            ${ORT_MODEL_DUMP_SOURCE}
        TEST_LIST unittests_TESTS
    )
elseif(UNIX AND NOT APPLE)
    gtest_discover_tests(unittests DISCOVERY_MODE PRE_TEST)
else()
    gtest_discover_tests(unittests)
endif()
