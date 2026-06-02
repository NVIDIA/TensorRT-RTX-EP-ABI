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
project(ep-context-samples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(onnxruntimesetup)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(sample sample.cpp)
target_link_libraries(sample PRIVATE onnxruntime_interface example_utils)

add_executable(sample_buffer sample_buffer.cpp)
target_link_libraries(sample_buffer PRIVATE onnxruntime_interface example_utils)

set(ONNX "candy.onnx")
set(CANDY_ONNX_PATH "${CMAKE_CURRENT_BINARY_DIR}/../candy.onnx")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ONNX}")
    copy_file_to_bin_dir("${CMAKE_CURRENT_SOURCE_DIR}/${ONNX}")
elseif(EXISTS "${CANDY_ONNX_PATH}")
    copy_file_to_bin_dir("${CANDY_ONNX_PATH}")
else()
    message(STATUS "candy.onnx not found -- sample 40 will need a model at runtime")
endif()

set_target_properties(sample sample_buffer
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
)

copy_ep_dll_post_build(sample)

# sample: no required args, defaults to candy.onnx -> candy_compiled.onnx in the exe dir.
add_test(
    NAME    examples/40_ep-context/sample
    COMMAND $<TARGET_FILE:sample>
    WORKING_DIRECTORY $<TARGET_FILE_DIR:sample>
)
set_tests_properties(examples/40_ep-context/sample PROPERTIES
    LABELS  "examples"
    TIMEOUT 300
)
