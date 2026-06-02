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
project(devicetensors-datatransfer-async-sample)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(onnxruntimesetup)

add_executable(devicetensors-datatransfer-async
        main.cpp
)

set_target_properties(devicetensors-datatransfer-async PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
)
target_link_libraries(devicetensors-datatransfer-async PRIVATE
        onnxruntime_interface
        example_utils
)

set(ONNX "candy.onnx")
set(CANDY_ONNX_PATH "${CMAKE_CURRENT_BINARY_DIR}/../candy.onnx")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${ONNX}")
    copy_file_to_bin_dir("${CMAKE_CURRENT_SOURCE_DIR}/${ONNX}")
elseif(EXISTS "${CANDY_ONNX_PATH}")
    copy_file_to_bin_dir("${CANDY_ONNX_PATH}")
else()
    message(STATUS "candy.onnx not found -- sample 21 will need the model at runtime")
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Input.png")
    copy_file_to_bin_dir("${CMAKE_CURRENT_SOURCE_DIR}/Input.png")
elseif(EXISTS "${ASSETS_DIR}/Input.png")
    copy_file_to_bin_dir("${ASSETS_DIR}/Input.png")
endif()

set_target_properties(devicetensors-datatransfer-async
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
)

copy_ep_dll_post_build(devicetensors-datatransfer-async)

add_test(
        NAME    examples/21_devicetensors-datatransfer-async
        COMMAND $<TARGET_FILE:devicetensors-datatransfer-async>
        WORKING_DIRECTORY $<TARGET_FILE_DIR:devicetensors-datatransfer-async>
)
set_tests_properties(examples/21_devicetensors-datatransfer-async PROPERTIES
        LABELS  "examples"
        TIMEOUT 300
)
