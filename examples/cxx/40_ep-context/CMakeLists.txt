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

add_cxx_example(sample
    SOURCES sample.cpp
    TEST_NAME examples/40_ep-context/sample)
add_cxx_example(sample_buffer SOURCES sample_buffer.cpp)
add_cxx_example(sample_weightless
    SOURCES sample_weightless.cpp
    NO_EP_RUNTIME_DEPENDENCY)
add_cxx_example(sample_weightless_buffer
    SOURCES sample_weightless_buffer.cpp
    NO_EP_RUNTIME_DEPENDENCY)

# sample: no required args, defaults to candy.onnx -> candy_compiled.onnx in the exe dir.
