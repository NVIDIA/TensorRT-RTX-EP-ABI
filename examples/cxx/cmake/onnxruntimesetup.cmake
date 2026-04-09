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

if(TARGET onnxruntime_interface)
else()
  set(ONNX_RUNTIME_PATH "$ENV{ONNX_RUNTIME_PATH}" CACHE PATH "Where to find ONNX runtime")
  if("${ONNX_RUNTIME_PATH}" STREQUAL "")
    message(FATAL_ERROR "Please specify cmake variable ONNX_RUNTIME_PATH! E.g. via -DONNX_RUNTIME_PATH=/path/to/onnxruntime")
  endif()
  set(TRTRTX_RUNTIME_PATH "$ENV{TRTRTX_RUNTIME_PATH}" CACHE PATH "Where to find TensorRT RTX")
  if("${TRTRTX_RUNTIME_PATH}" STREQUAL "")
    message(WARNING "Please specify cmake variable TRTRTX_RUNTIME_PATH! E.g. via -DTRTRTX_RUNTIME_PATH=/path/to/tensorrt_rtx. This will ensure all libraries are copied to the execution directory.")
  endif()

  # Core ORT libraries (required at link time)
  find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib REQUIRED)
  find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB onnxruntime_providers_shared HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib REQUIRED)

  # EP libraries (loaded at runtime via RegisterExecutionProviderLibrary, not linked)
  find_library(ONNXRUNTIME_TRT_EP_LIB onnxruntime_providers_nv_tensorrt_rtx HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib)
  find_library(ONNXRUNTIME_CUDA_EP_LIB onnxruntime_providers_cuda HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib)
  # Detect TensorRT RTX version from NvInferVersion.h instead of hardcoding
  set(_trt_rtx_lib_names "")
  set(_trt_rtx_parser_lib_names "")
  if(NOT "${TRTRTX_RUNTIME_PATH}" STREQUAL "")
    find_file(_nvinfer_ver_h NvInferVersion.h HINTS ${TRTRTX_RUNTIME_PATH}/include)
    if(_nvinfer_ver_h)
      file(READ "${_nvinfer_ver_h}" _ver_content)
      string(REGEX MATCH "define TRT_MAJOR_RTX * +([0-9]+)" _ "${_ver_content}")
      set(_trt_major "${CMAKE_MATCH_1}")
      string(REGEX MATCH "define TRT_MINOR_RTX * +([0-9]+)" _ "${_ver_content}")
      set(_trt_minor "${CMAKE_MATCH_1}")
      if(_trt_major)
        message(STATUS "Detected TensorRT RTX version: ${_trt_major}.${_trt_minor}")
        list(APPEND _trt_rtx_lib_names "tensorrt_rtx_${_trt_major}_${_trt_minor}")
        list(APPEND _trt_rtx_parser_lib_names "tensorrt_onnxparser_rtx_${_trt_major}_${_trt_minor}")
      endif()
    endif()
  endif()
  list(APPEND _trt_rtx_lib_names tensorrt_rtx)
  list(APPEND _trt_rtx_parser_lib_names tensorrt_onnxparser_rtx)

  if(WIN32)
    find_file(ONNXRUNTIME_DLL onnxruntime.dll HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib REQUIRED)
    find_file(ONNXRUNTIME_PROVIDERS_SHARED_DLL onnxruntime_providers_shared.dll HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib REQUIRED)
    find_file(ONNXRUNTIME_TRT_EP_DLL onnxruntime_providers_nv_tensorrt_rtx.dll HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib)
    find_file(ONNXRUNTIME_CUDA_EP_DLL onnxruntime_providers_cuda.dll HINTS ${ONNX_RUNTIME_PATH} ${ONNX_RUNTIME_PATH}/lib)

    set(_trt_rtx_dll_names "")
    set(_trt_rtx_parser_dll_names "")
    foreach(_name IN LISTS _trt_rtx_lib_names)
      list(APPEND _trt_rtx_dll_names "${_name}.dll")
    endforeach()
    foreach(_name IN LISTS _trt_rtx_parser_lib_names)
      list(APPEND _trt_rtx_parser_dll_names "${_name}.dll")
    endforeach()

    find_file(TRTRTX_DLL
        NAMES ${_trt_rtx_dll_names}
        HINTS ${TRTRTX_RUNTIME_PATH} ${TRTRTX_RUNTIME_PATH}/lib ${TRTRTX_RUNTIME_PATH}/bin
              ${TRTRTX_RUNTIME_PATH}/../bin)
    find_file(TRTRTX_PARSER_DLL
        NAMES ${_trt_rtx_parser_dll_names}
        HINTS ${TRTRTX_RUNTIME_PATH} ${TRTRTX_RUNTIME_PATH}/lib ${TRTRTX_RUNTIME_PATH}/bin
              ${TRTRTX_RUNTIME_PATH}/../bin)
  else()
    find_library(TRTRTX_LIB
        NAMES ${_trt_rtx_lib_names}
        HINTS ${TRTRTX_RUNTIME_PATH} ${TRTRTX_RUNTIME_PATH}/lib)
    find_library(TRTRTX_PARSER_LIB
        NAMES ${_trt_rtx_parser_lib_names}
        HINTS ${TRTRTX_RUNTIME_PATH} ${TRTRTX_RUNTIME_PATH}/lib)
  endif()

  find_path(ONNXRUNTIME_INCLUDE
      onnxruntime_cxx_api.h
      HINTS ${ONNX_RUNTIME_PATH}/include
      REQUIRED)
  add_library(onnxruntime_interface INTERFACE)
  target_include_directories(onnxruntime_interface SYSTEM INTERFACE ${ONNXRUNTIME_INCLUDE})
  target_link_libraries(onnxruntime_interface INTERFACE ${ONNXRUNTIME_LIB} ${ONNXRUNTIME_PROVIDERS_SHARED_LIB})

  message(STATUS "ONNX runtime include \"${ONNXRUNTIME_INCLUDE}\"")
  message(STATUS "ONNX runtime lib \"${ONNXRUNTIME_LIB}\"")
  if(ONNXRUNTIME_TRT_EP_LIB)
    message(STATUS "TRT RTX EP lib \"${ONNXRUNTIME_TRT_EP_LIB}\"")
  else()
    message(STATUS "TRT RTX EP lib not found (will be loaded at runtime)")
  endif()
endif()

set(RUNTIME_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

macro (copy_file_to_bin_dir file)
    get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(${is_multi_config})
      foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
          set(OUTPUT_DIR ${RUNTIME_DIRECTORY}/${config})
          execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}")
          file(COPY ${file} DESTINATION ${OUTPUT_DIR} FOLLOW_SYMLINK_CHAIN)
      endforeach()
  else()
      file(COPY ${file} DESTINATION ${RUNTIME_DIRECTORY} FOLLOW_SYMLINK_CHAIN)
  endif()
endmacro()

# Copy only the core ORT DLLs that are always needed.
# EP DLLs and TRT RTX runtime are optional — copied if found.
if(WIN32)
    copy_file_to_bin_dir(${ONNXRUNTIME_DLL})
    copy_file_to_bin_dir(${ONNXRUNTIME_PROVIDERS_SHARED_DLL})
    if(ONNXRUNTIME_TRT_EP_DLL)
        copy_file_to_bin_dir(${ONNXRUNTIME_TRT_EP_DLL})
    endif()
    if(TRTRTX_DLL)
        copy_file_to_bin_dir(${TRTRTX_DLL})
    endif()
    if(TRTRTX_PARSER_DLL)
        copy_file_to_bin_dir(${TRTRTX_PARSER_DLL})
    endif()
    if(ONNXRUNTIME_CUDA_EP_DLL)
        copy_file_to_bin_dir(${ONNXRUNTIME_CUDA_EP_DLL})
    endif()
else()
    copy_file_to_bin_dir(${ONNXRUNTIME_LIB})
    copy_file_to_bin_dir(${ONNXRUNTIME_PROVIDERS_SHARED_LIB})
    if(ONNXRUNTIME_TRT_EP_LIB)
        copy_file_to_bin_dir(${ONNXRUNTIME_TRT_EP_LIB})
    endif()
    if(TRTRTX_LIB)
        copy_file_to_bin_dir(${TRTRTX_LIB})
    endif()
    if(TRTRTX_PARSER_LIB)
        copy_file_to_bin_dir(${TRTRTX_PARSER_LIB})
    endif()
    if(ONNXRUNTIME_CUDA_EP_LIB)
        copy_file_to_bin_dir(${ONNXRUNTIME_CUDA_EP_LIB})
    endif()
endif()
