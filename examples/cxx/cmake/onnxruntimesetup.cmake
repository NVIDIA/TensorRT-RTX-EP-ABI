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
    # -----------------------------------------------------------------------
    # Building as part of the root project.
    # onnxruntime_interface was pre-created by the root; skip SDK discovery.
    # Populate the DLL / SO path variables used by the copy block below.
    # -----------------------------------------------------------------------
    if(WIN32 AND DEFINED ONNXRUNTIME_LIB_DIR)
        set(ONNXRUNTIME_DLL             "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
        set(ONNXRUNTIME_PROVIDERS_SHARED_DLL
                                        "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.dll")
        # TRT RTX DLLs may sit in lib/ or a sibling bin/ directory.
        if(DEFINED TRT_RTX_DLL_NAME AND DEFINED TRT_RTX_LIB_DIR)
            foreach(_dir
                    "${TRT_RTX_LIB_DIR}"
                    "${TRT_RTX_ROOT}/bin"
                    "${TRT_RTX_ROOT}/../bin")
                if(NOT TRTRTX_DLL AND EXISTS "${_dir}/${TRT_RTX_DLL_NAME}")
                    set(TRTRTX_DLL "${_dir}/${TRT_RTX_DLL_NAME}")
                endif()
                if(NOT TRTRTX_PARSER_DLL AND EXISTS "${_dir}/${TRT_ONNX_PARSER_DLL_NAME}")
                    set(TRTRTX_PARSER_DLL "${_dir}/${TRT_ONNX_PARSER_DLL_NAME}")
                endif()
            endforeach()
        endif()
    elseif(NOT WIN32 AND DEFINED ONNXRUNTIME_LIB_DIR)
        set(ONNXRUNTIME_PROVIDERS_SHARED_LIB
                                        "${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_shared.so")
        if(DEFINED TRT_RTX_LIB AND EXISTS "${TRT_RTX_LIB}")
            set(TRTRTX_LIB "${TRT_RTX_LIB}")
        endif()
        if(DEFINED TRT_ONNX_PARSER_LIB AND EXISTS "${TRT_ONNX_PARSER_LIB}")
            set(TRTRTX_PARSER_LIB "${TRT_ONNX_PARSER_LIB}")
        endif()
    endif()
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
endif()

if(NOT DEFINED TRT_RTX_BIN_DIR)
    set(TRT_RTX_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
endif()
if(NOT DEFINED TRT_RTX_EP_BIN_DIR)
    set(TRT_RTX_EP_BIN_DIR "${TRT_RTX_BIN_DIR}")
endif()
if(NOT DEFINED TRT_RTX_ARCHIVE_DIR)
    set(TRT_RTX_ARCHIVE_DIR "${CMAKE_BINARY_DIR}/lib")
endif()

set(RUNTIME_DIRECTORY "${TRT_RTX_BIN_DIR}")
set(EP_RUNTIME_DIRECTORY "${TRT_RTX_EP_BIN_DIR}")

function(copy_file_to_target_runtime_dir target file)
    if(file AND EXISTS "${file}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${file}"
                "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endif()
endfunction()

function(add_ep_runtime_dependency target)
    if(TARGET TensorRTRtxEp)
        add_dependencies(${target} TensorRTRtxEp)
    elseif(TARGET trt_rtx_example_ep_runtime_dependencies)
        add_dependencies(${target} trt_rtx_example_ep_runtime_dependencies)
    endif()
endfunction()

if(NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit QUIET)
endif()
if(CUDAToolkit_FOUND)
    get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION_RELEASE)
    if(NOT _cudart_loc)
        get_target_property(_cudart_loc CUDA::cudart IMPORTED_LOCATION)
    endif()
    if(WIN32 AND _cudart_loc)
        get_filename_component(_cudart_dir "${_cudart_loc}" DIRECTORY)
        get_filename_component(_cudart_name "${_cudart_loc}" NAME_WE)
        find_file(CUDART_DLL
            NAMES "${_cudart_name}.dll" "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll" "cudart64_12.dll" "cudart64.dll"
            HINTS "${CUDAToolkit_BIN_DIR}" "${_cudart_dir}" "${_cudart_dir}/../bin")
    endif()
endif()

set(_onnxruntime_example_runtime_files "")
if(WIN32)
    if(ONNXRUNTIME_DLL)
        list(APPEND _onnxruntime_example_runtime_files "${ONNXRUNTIME_DLL}")
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_DLL)
        list(APPEND _onnxruntime_example_runtime_files "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}")
    endif()
else()
    if(ONNXRUNTIME_LIB)
        list(APPEND _onnxruntime_example_runtime_files "${ONNXRUNTIME_LIB}")
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_LIB)
        list(APPEND _onnxruntime_example_runtime_files "${ONNXRUNTIME_PROVIDERS_SHARED_LIB}")
    endif()
endif()

if(NOT TARGET onnxruntime_runtime_dependencies AND _onnxruntime_example_runtime_files)
    add_custom_target(onnxruntime_runtime_dependencies
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIRECTORY}/$<CONFIG>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_onnxruntime_example_runtime_files}
            "${RUNTIME_DIRECTORY}/$<CONFIG>"
        COMMENT "Copying ONNX Runtime runtime dependencies"
        VERBATIM
    )
endif()
if(TARGET onnxruntime_runtime_dependencies)
    add_dependencies(onnxruntime_interface onnxruntime_runtime_dependencies)
endif()
unset(_onnxruntime_example_runtime_files)

# Stage the EP and all of its dependencies beside the executables in bin.
set(_trt_rtx_example_ep_runtime_files "")
if(WIN32)
    if(TRTRTX_DLL)
        list(APPEND _trt_rtx_example_ep_runtime_files "${TRTRTX_DLL}")
    endif()
    if(TRTRTX_PARSER_DLL)
        list(APPEND _trt_rtx_example_ep_runtime_files "${TRTRTX_PARSER_DLL}")
    endif()
    if(CUDART_DLL)
        list(APPEND _trt_rtx_example_ep_runtime_files "${CUDART_DLL}")
    endif()
else()
    if(TRTRTX_LIB)
        list(APPEND _trt_rtx_example_ep_runtime_files "${TRTRTX_LIB}")
    endif()
    if(TRTRTX_PARSER_LIB)
        list(APPEND _trt_rtx_example_ep_runtime_files "${TRTRTX_PARSER_LIB}")
    endif()
    if(_cudart_loc)
        list(APPEND _trt_rtx_example_ep_runtime_files "${_cudart_loc}")
    endif()
endif()

if(NOT TARGET TensorRTRtxEp AND _trt_rtx_example_ep_runtime_files)
    add_custom_target(trt_rtx_example_ep_runtime_dependencies
        COMMAND ${CMAKE_COMMAND} -E make_directory "${RUNTIME_DIRECTORY}/$<CONFIG>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_trt_rtx_example_ep_runtime_files}
            "${RUNTIME_DIRECTORY}/$<CONFIG>"
        COMMENT "Copying TensorRT RTX EP runtime dependencies"
        VERBATIM
    )
endif()
unset(_trt_rtx_example_ep_runtime_files)
