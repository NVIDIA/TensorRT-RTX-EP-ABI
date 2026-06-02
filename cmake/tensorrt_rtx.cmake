# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(TRT_RTX_ROOT "" CACHE PATH "Path to TensorRT RTX SDK root directory (must contain include/ and lib/)")
if(NOT TRT_RTX_ROOT)
    message(FATAL_ERROR "TRT_RTX_ROOT must be set. "
                        "Example: cmake -DTRT_RTX_ROOT=/path/to/tensorrt-rtx -B build")
endif()

message(STATUS "Using TRT_RTX_ROOT: ${TRT_RTX_ROOT}")

set(TRT_RTX_INCLUDE_DIR "${TRT_RTX_ROOT}/include")
set(TRT_RTX_LIB_DIR "${TRT_RTX_ROOT}/lib")

find_path(TENSORRT_RTX_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS "${TRT_RTX_ROOT}"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    REQUIRED
)

set(NV_TRT_MAJOR_RTX "")
set(NV_TRT_MINOR_RTX "")
if(EXISTS "${TENSORRT_RTX_INCLUDE_DIR}/NvInferVersion.h")
    file(READ "${TENSORRT_RTX_INCLUDE_DIR}/NvInferVersion.h" NVINFER_VER_CONTENT)
    string(REGEX MATCH "define TRT_MAJOR_RTX * +([0-9]+)" _nv_trt_major_match "${NVINFER_VER_CONTENT}")
    if(_nv_trt_major_match)
        set(NV_TRT_MAJOR_RTX "${CMAKE_MATCH_1}")
    endif()
    string(REGEX MATCH "define TRT_MINOR_RTX * +([0-9]+)" _nv_trt_minor_match "${NVINFER_VER_CONTENT}")
    if(_nv_trt_minor_match)
        set(NV_TRT_MINOR_RTX "${CMAKE_MATCH_1}")
    endif()
endif()

if(NV_TRT_MAJOR_RTX)
    message(STATUS "NV_TRT_MAJOR_RTX is ${NV_TRT_MAJOR_RTX}")
    message(STATUS "NV_TRT_MINOR_RTX is ${NV_TRT_MINOR_RTX}")
else()
    message(STATUS "Can't find NV_TRT_MAJOR_RTX macro")
endif()

if(WIN32 AND NV_TRT_MAJOR_RTX)
    set(TRT_RTX_LIB_NAME "tensorrt_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}")
    set(TRT_ONNX_PARSER_LIB_NAME "tensorrt_onnxparser_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}")
else()
    set(TRT_RTX_LIB_NAME "tensorrt_rtx")
    set(TRT_ONNX_PARSER_LIB_NAME "tensorrt_onnxparser_rtx")
endif()

set(TRT_RTX_DLL_NAME "${TRT_RTX_LIB_NAME}.dll")
set(TRT_ONNX_PARSER_DLL_NAME "${TRT_ONNX_PARSER_LIB_NAME}.dll")

message(STATUS "Looking for ${TRT_RTX_LIB_NAME}")
message(STATUS "Looking for ${TRT_ONNX_PARSER_LIB_NAME}")

find_library(TRT_RTX_LIB
    NAMES ${TRT_RTX_LIB_NAME}
    PATHS "${TRT_RTX_LIB_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(TRT_ONNX_PARSER_LIB
    NAMES ${TRT_ONNX_PARSER_LIB_NAME}
    PATHS "${TRT_RTX_LIB_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

if(WIN32)
    find_file(TRT_RTX_DLL
        NAMES ${TRT_RTX_DLL_NAME}
        HINTS "${TRT_RTX_ROOT}" "${TRT_RTX_LIB_DIR}" "${TRT_RTX_ROOT}/bin" "${TRT_RTX_ROOT}/../bin"
        NO_DEFAULT_PATH
    )
    find_file(TRT_ONNX_PARSER_DLL
        NAMES ${TRT_ONNX_PARSER_DLL_NAME}
        HINTS "${TRT_RTX_ROOT}" "${TRT_RTX_LIB_DIR}" "${TRT_RTX_ROOT}/bin" "${TRT_RTX_ROOT}/../bin"
        NO_DEFAULT_PATH
    )
    set(TRTRTX_DLL "${TRT_RTX_DLL}")
    set(TRTRTX_PARSER_DLL "${TRT_ONNX_PARSER_DLL}")
else()
    set(TRTRTX_LIB "${TRT_RTX_LIB}")
    set(TRTRTX_PARSER_LIB "${TRT_ONNX_PARSER_LIB}")
endif()

if(NOT TARGET tensorrt_rtx::runtime)
    add_library(tensorrt_rtx::runtime UNKNOWN IMPORTED)
    set_target_properties(tensorrt_rtx::runtime PROPERTIES
        IMPORTED_LOCATION "${TRT_RTX_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_RTX_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TENSORRT_RTX_INCLUDE_DIR}"
    )
endif()

if(NOT TARGET tensorrt_rtx::onnx_parser)
    add_library(tensorrt_rtx::onnx_parser UNKNOWN IMPORTED)
    set_target_properties(tensorrt_rtx::onnx_parser PROPERTIES
        IMPORTED_LOCATION "${TRT_ONNX_PARSER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_RTX_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TENSORRT_RTX_INCLUDE_DIR}"
    )
endif()

if(NOT TARGET tensorrt_rtx::tensorrt_rtx)
    add_library(tensorrt_rtx::tensorrt_rtx INTERFACE IMPORTED)
    set_target_properties(tensorrt_rtx::tensorrt_rtx PROPERTIES
        INTERFACE_LINK_LIBRARIES "tensorrt_rtx::runtime;tensorrt_rtx::onnx_parser"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_RTX_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TENSORRT_RTX_INCLUDE_DIR}"
    )
endif()

message(STATUS "TensorRT RTX include: ${TENSORRT_RTX_INCLUDE_DIR}")
message(STATUS "TensorRT RTX library: ${TRT_RTX_LIB}")
message(STATUS "TensorRT ONNX Parser library: ${TRT_ONNX_PARSER_LIB}")
