# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Optional path to ONNX Runtime SDK root directory (must contain include/ and lib/)")
set(ONNXRUNTIME_VERSION "1.26.0" CACHE STRING "ONNX Runtime version to download when ONNXRUNTIME_ROOT is empty")
set(ONNXRUNTIME_DOWNLOAD_BASE_URL
    "https://github.com/microsoft/onnxruntime/releases/download"
    CACHE STRING "Base URL for ONNX Runtime release downloads")
string(REGEX REPLACE "^v" "" _onnxruntime_version "${ONNXRUNTIME_VERSION}")

set(_onnxruntime_arch "x64")
if(NOT ONNXRUNTIME_ROOT)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        if(_onnxruntime_arch STREQUAL "x64")
            set(_onnxruntime_package_platform "win-x64")
        else()
            message(FATAL_ERROR "Unsupported ONNX Runtime Windows architecture: ${_onnxruntime_arch}")
        endif()
        set(_onnxruntime_archive_extension "zip")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(_onnxruntime_arch STREQUAL "x64")
            set(_onnxruntime_package_platform "linux-x64")
            set(_onnxruntime_archive_extension "tgz")
        else ()
            message(FATAL_ERROR "Only ONNX Runtime linux-x64 downloads are configured. Set ONNXRUNTIME_ROOT for ${_onnxruntime_arch}.")
        endif()
    else()
        message(FATAL_ERROR "No ONNX Runtime download package configured for ${CMAKE_SYSTEM_NAME}. Set ONNXRUNTIME_ROOT explicitly.")
    endif()

    set(_onnxruntime_package_name "onnxruntime-${_onnxruntime_package_platform}-${_onnxruntime_version}")
    set(_onnxruntime_archive_name "${_onnxruntime_package_name}.${_onnxruntime_archive_extension}")
    string(REGEX REPLACE "/+$" "" _onnxruntime_download_base_url "${ONNXRUNTIME_DOWNLOAD_BASE_URL}")
    set(_onnxruntime_download_url "${_onnxruntime_download_base_url}/v${_onnxruntime_version}/${_onnxruntime_archive_name}")
    set(_onnxruntime_download_dir "${CMAKE_BINARY_DIR}/_deps/onnxruntime")
    set(_onnxruntime_archive_path "${_onnxruntime_download_dir}/${_onnxruntime_archive_name}")
    set(_onnxruntime_downloaded_root "${_onnxruntime_download_dir}/${_onnxruntime_package_name}")

    if(NOT EXISTS "${_onnxruntime_downloaded_root}/include/onnxruntime_cxx_api.h")
        file(MAKE_DIRECTORY "${_onnxruntime_download_dir}")
        if(NOT EXISTS "${_onnxruntime_archive_path}")
            message(STATUS "Downloading ONNX Runtime ${_onnxruntime_version} from ${_onnxruntime_download_url}")
            file(DOWNLOAD
                "${_onnxruntime_download_url}"
                "${_onnxruntime_archive_path}"
                SHOW_PROGRESS
                STATUS _onnxruntime_download_status
                TLS_VERIFY ON
            )
            list(GET _onnxruntime_download_status 0 _onnxruntime_download_code)
            if(NOT _onnxruntime_download_code EQUAL 0)
                list(GET _onnxruntime_download_status 1 _onnxruntime_download_message)
                message(FATAL_ERROR "Failed to download ONNX Runtime: ${_onnxruntime_download_message}")
            endif()
        endif()

        message(STATUS "Extracting ONNX Runtime to ${_onnxruntime_download_dir}")
        file(ARCHIVE_EXTRACT
            INPUT "${_onnxruntime_archive_path}"
            DESTINATION "${_onnxruntime_download_dir}"
        )
    endif()

    if(NOT EXISTS "${_onnxruntime_downloaded_root}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "Downloaded ONNX Runtime archive did not produce expected SDK root: ${_onnxruntime_downloaded_root}")
    endif()

    set(ONNXRUNTIME_ROOT "${_onnxruntime_downloaded_root}")
    message(STATUS "Using downloaded ONNX Runtime: ${ONNXRUNTIME_ROOT}")
endif()

message(STATUS "Using ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")
# The EP negotiates the ORT API version with the host at load time (see
# src/utils/ort_api_init.h). A single binary built against any
# ONNXRUNTIME_ROOT >= 1.24 will load on any ORT host whose API version is in
# [kMinSupportedOrtApiVersion, compile-time ORT_API_VERSION].
message(STATUS "Minimum supported ORT runtime: 1.24.x (API version 24)")

set(_ONNXRUNTIME_INCLUDE_HINT "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${_ONNXRUNTIME_INCLUDE_HINT}"
    NO_DEFAULT_PATH
    REQUIRED
)
unset(_ONNXRUNTIME_INCLUDE_HINT)

set(_onnxruntime_c_api_header "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
if(NOT EXISTS "${_onnxruntime_c_api_header}")
    message(FATAL_ERROR "ONNX Runtime header not found: ${_onnxruntime_c_api_header}")
endif()

file(READ "${_onnxruntime_c_api_header}" _onnxruntime_c_api_content)
string(REGEX MATCH "#[ \t]*define[ \t]+ORT_API_VERSION[ \t]+([0-9]+)" _onnxruntime_api_version_match "${_onnxruntime_c_api_content}")
if(NOT _onnxruntime_api_version_match)
    message(FATAL_ERROR "Could not determine ORT_API_VERSION from ${_onnxruntime_c_api_header}")
endif()

set(ONNXRUNTIME_API_VERSION "${CMAKE_MATCH_1}")
if(ONNXRUNTIME_API_VERSION LESS 24)
    message(FATAL_ERROR
        "ONNX Runtime headers are too old: ORT_API_VERSION=${ONNXRUNTIME_API_VERSION}. "
        "Minimum required API version is 24.")
endif()
message(STATUS "ONNX Runtime API version: ${ONNXRUNTIME_API_VERSION}")

unset(_onnxruntime_c_api_header)
unset(_onnxruntime_c_api_content)
unset(_onnxruntime_api_version_match)

unset(ONNXRUNTIME_LIB CACHE)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_LIB_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

if(WIN32)
    unset(ONNXRUNTIME_DLL CACHE)
    find_file(ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        PATHS "${ONNXRUNTIME_LIB_DIR}" "${ONNXRUNTIME_ROOT}/bin" "${ONNXRUNTIME_ROOT}"
        NO_DEFAULT_PATH
    )
    unset(ONNXRUNTIME_PROVIDERS_SHARED_DLL CACHE)
    find_file(ONNXRUNTIME_PROVIDERS_SHARED_DLL
        NAMES onnxruntime_providers_shared.dll
        PATHS "${ONNXRUNTIME_LIB_DIR}" "${ONNXRUNTIME_ROOT}/bin" "${ONNXRUNTIME_ROOT}"
        NO_DEFAULT_PATH
    )
    unset(ONNXRUNTIME_PROVIDERS_SHARED_LIB CACHE)
    find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB
        NAMES onnxruntime_providers_shared
        PATHS "${ONNXRUNTIME_LIB_DIR}"
        NO_DEFAULT_PATH
    )
else()
    unset(ONNXRUNTIME_PROVIDERS_SHARED_LIB CACHE)
    find_library(ONNXRUNTIME_PROVIDERS_SHARED_LIB
        NAMES onnxruntime_providers_shared
        PATHS "${ONNXRUNTIME_LIB_DIR}"
        NO_DEFAULT_PATH
    )
endif()

if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
endif()

if(ONNXRUNTIME_PROVIDERS_SHARED_LIB AND NOT TARGET onnxruntime::providers_shared)
    add_library(onnxruntime::providers_shared UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::providers_shared PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_PROVIDERS_SHARED_LIB}"
    )
endif()

if(NOT TARGET onnxruntime_interface)
    add_library(onnxruntime_interface INTERFACE)
    target_link_libraries(onnxruntime_interface INTERFACE onnxruntime::onnxruntime)
    if(TARGET onnxruntime::providers_shared)
        target_link_libraries(onnxruntime_interface INTERFACE onnxruntime::providers_shared)
    endif()
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
