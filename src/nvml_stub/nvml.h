// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal NVML declarations used by the TensorRT RTX EP. Some CUDA toolkit
// layouts provide nvml.lib/nvidia-ml but omit nvml.h from the include tree.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0
} nvmlReturn_t;

#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80

nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length);
const char* nvmlErrorString(nvmlReturn_t result);

#ifdef __cplusplus
}
#endif
