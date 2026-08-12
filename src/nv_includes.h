// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// File to include the required TRT headers with workarounds for warnings we can't fix or not fixed yet.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)  // Ignore warning C4100: unreferenced formal parameter
#pragma warning(disable : 4996)  // Ignore warning C4996: 'nvinfer1::IPluginV2' was declared deprecated
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// Weightless EPContext refit requires TensorRT-RTX >= 1.6: the weight-strip build capability plus the
// IRefitterObserver / RefitRecord / setRefitObserver parser API (parser 0.2.0) it depends on are only
// available from 1.6. TensorRT-RTX < 1.6 is NOT supported. The minimum is enforced at CMake configure
// time (cmake/tensorrt_rtx.cmake); this hard compile-time guard re-checks it so a build that bypasses
// CMake still fails clearly rather than silently. (TRT_MAJOR_RTX / TRT_MINOR_RTX come from
// NvInferVersion.h, included above; requiring >= 1.6 subsumes the previous NV_ONNX_PARSER_VERSION >= 200
// gate.) A per-GPU-arch functional floor also applies, e.g. SM120 -> >= 1.6.1.106, resolved at
// engine-build time (see the build path), not here.
#if (TRT_MAJOR_RTX < 1) || (TRT_MAJOR_RTX == 1 && TRT_MINOR_RTX < 6)
#error \
    "Weightless EPContext refit / weight-stripped engine requires TensorRT-RTX >= 1.6. Build against a TensorRT-RTX 1.6 or newer SDK."
#endif
// Always 1 in a supported build (TensorRT-RTX >= 1.6 is enforced above). Retained so the existing
// `#if TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED` guards in the sources continue to compile the feature in.
#define TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED 1
