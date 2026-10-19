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

// Weightless EPContext refit requires TensorRT-RTX >= 1.7: the weight-strip build capability, the
// IRefitterObserver / RefitRecord API and the parse-time emission point the capture uses --
// IParser::setRefitObserver -- are all taken from the TensorRT-RTX 1.7 release (also in the 1.8
// development branch). Older SDKs are NOT supported for this feature. The minimum is enforced at CMake
// configure time (cmake/tensorrt_rtx.cmake); this hard compile-time guard re-checks it so a build that
// bypasses CMake still fails clearly rather than silently. (TRT_MAJOR_RTX / TRT_MINOR_RTX come from
// NvInferVersion.h, included above.) Whether a given GPU architecture supports weight stripping is
// still decided by the builder at engine-build time (see the build path), not here.
#if (TRT_MAJOR_RTX < 1) || (TRT_MAJOR_RTX == 1 && TRT_MINOR_RTX < 7)
#error \
    "Weightless EPContext refit / weight-stripped engine requires TensorRT-RTX >= 1.7 (IParser::setRefitObserver). Build against a TensorRT-RTX 1.7 or newer SDK."
#endif
// Always 1 in a supported build (TensorRT-RTX >= 1.7 is enforced above). Retained so the existing
// `#if TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED` guards in the sources continue to compile the feature in.
#define TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED 1
