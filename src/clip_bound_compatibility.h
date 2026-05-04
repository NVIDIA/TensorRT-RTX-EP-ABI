// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/onnx_pb.h"

namespace trt_rtx_ep
{

// Rewrites ONNX Clip nodes that use WebNN's unbounded clamp defaults into
// parser-friendly ONNX nodes before TRT-RTX sees the proto.
//
// WebNN clamp with an omitted min/max means "no lower/upper clamp". Chromium's
// ONNX lowering can represent that as Clip bounds of -inf/+inf, but TRT-RTX's
// parser maps Clip to activation alpha/beta parameters that must be finite.
//
// Safe rewrites:
//   Clip(x, -inf, +inf)       -> Identity(x)
//   Clip(x, finite_min, +inf) -> Max(x, finite_min)
//   Clip(x, -inf, finite_max) -> Min(x, finite_max)
//
// Missing ONNX Clip bounds default to finite dtype limits, not infinity, so
// they are intentionally left untouched. Explicit cases that would need to
// produce infinities, such as min=+inf or max=-inf, are also left untouched.
void RunClipBoundCompatibilityForTensorRt(onnx::ModelProto& model_proto);

}  // namespace trt_rtx_ep
