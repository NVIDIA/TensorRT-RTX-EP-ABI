// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/onnx_pb.h"

namespace trt_rtx_ep
{

// Rewrites ONNX AveragePool/MaxPool nodes with non-unit dilations into
// parser-friendly primitive ONNX nodes before TRT-RTX sees the proto.
//
// TensorRT-RTX currently rejects pooling dilations other than 1. For cases
// where every dilated pooling window maps to real input elements, a dilated
// pool is exactly equivalent to:
//   * Slice each tapped kernel position across all output locations.
//   * AveragePool: Add all tap tensors and divide by tap count.
//   * MaxPool: take elementwise Max across all tap tensors.
//
// The pass intentionally lowers only the subset it can prove equivalent:
// static-shape pooling, zero explicit padding, one output, and no partial
// windows. Padding/ceil partial-window cases need additional masking/divisor
// logic and are left native rather than approximated.
void RunPoolingDilationCompatibilityForTensorRt(onnx::ModelProto& model_proto);

}  // namespace trt_rtx_ep
