// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "onnx/onnx_pb.h"

namespace trt_rtx_ep
{

// Adds the parser attribute that lets TRT-RTX attention plugins select the
// long rotary cache from a concatenated short/long cache tensor.
void RunMultiRotaryCacheConcatOffsetForTensorRt(onnx::ModelProto& model_proto, int64_t offset);

}  // namespace trt_rtx_ep
