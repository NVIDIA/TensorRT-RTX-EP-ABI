// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "trt_proto_preprocessing.h"

#include "clip_bound_compatibility.h"
#include "logical_output_compatibility.h"
#include "pooling_dilation_compatibility.h"

namespace trt_rtx_ep
{

LoweredQdqInfo RunTensorRtProtoPreprocessing(onnx::ModelProto& model_proto)
{
    auto lowered_qdq_info = RunQdqLoweringForTensorRt(model_proto);

    // Run logical-output cleanup after Q/DQ lowering so the final proto handed
    // to TRT never contains Chromium's redundant bool -> uint8 bridge when it
    // can be canonicalized safely to an int32 bridge.
    RunLogicalOutputCompatibilityForTensorRt(model_proto);

    // Canonicalize WebNN's unbounded clamp defaults before TRT parses Clip
    // bounds as finite activation parameters.
    RunClipBoundCompatibilityForTensorRt(model_proto);

    // Lower provably-equivalent dilated AveragePool/MaxPool cases into
    // primitive ops so TRT does not see unsupported pooling dilations.
    RunPoolingDilationCompatibilityForTensorRt(model_proto);
    return lowered_qdq_info;
}

}  // namespace trt_rtx_ep
