#pragma once

#include <cstdint>

#include "qdq_lowering.h"

namespace trt_rtx_ep
{

// Runs every graph rewrite the EP needs before handing the serialized ONNX
// proto to TRT-RTX.
//
// This wrapper keeps provider call sites honest: capability discovery and
// engine build should both see the same rewritten proto, even as we add
// compatibility passes that are orthogonal to Q/DQ lowering.
//
// Current stages:
//   1. Policy-driven Q/DQ lowering / folding.
//   2. Logical-output compatibility cleanup for Chromium's bool->uint8 bridge.
//   3. Clip bound compatibility for WebNN clamp defaults represented as
//      -inf/+inf.
//   4. Pooling dilation compatibility for static, zero-padding AveragePool /
//      MaxPool cases TRT-RTX rejects natively.
//   5. Optional long-rope rotary cache offset propagation for TRT-RTX
//      attention plugins.
struct TensorRtProtoPreprocessingOptions
{
    int64_t multi_rotary_cache_concat_offset = 0;
};

LoweredQdqInfo RunTensorRtProtoPreprocessing(onnx::ModelProto& model_proto,
                                             const TensorRtProtoPreprocessingOptions& options = {});

}  // namespace trt_rtx_ep
