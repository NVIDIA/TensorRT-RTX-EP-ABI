#pragma once

#include "onnx/onnx_pb.h"

namespace trt_rtx_ep
{

// Canonicalizes bool-output bridge patterns that are legal in the WebNN /
// Chromium stack but problematic for TRT-RTX's ONNX parser.
//
// Chromium currently materializes WebNN logical outputs as:
//
//   bool_tensor -> Cast(uint8) -> Cast(T)
//
// because ONNX logical ops produce bool while WebNN logical outputs are typed
// as uint8. TRT-RTX rejects UINT8 as an intermediate tensor on the parser
// path, even though the graph usually casts that bridge again immediately.
//
// This pass canonicalizes the bridge to int32 when doing so is provably
// semantics-preserving:
//   * the bridge input is a bool tensor,
//   * the bridge output is not a graph output,
//   * every downstream use is a Cast,
//   * and none of those downstream casts keep the value as uint8.
//
// The resulting graph becomes:
//
//   bool_tensor -> Cast(int32) -> Cast(T)
//
// Canonicalizing to one parser-friendly bridge type keeps the rewrite generic:
// multiple downstream casts can target different final types while still
// avoiding the unsupported UINT8 intermediate before TRT sees the serialized
// proto.
void RunLogicalOutputCompatibilityForTensorRt(onnx::ModelProto& model_proto);

}  // namespace trt_rtx_ep
