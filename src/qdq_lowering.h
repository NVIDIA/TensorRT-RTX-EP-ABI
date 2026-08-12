#pragma once

// Policy-driven Q/DQ lowering for the TensorRT-RTX execution provider.
//
// The TRT-RTX ONNX parser accepts a *subset* of ONNX Q/DQ: it handles the
// symmetric, low-bit patterns that its quantization-aware fuser is tuned for
// (e.g. int8/uint8 Q/DQ around Gemm/MatMul/Conv/ConvTranspose with zp == 0),
// but it does not accept:
//   * asymmetric Q/DQ (zero_point != 0),
//   * int32 / uint32 as DequantizeLinear input types.
//
// In addition, this module deliberately avoids lowering some otherwise-valid
// ONNX Q/DQ patterns when doing so would be numerically unsafe (for example,
// QuantizeLinear to int16/uint16 with fp16/bf16 scales, or QuantizeLinear to
// int32/uint32 where a floating rewrite cannot represent the full integer
// range exactly).
//
// Rather than lose an entire subgraph or silently miscompile, this module
// rewrites *only* those non-native Q/DQ patterns into equivalent ONNX
// arithmetic before the proto is handed to TRT, while leaving the native-
// friendly patterns completely untouched so TRT can still apply its fusions.
//
// The rewrite is run twice with identical inputs:
//   1. During capability discovery (GetSupportedList), so the set of nodes
//      we claim must match what TRT will actually parse.
//   2. During engine build (CreateNodeComputeInfoFromGraph), so TRT builds
//      against the same graph we advertised support for.
//
// Both call sites MUST call RunQdqLoweringForTensorRt on the serialized
// proto. Any drift between the two (e.g. policy change that only runs on
// one side) would cause TRT to refuse nodes that ORT already partitioned
// to this EP.

#include <cstddef>
#include <string>
#include <vector>

#include "onnx/onnx_pb.h"

namespace trt_rtx_ep
{

// One folded DequantizeLinear record.
//
// Constant DQ is folded to a float initializer + trivial Identity. When TRT
// later prunes that Identity, the parser's supported-node list can lose the
// mapping back to the original ORT node id. The provider uses this struct to
// reattach the original ORT node to any parser subgraph that still references
// `output_name`, so ORT partition ownership matches the pre-lowered graph.
struct FoldedQdqNodeInfo
{
    size_t original_node_id = 0;
    std::string output_name;
};

// Summary of what lowering did to the proto.
//
//   lowered_node_count     : number of Q/DQ nodes that were rewritten or
//                            folded (diagnostic counter, not load-bearing).
//   folded_constant_nodes  : must be replayed by the provider during the
//                            proto -> ORT node remap (see comments above).
struct LoweredQdqInfo
{
    size_t lowered_node_count = 0;
    std::vector<FoldedQdqNodeInfo> folded_constant_nodes;
};

// Rewrites Q/DQ nodes in `model_proto` in-place according to the policy
// described at the top of this header. Non-Q/DQ nodes, and Q/DQ nodes that
// fall outside the lowering policy, are left byte-for-byte unchanged so they
// continue to go through TRT's native Q/DQ path.
//
// Preconditions: initializer payloads may be inline (raw_data / typed fields)
// or external with location == "_MEM_ADDR_" and offset == reinterpret_cast<
// int64_t>(raw_ptr) -- the convention produced by OrtGraphToProto's custom
// initializer handler in the provider.
LoweredQdqInfo RunQdqLoweringForTensorRt(onnx::ModelProto& model_proto);

}  // namespace trt_rtx_ep
