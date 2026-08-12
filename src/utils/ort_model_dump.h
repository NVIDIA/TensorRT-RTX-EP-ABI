// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_ORT_MODEL_DUMP_H_
#define INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_ORT_MODEL_DUMP_H_

#include "onnxruntime_cxx_api.h"

#include <string>

#include "onnx/onnx_pb.h"

namespace OrtEpUtils
{

/// <summary>
/// Dumps a ModelProto with initializers inline when it fits protobuf limits, otherwise uses an external data file.
/// </summary>
/// <param name="model_proto">ModelProto to update and serialize.</param>
/// <param name="onnx_path">Path to the ONNX model file to write.</param>
/// <param name="data_path">Path to the external initializer data file to write.</param>
/// <returns>An Ort::Status indicating success or an error.</returns>
Ort::Status DumpModelWithInitializers(onnx::ModelProto& model_proto, const std::string& onnx_path,
                                      const std::string& data_path);

}  // namespace OrtEpUtils

#endif  // INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_ORT_MODEL_DUMP_H_
