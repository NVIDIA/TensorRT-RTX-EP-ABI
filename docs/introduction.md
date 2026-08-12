<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Introduction

The NVIDIA TensorRT RTX Execution Provider (EP) is an inference deployment solution
designed specifically for NVIDIA RTX GPUs, optimized for client-centric use cases.

It is built as a standalone plugin that implements the
[ORT Plugin EP ABI](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)
interfaces (`OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtDataTransferImpl`)
introduced in ONNX Runtime 1.23.0. It does not need to be built together with
ONNX Runtime itself.

The EP leverages [TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) to
accelerate ONNX models on RTX GPUs based on Ampere and later architectures
(GeForce RTX 30xx and above).

## Key benefits

- Small package footprint — optimized resource usage on end-user systems at
  just under 200 MB.
- Fast model compile and load times — leverages just-in-time compilation to
  build RTX hardware-optimized engines on end-user devices in seconds.
- Portability — seamlessly use cached engines across multiple RTX GPUs.

## Next steps

- [Installation](installation.md) — install via pip (Python wheel)
- [Build guide](build-guide.md) — build the EP from source
- [Integration guide](integration-guide.md) — use the EP in a C++ or Python application
