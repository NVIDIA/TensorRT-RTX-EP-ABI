<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# SyncStreams (CUDA)

This sample demonstrates how to use ORT's `SyncStream` and `SyncNotification` APIs for
asynchronous data transfers and synchronization between multiple CUDA streams.

## Key Concepts

- **Dual streams**: A separate upload stream handles CPU-to-GPU data transfers while
  the inference stream runs the model.
- **SyncNotifications**: Signal/wait primitives that synchronize work across streams
  without blocking the CPU.
- **Pinned memory**: Page-locked CPU memory enables truly asynchronous `CopyTensors`.
- **CUDA interop**: Direct `cudaMemcpy2DAsync` on the ORT compute stream for D2D copies.

## How It Works

1. Register TensorRT RTX and CUDA EPs.
2. Create two `SyncStream` objects (inference + upload) from the TRT RTX EP device.
3. Pass the inference stream as `user_compute_stream` to session options.
4. Allocate pinned CPU memory and GPU tensors.
5. Use `CopyTensors` on the upload stream for async H2D transfer.
6. Signal completion via `SyncNotification::Activate`, wait on the inference stream.
7. Run a D2D sub-region copy with `cudaMemcpy2DAsync` on the inference stream.
8. Bind and run inference, then copy results back.

## Model

This sample uses `candy.onnx` from
https://github.com/yakhyo/fast-neural-style-transfer (MIT license).

Both `candy.onnx` and `Input.png` (240x240) are provided automatically by the build system.

## Note

SyncStream is currently only tested with CUDA streams (CUDA EP and TRT RTX EP).

## Running

```
./syncstreams_cuda
```
