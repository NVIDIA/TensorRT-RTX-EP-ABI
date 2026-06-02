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

# Async Device Tensor Transfers with Pinned Memory

This sample demonstrates asynchronous host/device transfers with ONNX Runtime's
EP APIs while still allocating pinned host memory through EP-provided ORT
allocators. It avoids direct CUDA allocation and copy calls in application code.

## Key Concepts

- **EP-provided pinned memory**: Discover the TensorRT RTX EP device's
  `OrtDeviceMemoryType_HOST_ACCESSIBLE` memory info and request its shared ORT
  allocator with `GetSharedAllocator`.
- **EP-provided device memory**: Allocate GPU tensors from the same EP using the
  `OrtDeviceMemoryType_DEFAULT` shared allocator.
- **Async `CopyTensors`**: Queue H2D and D2H copies on ORT `SyncStream` objects.
- **Sync notifications**: Use `SyncNotification` to order the upload stream
  before inference and to wait on the host only after the final output copy.
- **Async run submissions**: Set
  `kOrtRunOptionsConfigDisableSynchronizeExecutionProviders=1` so `Run` queues
  work on the provided compute stream without synchronizing at the end of each
  submission.

## How It Works

1. Register the TensorRT RTX EP and select a GPU EP device.
2. Create separate ORT `SyncStream` objects for upload and inference.
3. Pass the inference stream as `user_compute_stream` in EP provider options.
4. Allocate pinned CPU input/output tensors with the EP's
   `HOST_ACCESSIBLE` shared allocator.
5. Allocate GPU input/output tensors with the EP's default device allocator.
6. Upload input with `CopyTensors` on the upload stream and wait on the compute
   stream using `SyncNotification`.
7. Bind GPU tensors and submit multiple runs with disable synchronization set.
8. Queue the output copy back to pinned memory, wait once on the host, and save
   the image.

## Model

This sample uses `candy.onnx` (neural style transfer) from
https://github.com/yakhyo/fast-neural-style-transfer (MIT license).

Both `candy.onnx` and `Input.png` (240x240) are provided automatically by the
build system.

## Running

```bash
./devicetensors-datatransfer-async
```

## Nsight Systems Verification

To verify that transfers and inference submissions are queued asynchronously,
profile the sample on a CUDA-capable system with Nsight Systems, for example:

```bash
nsys profile --trace=cuda,nvtx,osrt --output devicetensors-datatransfer-async \
    ./devicetensors-datatransfer-async
```

The trace should show H2D and D2H copies issued on ORT streams and the sample's
`Run` calls returning without an end-of-run EP synchronization until the final
host wait.
