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

# EP-Agnostic IO Binding and Device Tensors

This sample demonstrates hardware-agnostic memory allocation and IO binding using
ONNX Runtime's V2 APIs (`CreateMemoryInfo_V2`, `CopyTensors`, `GetSharedAllocator`).

## Key Concepts

- **EP-agnostic device memory**: Allocate GPU tensors without knowing which EP is active,
  using `CreateMemoryInfo_V2` with a runtime-discovered vendor ID.
- **CopyTensors**: Transfer data between CPU and GPU without vendor-specific APIs.
- **IO Binding**: Bind device tensors once, run inference multiple times without repeated transfers.

## How It Works

1. Register available EPs and set `PREFER_GPU` policy.
2. Create a session and discover which EP device handles the inputs.
3. Allocate device tensors using `CreateMemoryInfo_V2` with the discovered vendor ID.
4. Use `CopyTensors` to upload input data to GPU once.
5. Bind device tensors via `IoBinding` and run inference in a loop.
6. Copy output back to CPU after the loop.

## Model

This sample uses `candy.onnx` (neural style transfer) from
https://github.com/yakhyo/fast-neural-style-transfer (MIT license).

Both `candy.onnx` and `Input.png` (240x240) are provided automatically by the build system.

## Running

```
./devicetensors-datatransfer
```
