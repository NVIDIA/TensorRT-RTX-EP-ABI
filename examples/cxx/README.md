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

# TensorRT RTX Execution Provider - Examples

C++ samples demonstrating how to use the **TensorRT RTX Execution Provider** with
**ONNX Runtime 1.24.2** using the V2 device-based EP APIs introduced in ORT 1.23.0.

These examples use `onnxruntime_providers_nv_tensorrt_rtx` built from this project,
together with [ONNX Runtime 1.24.2](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.2)
and [TensorRT RTX 1.4.0](https://developer.nvidia.com/tensorrt-rtx).

## Samples

| # | Sample | Description | Model |
|---|--------|-------------|-------|
| 10 | [EP Device Selection](10_ep-device-selection/) | Register EPs, enumerate devices, select by vendor/policy | `candy.onnx` |
| 20 | [Device Tensors & Data Transfer](20_devicetensors-datatransfer/) | EP-agnostic GPU memory, `CopyTensors`, IO binding | `candy.onnx` |
| 21 | [Async Device Tensor Transfers](21_devicetensors-datatransfer-async/) | EP-provided pinned memory, async `CopyTensors`, disable-sync runs | `candy.onnx` |
| 30 | [SyncStreams (CUDA)](30_syncstreams-cuda/) | Async upload + inference with `SyncStream`/`SyncNotification` | `candy.onnx` |
| 40 | [EP Context](40_ep-context/) | Pre-compile models for fast loading (file & buffer modes) | `candy.onnx` (default) |

All samples use the same V2 EP registration pattern:

```cpp
// 1. Register EP libraries dynamically
env.RegisterExecutionProviderLibrary("nv_tensorrt_rtx",
    ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

// 2. Enumerate and select EP devices
auto ep_devices = env.GetEpDevices();
Ort::ConstEpDevice selected_device = ep_devices.front();

// 3. Append selected devices to session options
Ort::KeyValuePairs ep_options;
ep_options.Add("key", "value");
std::vector<Ort::ConstEpDevice> devices = {selected_device};
session_options.AppendExecutionProvider_V2(env, devices, ep_options);
```

## Prerequisites

| Component | Version | Download |
|-----------|---------|----------|
| ONNX Runtime | 1.24.2 | [GitHub Release](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.2) |
| TensorRT RTX | 1.4.0 | [NVIDIA Developer](https://developer.nvidia.com/tensorrt-rtx) |
| CUDA Toolkit | 12.x | [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads) |
| CMake | 3.20+ | [cmake.org](https://cmake.org/download/) |
| Visual Studio | 2022+ | [visualstudio.com](https://visualstudio.microsoft.com/) |

**Build the TRT RTX EP first** by following the [Build Guide](../doc/BUILD_GUIDE.md). Copy the
resulting `onnxruntime_providers_nv_tensorrt_rtx.dll` into the ONNX Runtime `lib/` directory.

## Building the Examples

```bash
cmake -B build -S . \
    -DONNX_RUNTIME_PATH=<path/to/onnxruntime> \
    -DTRTRTX_RUNTIME_PATH=<path/to/TensorRT-RTX>
cmake --build build --config Release
```

- `ONNX_RUNTIME_PATH`: Directory containing ONNX Runtime with `include/` and `lib/` subdirectories.
  The `onnxruntime_providers_nv_tensorrt_rtx.dll` should be placed in the `lib/` directory.
- `TRTRTX_RUNTIME_PATH` (optional): Root of TensorRT RTX package. Ensures `tensorrt_rtx_1_*.dll`
  and `tensorrt_onnxparser_rtx_1_*.dll` are copied to the output directory.
  If omitted, these DLLs must be on the system PATH.
- `CUDAToolkit_ROOT` (optional): Set if CUDA is not found automatically (needed for sample 30).
  Sample 30 is skipped if CUDA is not available.

The shared CMake module [`cmake/onnxruntimesetup.cmake`](cmake/onnxruntimesetup.cmake) handles
finding ONNX Runtime and TensorRT RTX libraries, creates the `onnxruntime_interface` link target,
and copies all required DLLs (ORT core, EP provider, TRT RTX runtime + parser) to each sample's
output directory.

## Directory Layout

```
examples/cxx/
├── CMakeLists.txt                      # Root build (fetches lodepng, downloads candy.onnx)
├── README.md                           # This file
├── assets/
│   └── Input.png                       # Shared test image
├── cmake/
│   └── onnxruntimesetup.cmake          # Shared CMake setup
├── 10_ep-device-selection/
│   ├── main.cpp                        # EP registration & device selection
│   ├── utils.cpp/h, argparsing.h       # Helpers
│   └── README.md
├── 20_devicetensors-datatransfer/
│   ├── main.cpp                        # CopyTensors & IO binding
│   ├── utils.cpp/h                     # Image I/O helpers
│   └── README.md
├── 21_devicetensors-datatransfer-async/
│   ├── main.cpp                        # Async CopyTensors with EP pinned allocator
│   ├── utils.cpp/h                     # Image I/O helpers
│   └── README.md
├── 30_syncstreams-cuda/
│   ├── main.cpp                        # SyncStream + CUDA interop
│   ├── utils.cpp/h                     # Image I/O helpers
│   └── README.md
└── 40_ep-context/
    ├── sample.cpp                      # File-based EP context
    ├── sample_buffer.cpp               # Buffer-based EP context
    ├── utils.cpp/h                     # EP registration helpers
    └── README.md
```

## Models and Assets

- **`candy.onnx`** (samples 10, 20, 21, 30): Neural style transfer model from the
  [ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/candy-9) (opset 9).
  **Downloaded automatically** during CMake configure. Input: `[batch_size, 3, H, W]` float32 RGB.
- **`Input.png`** (samples 10, 20, 21, 30): A test image included in the `assets/` directory
  and copied to output directories at configure time.
- **Sample 40**: Defaults to `candy.onnx`; can also use any user-provided ONNX model via command-line args.

## Third-Party Dependencies

- **lodepng** (Zlib license): PNG encoding/decoding, fetched automatically via CMake FetchContent.
