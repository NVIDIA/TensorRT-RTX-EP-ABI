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

# ONNX Runtime EP Context Samples

These sample programs demonstrate how to use the **ONNX Runtime EP Context API** with the
**TensorRT RTX Execution Provider** (`nv_tensorrt_rtx`) using the V2 device-based
EP registration and selection APIs (same approach as the other examples in this repository).

## What is EP Context?

ONNX Runtime's EP Context feature allows:

* **Pre-compilation of models** with a specific Execution Provider (e.g., TensorRT RTX).
* Faster **loading of compiled models** by reusing previously generated execution engines.
* Two storage modes:
  * **Embedded Mode**: Compiled binary is embedded inside the ONNX file.
  * **External Mode**: Compiled binary is stored as an external file alongside the ONNX.
* Two ways of loading the models:
  * **Disk Load**: Load the model files from direct disk access.
  * **Buffer Load**: Load the models from memory buffers.

## EP Registration (V2 API)

These samples use the same EP registration pattern as the other examples in this repository:

```cpp
// 1. Register the EP library
env.RegisterExecutionProviderLibrary("nv_tensorrt_rtx",
    ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

// 2. Enumerate and select EP devices
auto ep_devices = env.GetEpDevices();
// Select the TensorRT RTX device...

// 3. Append EP to session options using V2 API
ortApi.SessionOptionsAppendExecutionProvider_V2(
    session_options, env, &trt_device, 1,
    option_keys, option_values, num_options);
```

## Samples Included

### sample (File-based Model)

* Loads an ONNX model from disk (defaults to `candy.onnx` next to the executable).
* Compiles it with the TensorRT RTX EP.
* Saves compiled ONNX file.
* Loads the compiled model and measures load times.

```
sample.exe                                                       # uses candy.onnx by default
sample.exe <input_model.onnx> <compiled_output.onnx> [embed_mode]
```

* `embed_mode`: `0` = external (default), `1` = embedded.

### sample_buffer (Buffer-based Model with External Initializers)

* Loads model and weights directly into memory buffers.
* Registers external initializers (for `.onnx.data` files).
* Compiles the model to an in-memory buffer.
* Loads the compiled EP Context model from memory.

```
sample_buffer.exe <input.onnx> <weights.onnx.data> <output.onnx> <ext_data_filename> [embed_mode]
```

* `ext_data_filename`: The name used for external data in the model (e.g., `model.onnx.data`).
* `embed_mode`: `0` = external (default), `1` = embedded.

## Performance (RTX 5090)

| Model | Normal Load (sec) | Compile (sec) | EP Context Load (sec) | EP Context + Cache (sec) |
|-------|-------------------|---------------|-----------------------|--------------------------|
| Deepseek qwen 14B - INT4 | 31.23 | 34.92 | 4.95 | 3.73 |
| Llama-3.1-8B-Instruct - FP16 | 28.26 | 30.87 | 6.78 | 6.03 |
| SD 3.5 - transformer | 107.30 | 121.26 | 24.81 | 9.08 |
