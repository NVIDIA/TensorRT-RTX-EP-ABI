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

# EP Device Selection

ONNX Runtime provides an execution provider independent way of querying and selecting
inference devices via the V2 device-based APIs. This involves 3 steps:

### 1. Registration of execution provider libraries
```cpp
auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
env.RegisterExecutionProviderLibrary("nv_tensorrt_rtx",
    ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));
```

### 2. Querying and selecting EP Devices
```cpp
auto ep_devices = env.GetEpDevices();
auto selected_devices = my_ep_selection_function(ep_devices);

Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_V2(env, selected_devices, ep_options);
session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
```

### 3. Create an inference session
```cpp
Ort::Session session(env, ORT_TSTR("path/to/model.onnx"), session_options);
```

## Model

This sample uses `candy.onnx` (neural style transfer) from the
[ONNX Model Zoo](https://huggingface.co/onnxmodelzoo/candy-9).
The model is downloaded automatically during CMake configure.

Input: `[batch_size, 3, H, W]` float32 RGB. Output: stylized image with the same shape.

## Running

```
./ep-device-selection -i ./Input.png -o ./output.png
```

Run with `-h` for more options including device selection and EP policy configuration.
