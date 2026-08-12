<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Integration guide

This guide covers integrating the TensorRT RTX Execution Provider into a C++ or
Python application, whether you installed it via the Python wheel or built it from source.

## Runtime dependencies

The EP plugin is a shared library that depends on ONNX Runtime, TensorRT RTX,
and CUDA at runtime. All dependencies must be discoverable by the application.

Copy the following files to your application directory or ensure they are on
the system `PATH`.

**From the build output:**

```text
onnxruntime_providers_nv_tensorrt_rtx.dll     (Windows)
libonnxruntime_providers_nv_tensorrt_rtx.so   (Linux)
```

**From the ONNX Runtime:**

```text
<ONNXRUNTIME_ROOT>\lib\onnxruntime.dll
<ONNXRUNTIME_ROOT>\lib\onnxruntime_providers_shared.dll
```

**From the TensorRT RTX:**

```text
<TRT_RTX_ROOT>\bin\tensorrt_rtx_<major>_<minor>.dll
<TRT_RTX_ROOT>\bin\tensorrt_onnxparser_rtx_<major>_<minor>.dll
```

**From the CUDA Toolkit:**

```text
<CUDA_PATH>\bin\cudart64_<version>.dll
```

> **Note:** Python wheel users (`pip install onnxruntime-ep-nv-tensorrt-rtx`) do not
> need to manage DLL deployment — the wheel bundles all runtime dependencies.

## C++ usage

The EP uses the ONNX Runtime V2 device-based EP API. The library is registered
dynamically at runtime, devices are enumerated, and the EP is appended to the session.

```cpp
#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <stdexcept>
#include <vector>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");
Ort::SessionOptions session_options;

// 1. Register the EP plugin library
env.RegisterExecutionProviderLibrary(
    "NvTensorRTRTXExecutionProvider",
    ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

// 2. Enumerate available EP devices and find TensorRT RTX
Ort::ConstEpDevice trt_device = {};
for (auto& ep_device : env.GetEpDevices()) {
    if (std::strcmp(ep_device.EpName(), "NvTensorRTRTXExecutionProvider") == 0) {
        trt_device = ep_device;
        break;
    }
}
if (!trt_device) {
    throw std::runtime_error("TensorRT RTX EP device not found");
}

// 3. Append the EP with provider options
Ort::KeyValuePairs ep_options;
ep_options.Add("enable_cuda_graph", "1");
ep_options.Add("nv_runtime_cache_path", "./cache");
std::vector<Ort::ConstEpDevice> devices = {trt_device};
session_options.AppendExecutionProvider_V2(env, devices, ep_options);

// 4. Create session and run inference
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

See the [C++ samples](cpp-samples.md) for complete runnable examples.

## Python usage

> **Note:** If you installed via `pip install onnxruntime-ep-nv-tensorrt-rtx`, see the
> [installation quick start](installation.md#quick-start) for the recommended pattern
> using `trt_ep.get_ep_name()` and `trt_ep.get_library_path()`. The example below
> uses explicit strings for source-build or custom DLL deployments.

```python
import onnxruntime as ort

# 1. Register the EP plugin library
ort.register_execution_provider_library(
    "NvTensorRTRTXExecutionProvider",
    "onnxruntime_providers_nv_tensorrt_rtx.dll")

# 2. Discover the TensorRT RTX EP device
trt_device = None
for ep_device in ort.get_ep_devices():
    if ep_device.ep_name == "NvTensorRTRTXExecutionProvider":
        trt_device = ep_device
        break

if trt_device is None:
    raise RuntimeError("TensorRT RTX EP device not found")

# 3. Add EP device to session options with provider options
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(
    [trt_device],
    {"enable_cuda_graph": "1", "nv_runtime_cache_path": "./cache"})

# 4. Create session and run inference
session = ort.InferenceSession("model.onnx", sess_opts=session_options)
result = session.run([], {"input": input_data})

# 5. Unregister before exit
del session
ort.unregister_execution_provider_library("NvTensorRTRTXExecutionProvider")
```

> **Note:** When using the Python wheel (`import onnxruntime_ep_nv_tensorrt_rtx`),
> the library path is resolved automatically — no need to specify the DLL path
> manually.

## Provider options

Options are passed as string key-value pairs when appending the EP. A few
commonly used options:

| Option | Description |
|--------|-------------|
| `enable_cuda_graph` | Set to `"1"` to enable CUDA graph capture for reduced kernel-launch overhead |
| `nv_runtime_cache_path` | Directory where compiled TRT engines are cached between sessions |
| `nv_max_workspace_size` | Maximum GPU memory (bytes) TensorRT-RTX may use during engine build |
| `nv_weight_streaming_budget` | GPU memory budget for weight streaming; `"-1"` lets TensorRT-RTX choose the budget automatically |

See the [provider options](_generated/provider-options.rst) page for the full reference including
types, defaults, and accepted values for every option.
