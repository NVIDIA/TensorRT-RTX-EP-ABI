# NVIDIA TensorRT RTX Execution Provider

The NVIDIA TensorRT RTX Execution Provider (EP) is an inference deployment solution designed specifically for NVIDIA RTX GPUs, optimized for client-centric use cases.

This EP is built as a **standalone plugin** (`onnxruntime_providers_nv_tensorrt_rtx.dll`) that implements the ORT EP ABI interfaces (`OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtDataTransferImpl`, etc.) introduced in ORT 1.23.0. It does **not** need to be built together with ONNX Runtime.

The TensorRT RTX EP leverages NVIDIA's [TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) engine to accelerate ONNX models on RTX GPUs. It supports RTX GPUs based on Ampere and later architectures (NVIDIA GeForce RTX 30xx and above).

**Benefits:**

- **Small package footprint** — optimized resource usage on end-user systems at just under 200 MB.
- **Faster model compile and load times** — leverages just-in-time compilation to build RTX hardware-optimized engines on end-user devices in seconds.
- **Portability** — seamlessly use cached models across multiple RTX GPUs.

## Contents

- [Build from Source](#build-from-source)
- [Usage](#usage)
- [Features](#features)
  - [CUDA Graph](#cuda-graph)
  - [EP Context Model](#ep-context-model)
  - [Runtime Cache](#runtime-cache)
- [Execution Provider Options](#execution-provider-options)
- [Performance Test](#performance-test)
- [Plugin Support](#plugin-support)
- [Project Layout](#project-layout)
- [Documentation](#documentation)
- [Examples](#examples)

## Build from Source

### Prerequisites

| Dependency | Minimum Version | Platform |
|------------|-----------------|----------|
| CMake | 3.15 | All |
| Visual Studio | 2019 or 2022 (Desktop C++ workload) | Windows |
| GCC / Clang | C++17-capable | Linux |
| CUDA Toolkit | 12.9+ | All |
| ONNX Runtime SDK | 1.24.0+ | All |
| TensorRT RTX SDK | 1.1.1+ | All |

### Quick Build

Configure and build using standard CMake commands. Three CMake cache variables control where the dependencies are found:

| CMake Variable | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| `CUDAToolkit_ROOT` | Path to the CUDA Toolkit installation (optional as it will be taken from environment) |
| `ONNXRUNTIME_ROOT` | Path to the ONNX Runtime SDK (contains `include/` and `lib/`)                         |
| `TRT_RTX_ROOT` | Path to the TensorRT RTX SDK (contains `include/` and `lib/`)                         |

**Windows**

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-x64-1.24.0" `
      -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-1.1.1.36"
cmake --build build --config Release
```

Note: If you already have protobuf installed on your system from e.g. `winget` this will conflict with cmake and fail the configuration. 

**Linux**

```bash
cmake -B build \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DTRT_RTX_ROOT=/path/to/tensorrt-rtx
cmake --build build
```

The output library is at:
- Windows: `build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll`
- Linux: `build/libonnxruntime_providers_nv_tensorrt_rtx.so`


## Usage

The TensorRT RTX EP uses the **V2 device-based EP API** introduced in ORT 1.23.0. The EP library is registered dynamically at runtime, then devices are enumerated and appended to the session.

### C/C++

```cpp
#include <onnxruntime_cxx_api.h>

OrtApi const& ortApi = Ort::GetApi();
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");
Ort::SessionOptions session_options;

// 1. Register the EP plugin library
ortApi.RegisterExecutionProviderLibrary(
    env, "NvTensorRTRTXExecutionProvider",
    ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

// 2. Enumerate available EP devices and find TensorRT RTX
const OrtEpDevice* const* ep_devices = nullptr;
size_t num_ep_devices;
ortApi.GetEpDevices(env, &ep_devices, &num_ep_devices);

const OrtEpDevice* trt_device = nullptr;
for (size_t i = 0; i < num_ep_devices; i++) {
    if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]),
              "NvTensorRTRTXExecutionProvider") == 0) {
        trt_device = ep_devices[i];
        break;
    }
}

// 3. Append the EP with provider options
const char* keys[]   = {"enable_cuda_graph"};
const char* values[] = {"1"};
ortApi.SessionOptionsAppendExecutionProvider_V2(
    session_options, env, &trt_device, 1,
    keys, values, 1);

// 4. Create session
Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

### Python

Register the EP plugin library, discover the EP device, and add it to session options with provider-specific options.

```python
import onnxruntime as ort

# 1. Register the EP plugin DLL
ort.register_execution_provider_library(
    "NvTensorRTRTXExecutionProvider",
    "onnxruntime_providers_nv_tensorrt_rtx.dll")

# 2. Discover the TensorRT RTX EP device
ep_devices = ort.get_ep_devices()
trt_device = None
for ep_device in ep_devices:
    if ep_device.ep_name == "NvTensorRTRTXExecutionProvider":
        trt_device = ep_device
        break

# 3. Add EP device to session options with provider options
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(
    [trt_device],
    {"nv_runtime_cache_path": "./cache"})

# 4. Create session and run inference
session = ort.InferenceSession("model.onnx", sess_options=session_options)
result = session.run([], {"input": input_data})

# 5. Cleanup: delete session before unregistering
del session
ort.unregister_execution_provider_library("NvTensorRTRTXExecutionProvider")
```

## Documentation

For detailed documentation on features, execution provider options, and API usage, see the official ONNX Runtime documentation:

- [NVIDIA TensorRT RTX Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)

## Examples

See [examples/cxx/README.md](examples/cxx/README.md) for build instructions and the full list of C++ samples demonstrating EP device selection, device tensors, CUDA stream interop, and EP context model workflows.
