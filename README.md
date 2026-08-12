# NVIDIA TensorRT RTX Execution Provider

The NVIDIA TensorRT RTX Execution Provider (EP) is an inference deployment solution designed specifically for NVIDIA RTX GPUs, optimized for client-centric use cases.

This EP is built as a **standalone plugin** (`onnxruntime_providers_nv_tensorrt_rtx.dll`) that implements the ORT EP ABI interfaces (`OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtDataTransferImpl`, etc.) introduced in ORT 1.23.0. It does **not** need to be built together with ONNX Runtime.

The TensorRT RTX EP leverages NVIDIA's [TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) engine to accelerate ONNX models on RTX GPUs. It supports RTX GPUs based on Ampere and later architectures (NVIDIA GeForce RTX 30xx and above).

**Benefits:**

- **Small package footprint** — optimized resource usage on end-user systems at just under 200 MB.
- **Faster model compile and load times** — leverages just-in-time compilation to build RTX hardware-optimized engines on end-user devices in seconds.
- **Portability** — seamlessly use cached models across multiple RTX GPUs.

## Contents

- [Compatibility Matrix](#compatibility-matrix)
- [Build from Source](#build-from-source)
  - [Prerequisites](#prerequisites)
  - [Quick Build](#quick-build)
  - [Python Wheel](#python-wheel)
- [Usage](#usage)
  - [C/C++](#cc)
  - [Python](#python)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Compatibility Matrix

| EP Version | ORT Version | TRT RTX Version | Notes |
|------------|-------------|-----------------|-------|
| 0.1 | 1.24.0+ | 1.4.x.x | Initial Windows Support |
| 0.3 | 1.25.0+ | 1.5.x.x | Linux Support, Weight Streaming, CIG Interop, ORT Version Negotiation |

## Build from Source

### Prerequisites

| Dependency | Minimum Version | Platform | Notes |
|------------|-----------------|----------|-------|
| CMake | 3.15 | All | |
| Visual Studio | 2019 or 2022 (Desktop C++ workload) | Windows | |
| GCC / Clang | C++20-capable | Linux | |
| CUDA Toolkit | 12.9+ | All | |
| ONNX Runtime SDK | 1.24.0+ | All | |
| TensorRT RTX SDK | 1.1.1+ | All | |

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

**Windows with vcpkg Package Manager**

vcpkg can optionally be used to manage dependencies (protobuf, ONNX, abseil) instead of CMake FetchContent.

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-x64-1.24.0" `
      -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-1.1.1.36" `
      -DUSE_VCPKG=ON `
      -DCMAKE_TOOLCHAIN_FILE="..\vcpkg\scripts\buildsystems\vcpkg.cmake" `
      -DVCPKG_TARGET_TRIPLET=x64-windows-static-md `
      -DVCPKG_HOST_TRIPLET=x64-windows
cmake --build build --config Release
```

**Linux**

```bash
cmake -B build \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DTRT_RTX_ROOT=/path/to/tensorrt-rtx
cmake --build build
```

**Linux with vcpkg Package Manager**

```bash
cmake -B build \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DTRT_RTX_ROOT=/path/to/tensorrt-rtx \
      -DUSE_VCPKG=ON \
      -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DVCPKG_TARGET_TRIPLET=x64-linux \
      -DVCPKG_HOST_TRIPLET=x64-linux
cmake --build build
```

The output library is at:
- Windows: `build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll`
- Linux: `build/libonnxruntime_providers_nv_tensorrt_rtx.so`

**Building with Unit Tests**

Unit tests are built by default (`BUILD_TESTS=ON`). D3D12 graphics interop is compiled in automatically on Windows.

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 `
      -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-x64-1.25.0" `
      -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-1.1.1.36"
cmake --build build --config Release
```

Run the tests:

```powershell
build\tests\Release\unittests.exe
```

Note: CIG interop test cases require ORT SDK 1.25+ (`ORT_API_VERSION >= 25`). With ORT 1.24, only the EP registration smoke test compiles.

See [doc/BUILD_GUIDE.md](doc/BUILD_GUIDE.md) for the full build guide with troubleshooting and integration instructions.

### Python Wheel

Use the `--build_wheel` flag with the provided build scripts to produce a Python wheel after the C++ build.

**Windows**

```powershell
build.bat --cuda_home "C:\CUDA\v12.9" `
          --onnxruntime_home "C:\SDK\onnxruntime-win-x64-1.25.0" `
          --trt_rtx_home "C:\SDK\TensorRT-RTX-1.5.0.114" `
          --version 0.3.0 --build_wheel
```

Output:
- `build\dist\onnxruntime_ep_nv_tensorrt_rtx_cu12-0.3.0-py3-none-win_amd64.whl`
- `build\dist\onnxruntime_ep_nv_tensorrt_rtx-0.3.0-py3-none-any.whl` (meta wheel)

**Linux**

```bash
./build.sh --cuda_home /usr/local/cuda-12.9 \
           --onnxruntime_home /path/to/onnxruntime \
           --trt_rtx_home /path/to/tensorrt-rtx \
           --version 0.3.0 --build_wheel
```

Output:
- `build/dist/onnxruntime_ep_nv_tensorrt_rtx_cu12-0.3.0-py3-none-linux_x86_64.whl`
- `build/dist/onnxruntime_ep_nv_tensorrt_rtx-0.3.0-py3-none-any.whl` (meta wheel)

The meta wheel (`onnxruntime_ep_nv_tensorrt_rtx`) is a platform-independent package that declares a dependency on the appropriate CUDA-versioned wheel.


## Usage

The TensorRT RTX EP uses the **V2 device-based EP API** introduced in ORT 1.23.0. The EP library is registered dynamically at runtime, then devices are enumerated and appended to the session.

### C/C++

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
std::vector<Ort::ConstEpDevice> devices = {trt_device};
session_options.AppendExecutionProvider_V2(env, devices, ep_options);

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

if trt_device is None:
    raise RuntimeError("TensorRT RTX EP device not found")

# 3. Add EP device to session options with provider options
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(
    [trt_device],
    {"enable_cuda_graph": "1", "nv_runtime_cache_path": "./cache"})

# 4. Create session and run inference
session = ort.InferenceSession("model.onnx", sess_options=session_options)
result = session.run([], {"input": input_data})

# 5. Cleanup: delete session before unregistering
del session
ort.unregister_execution_provider_library("NvTensorRTRTXExecutionProvider")
```
## Documentation

| Document | Description |
|----------|-------------|
| [Coding Guidelines](CODING-GUIDELINES.md) | Code style and conventions |

## Examples

See [examples/cxx/README.md](examples/cxx/README.md) for build instructions and the full list of C++ samples.

## Contributing

This project is not currently accepting external contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
