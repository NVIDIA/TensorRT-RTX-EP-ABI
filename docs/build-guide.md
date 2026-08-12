<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Build guide

This guide covers everything needed to build the TensorRT RTX Execution Provider
from source on Windows, Windows ARM64, and Linux, including the Python wheel.

> **Note:** The repository also provides convenience scripts `build.bat` (Windows)
> and `build.sh` (Linux) that wrap the CMake steps below and add path validation,
> vcpkg bootstrapping, and wheel packaging. The plain CMake commands documented
> here are sufficient for all standard builds.

## Prerequisites

### Software requirements

| Dependency | Minimum version | Platform | Notes |
|------------|-----------------|----------|-------|
| CMake | 3.20 | All | Must be on system `PATH` |
| Visual Studio | 2022 (17.0+) | Windows | "Desktop development with C++" workload required. VS 2026 also supported. |
| GCC / Clang | GCC 11+ / Clang 14+ | Linux | |
| Python 3 | 3.8+ | All | Required during CMake configuration for bundled ONNX dependency |
| CUDA Toolkit | 12.9 (cu12) or 13.x (cu13) | All | See [CUDA versions](#cuda-versions) below. Download from the [CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads). |
| ONNX Runtime | 1.24.0 | All | Prebuilt SDKs at [ORT releases](https://github.com/microsoft/onnxruntime/releases). |
| TensorRT RTX | 1.4.0 | All | Download from [TensorRT RTX](https://developer.nvidia.com/tensorrt-rtx). |

### CUDA versions

Two CUDA variants are supported:

- cu12 — CUDA 12.9 (pinned)
- cu13 — any CUDA 13.x release (Windows ARM64 requires 13.4+)

CUDA and TensorRT RTX must target the same CUDA major version.

> **Note:** If you already have `protobuf` installed via `winget` or another system
> package manager, it will conflict with CMake's FetchContent and fail
> configuration. Remove it or use `-DUSE_VCPKG=ON`.

## Build: Windows

Open a Developer PowerShell for VS 2022 (or VS 2026) so that MSVC and the
Windows SDK are on the path, then:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" `
  -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-x64-<version>" `
  -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-<version>"

cmake --build build --config Release --parallel
```

## Build: Linux

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime-<version> \
  -DTRT_RTX_ROOT=/opt/TensorRT-RTX-<version>

cmake --build build --parallel
```

## Build: Windows ARM64

Windows ARM64 (WoA) cross-compilation runs on an x64 Windows host. Open a
Developer PowerShell for VS 2022 and pass `-A ARM64`:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A ARM64 `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" `
  -DONNXRUNTIME_ROOT="C:\SDK\onnxruntime-win-arm64-<version>" `
  -DTRT_RTX_ROOT="C:\SDK\TensorRT-RTX-<version>-arm64" `
  -DUSE_PRECOMPILED_HOST_PROTOC=ON

cmake --build build --config Release --parallel
```

> **Note:** Windows ARM64 requires CUDA 13.4+. The cross-compiled output runs on ARM64
> devices but the build itself executes on an x64 Windows host.

## Python wheel

Building the wheel is a three-step process:

**Step 1** — Build the C++ library (as documented above)

**Step 2** — Stage runtime DLLs into the Python package directory (Windows):

```powershell
cd python
python scripts/stage_windows_dlls.py `
  --ep-dll "..\build\Release\onnxruntime_providers_nv_tensorrt_rtx.dll" `
  --trt-lib-dir "C:\SDK\TensorRT-RTX-<version>\bin" `
  --cuda-bin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\x64" `
  --trt-doc-dir "C:\SDK\TensorRT-RTX-<version>\doc"
```

For Linux use `python scripts/stage_linux_so.py` with equivalent arguments.

**Step 3** — Build the wheel:

Windows:

```powershell
$env:TRT_RTX_EP_VERSION = "<M.m.p>"
python -m build --wheel --no-isolation --outdir dist
```

Linux:

```bash
TRT_RTX_EP_VERSION="<M.m.p>" python -m build --wheel --no-isolation --outdir dist
```

**Outputs** (under `python\dist\` / `python/dist/`):

- `onnxruntime_ep_nv_tensorrt_rtx_cu12-<M.m.p>-py3-none-win_amd64.whl` — CUDA 12 variant
- `onnxruntime_ep_nv_tensorrt_rtx-<M.m.p>-py3-none-any.whl` — meta wheel

The meta wheel (`onnxruntime-ep-nv-tensorrt-rtx`) defaults to the cu13 variant.
CUDA 12 users must install `onnxruntime-ep-nv-tensorrt-rtx-cu12` explicitly.

> **Note:** `build.bat --build_wheel` / `build.sh --build_wheel` automate all three
> steps above in a single command.

## CMake options reference

| Option | Description |
|--------|-------------|
| `CUDAToolkit_ROOT` | CUDA Toolkit installation root |
| `ONNXRUNTIME_ROOT` | Root of an extracted ONNX Runtime. If omitted, CMake downloads it. |
| `ONNXRUNTIME_VERSION` | ORT version to download when `ONNXRUNTIME_ROOT` is not set |
| `TRT_RTX_ROOT` | Root of an extracted TensorRT RTX |
| `BUILD_TESTS` | Build the unit test suite (default: `ON`) |
| `BUILD_EXAMPLES` | Build the C++ examples (default: `OFF`) |
| `TRT_RTX_EP_PRODUCTION_BUILD` | Enable production-build checks and signature verification |
| `TRT_RTX_EP_VERSION` | Version string embedded in production artifacts |

## Run tests

Most runtime tests require an NVIDIA GPU. Building the tests does not.

**Windows:**

```powershell
.\build\tests\Release\unittests.exe
```

**Linux:**

```bash
./build/tests/unittests
```

## Troubleshooting

### CMake cannot find Python

Set `Python3_EXECUTABLE` explicitly:

```powershell
cmake ... -DPython3_EXECUTABLE="C:\Python311\python.exe" ...
```

### `TRT_RTX_ROOT must be set`

Same as above for TensorRT RTX:

```powershell
Test-Path "C:\SDK\TensorRT-RTX-<version>"
```

### `Could not find CUDAToolkit`

Provide the path explicitly and confirm CUDA is installed:

```powershell
nvcc --version
cmake ... -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" ...
```

### CUDA and TRT RTX major version mismatch

CUDA and TensorRT RTX must target the same CUDA major version — both cu12 or both
cu13. Mixing versions causes linker errors.

### protobuf conflict

If protobuf is installed system-wide (e.g. via `winget install protobuf`), it
conflicts with CMake's FetchContent. Uninstall it before configuring.

### Stale build cache

After changing SDK roots, generator, or CUDA version, configure a fresh build
directory:

```powershell
# Windows
Remove-Item -Recurse -Force build
```

```bash
# Linux
rm -rf build
```

### C++20 compiler errors

Ensure Visual Studio 2022 17.0+ is installed (`_MSC_VER >= 1930`).
On Linux, GCC 11+ or Clang 14+ are required.

See the [integration guide](integration-guide.md) for provider integration and runtime usage.
