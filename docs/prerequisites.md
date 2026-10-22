<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime prerequisites

This page covers runtime prerequisites for running an application with TensorRT RTX Execution
Provider (EP), ensure the system has a supported GPU, a compatible NVIDIA driver, and the
runtime libraries supplied for the EP release.

## System requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX GPU, Ampere or newer (compute capability 8.0 or higher). Turing GPUs are not supported by the EP. |
| NVIDIA driver | Install the latest supported driver for your GPU and operating system for best performance. See the driver requirements below. |
| Microsoft Visual C++ runtime (Windows) | Install the latest supported Visual C++ v14 Redistributable matching the application architecture; see [Windows runtime dependencies](#windows-runtime-dependencies). |

End users **do not** need to install CUDA Toolkit or the TensorRT RTX SDK separately
when running a distribution that includes the required runtime libraries. The
NVIDIA GPU driver must be installed separately.

## Runtime libraries

Keep the EP and its accompanying TensorRT RTX runtime, parser, and plugin
libraries from the same release together. Do not replace individual libraries
with those from another package. The EP and TensorRT RTX have separate version
numbers; use the release's dependency information to identify the matching
runtime versions.

For application-managed deployment, follow the
[runtime dependency instructions](integration-guide.md#runtime-dependencies).
Python wheel deployments bundle their runtime dependencies. Additional system
runtime dependencies depend on the platform and distribution; follow the
installation instructions supplied with the release.

### Windows runtime dependencies

Windows deployments must provide the Microsoft Visual C++ runtime required by
accompanying libraries such as `tensorrt_plugins.dll`. Bundling TensorRT RTX
and CUDA runtime libraries does not remove this system prerequisite.

Install the latest supported [Microsoft Visual C++ v14 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
for the application's architecture ([x64 download](https://aka.ms/vc14/vc_redist.x64.exe)
or [ARM64 download](https://aka.ms/vc14/vc_redist.arm64.exe)). Microsoft requires
the installed runtime to be at least as recent as the MSVC tools used to produce
the application and its libraries. End users need the redistributable, not
Visual Studio or its build tools.

### Prebuilt packages and CUDA compatibility

Prebuilt EP packages bundle **TensorRT RTX for CUDA 13 only**. Their NVIDIA driver
must therefore support CUDA 13 and satisfy any additional requirements of the
bundled TensorRT RTX runtime. A driver that supports CUDA 12.9 but not CUDA 13
cannot run these prebuilt packages.

To deploy on a driver that supports CUDA 12.9, **build the EP package from
source** with CUDA 12.9 and the matching TensorRT RTX CUDA 12 distribution,
following the [build guide](build-guide.md). The resulting package must meet the
GPU and runtime driver requirements for those dependencies. Installing CUDA 12.9
or replacing a DLL in a prebuilt CUDA 13 package does not convert it into a
compatible CUDA 12 package.

## NVIDIA GPU driver

The NVIDIA GPU driver is distributed separately from the CUDA Toolkit for the latest release and onward.
don't confuse the toolkit version and the driver version.

### Recommended driver

For best performance, always use the **latest NVIDIA GPU driver (CUDA driver)
available for your GPU and operating system**. Keeping the driver up to date is
the recommended practice. The recommended version advances as new drivers are
released.

### Minimum compatibility requirements

The driver must satisfy the requirements of the GPU and the TensorRT RTX and
CUDA runtime shipped with the EP. Minimum compatibility requirements are lower
bounds, not the recommended driver version.

#### Prebuilt packages (CUDA 13)

| RTX EP version | Minimum driver version (Linux) | Minimum driver version (Windows) |
|----------------|--------------------------------|----------------------------------|
| 0.4 (cu13 package) | >=590.48.01 | >=580.88 |
| 0.5 (cu13 package) | >=590.48.01 | >=580.88 |

[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
allows CUDA 13.x applications to run on R580 or newer with feature limitations.
The bundled TensorRT RTX runtime, GPU, or required features may impose a higher
minimum. Consult the deployed release's requirements. Future EP releases may
change these minimums; the latest supported driver remains recommended.

#### Source-built packages (CUDA 12.9)

The following lower minimums apply only when building with CUDA 12.9 and a
matching TensorRT RTX CUDA 12 distribution. They do not apply to the CUDA 13
prebuilt packages above.

| RTX EP version | Minimum driver version (Linux) | Minimum driver version (Windows) |
|------------------|--------------|----------------------|
| 0.4 | >=575.57.08 | >=576.57 |
| 0.5 | >=575.57.08 | >=576.57 |

Check the [CUDA driver compatibility requirements](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver)
and the [TensorRT RTX release notes](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes.html)
for the runtime versions included in your EP distribution.

## Download and verify the driver

1. Open [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).
2. Select your GPU and operating system, then download the latest supported
   driver. For systems requiring OEM drivers, use the latest supported driver
   from the system manufacturer.
3. Install the driver, restart if prompted, and check the installed version:

   ```console
   nvidia-smi --query-gpu=name,driver_version --format=csv
   ```

The CUDA version displayed by `nvidia-smi` describes driver capability. It does
not identify the CUDA runtime bundled with the EP or require a toolkit
installation.

## Troubleshooting

- **Missing `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, or `MSVCP140.dll`:** install
  or repair the Visual C++ Redistributable for the application's architecture
  using the Microsoft links above.

- **Driver too old or CUDA initialization fails:** update the NVIDIA driver and
  verify that it supports the deployed runtime and GPU.
- **Runtime library cannot be loaded:** verify that the application has the
  matching libraries and system dependencies for its EP distribution. Avoid
  mixing releases in the application's library search path.
- **No compatible device is discovered:** confirm the GPU meets the EP's Ampere
  minimum and that the host can register the EP library.

When reporting an issue, include the EP and runtime versions, host version,
GPU model, operating system, driver version, and error log.
