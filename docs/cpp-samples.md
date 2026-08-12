<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# C++ samples

The C++ samples demonstrate ONNX Runtime's V2 device-based execution provider
APIs with the TensorRT RTX Execution Provider. They share the `candy.onnx`
neural-style-transfer model and `Input.png` test image unless noted otherwise.

## Available samples

`10_ep-device-selection` — Registers execution provider libraries, enumerates
available EP devices, and selects a device explicitly or through an ONNX Runtime
selection policy. The executable is `ep-device-selection`.

`20_devicetensors-datatransfer` — Allocates EP-agnostic device tensors,
transfers them with `CopyTensors`, and uses I/O binding to avoid repeated
transfers during inference. The executable is `devicetensors-datatransfer`.

`21_devicetensors-datatransfer-async` — Uses EP-provided pinned and
device-memory allocators, asynchronous `CopyTensors` operations, sync
notifications, and non-synchronizing run submissions. The executable is
`devicetensors-datatransfer-async`.

`30_syncstreams-cuda` — Demonstrates CUDA interoperability with ONNX Runtime
`SyncStream` and `SyncNotification` objects across separate upload and inference
streams. This target is built only when CMake finds the CUDA Toolkit. The
executable is `syncstreams_cuda`.

`40_ep-context` — Precompiles models into reusable TensorRT RTX EP Context
models. `sample` covers file-based embedded and external context modes;
`sample_buffer` covers in-memory models with external initializer data.

## Build and run

Enable the samples while configuring the main project:

```text
-DBUILD_EXAMPLES=ON
```

Then build normally as described in the [build guide](build-guide.md). CMake
downloads `candy.onnx` and copies the common input image and required runtime
libraries to the sample output directories.

The sample executables are created below `build/examples/cxx`. Visual Studio
and other multi-configuration generators add a configuration directory such as
`Release`. Each sample directory contains a README with its command-line
arguments and additional implementation details.

The samples run inference on an NVIDIA GPU. Their CTest registrations may
therefore fail on systems without a compatible GPU or runtime driver.
