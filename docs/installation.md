<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Installation

The easiest way to use the TensorRT RTX EP is via the Python wheel, which bundles
the EP plugin and all TensorRT RTX runtime libraries — no separate TensorRT RTX
installation required.

To build the EP from source instead, see the [build guide](build-guide.md).

## Requirements

- NVIDIA RTX GPU (Ampere / RTX 30xx or later)
- NVIDIA GPU driver compatible with your installed CUDA version
- Python 3.8+
- `pip install "onnxruntime>=1.24"`

## Install

> **Note:** If you have `onnxruntime-gpu` installed, uninstall it first — it ships a different
> version of the `onnxruntime` module:
> ```bash
> pip uninstall onnxruntime-gpu
> ```

CUDA 13 (default):

```bash
pip install "onnxruntime>=1.24"
pip install onnxruntime-ep-nv-tensorrt-rtx
```

CUDA 12:

```bash
pip install "onnxruntime>=1.24"
pip install onnxruntime-ep-nv-tensorrt-rtx-cu12
```

The default meta wheel (`onnxruntime-ep-nv-tensorrt-rtx`) installs the cu13 variant.
CUDA 12 users must install the `cu12` variant explicitly.

## Quick start

```python
import onnxruntime as ort
import onnxruntime_ep_nv_tensorrt_rtx as trt_ep

# Register the EP plugin
ort.register_execution_provider_library(trt_ep.get_ep_name(), trt_ep.get_library_path())

# Discover available TensorRT RTX devices
devices = [d for d in ort.get_ep_devices() if d.ep_name == trt_ep.get_ep_name()]
if not devices:
    raise RuntimeError("No TensorRT RTX EP devices found")

# Create a session with the EP
so = ort.SessionOptions()
so.add_provider_for_devices(devices, {})
sess = ort.InferenceSession("model.onnx", sess_opts=so)
```

The `onnxruntime_ep_nv_tensorrt_rtx` package exposes three helper functions:

| Function | Returns |
|----------|---------|
| `get_ep_name()` | `"nv_tensorrt_rtx"` — the EP provider name used for registration and device filtering |
| `get_library_path()` | Absolute path to the bundled EP plugin DLL / SO |
| `get_ep_names()` | `["nv_tensorrt_rtx"]` — list form of the above |

For the full Python and C++ usage patterns including provider options, see the
[integration guide](integration-guide.md).
