# onnxruntime-ep-nv-tensorrt-rtx

NVIDIA TensorRT RTX Execution Provider plugin for [ONNX Runtime](https://onnxruntime.ai/).

Enables hardware-accelerated inference on NVIDIA RTX GPUs (Ampere / RTX 30xx and later) via the
[ORT Plugin EP ABI](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html).

---

## About NVIDIA TensorRT for RTX

NVIDIA® TensorRT™ for RTX (TensorRT-RTX) is an inference optimization library dedicated for
deploying AI inference on NVIDIA GeForce RTX GPUs. It is a great choice for developers building
applications that must run on Windows or Linux PCs, laptops, or workstations.

This package bundles the TensorRT-RTX runtime libraries alongside the ONNX Runtime EP plugin so
that no separate TensorRT-RTX installation is required.

For more information about TensorRT-RTX, visit https://developer.nvidia.com/tensorrt-rtx.  
Online documentation: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html  
License agreement: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html

---

## References

- Release Notes: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes.html
- Support Matrix: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html
- Installation Guide: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installation-overview.html
- C++ API: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/cpp-api/index.html
- Python API: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/python-api/index.html

---

## Requirements

- NVIDIA RTX GPU (Ampere or later)
- NVIDIA GPU driver with CUDA {CUDA_MAJOR} support
- `pip install onnxruntime>=1.24`

## Installation

```bash
pip install onnxruntime>=1.24
pip install onnxruntime-ep-nv-tensorrt-rtx
```

## Usage

```python
import onnxruntime as ort
import onnxruntime_ep_nv_tensorrt_rtx as trt_ep

# Register the EP plugin
ort.register_execution_provider_library(trt_ep.get_ep_name(), trt_ep.get_library_path())

# List available devices
devices = [d for d in ort.get_ep_devices() if d.ep_name == trt_ep.get_ep_name()]
print(f"TensorRT RTX devices: {len(devices)}")

# Create session with EP
so = ort.SessionOptions()
so.add_provider_for_devices(devices, {})
sess = ort.InferenceSession("model.onnx", sess_options=so)
```

## License

Apache 2.0. See [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
