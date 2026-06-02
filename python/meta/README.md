# onnxruntime-ep-nv-tensorrt-rtx

Meta-package for the NVIDIA TensorRT RTX Execution Provider plugin for
[ONNX Runtime](https://onnxruntime.ai/).

Installing this package pulls in the default CUDA variant
(`onnxruntime-ep-nv-tensorrt-rtx-cu13`). For a different CUDA version, install
the corresponding variant directly:

```bash
pip install onnxruntime-ep-nv-tensorrt-rtx-cu13   # CUDA 13 (default)
pip install onnxruntime-ep-nv-tensorrt-rtx-cu12   # CUDA 12
```

This follows the same pattern used by [TensorRT-RTX](https://pypi.org/project/tensorrt-rtx/).

---

## About NVIDIA TensorRT for RTX

NVIDIA® TensorRT™ for RTX (TensorRT-RTX) is an inference optimization library
dedicated for deploying AI inference on NVIDIA GeForce RTX GPUs. It is a great
choice for developers building applications that must run on Windows or Linux
PCs, laptops, or workstations.

The underlying packages bundle the TensorRT-RTX runtime libraries alongside the
ONNX Runtime EP plugin so that no separate TensorRT-RTX installation is required.

For more information about TensorRT-RTX, visit https://developer.nvidia.com/tensorrt-rtx.  
Online documentation: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html  
License agreement: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html

---

## References

- Release Notes: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes.html
- Support Matrix: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html
- Installation Guide: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installation-overview.html
