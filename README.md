# NVIDIA TensorRT RTX Execution Provider

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
![Platform](https://img.shields.io/badge/platform-**Windows**%20%7C%20Linux%20%7C%20ARM64-lightgrey)

The NVIDIA TensorRT RTX Execution Provider (EP) is an inference deployment solution designed
specifically for NVIDIA RTX GPUs, optimized for client-centric use cases. It is built as a
standalone plugin that implements the ORT Plugin EP ABI and leverages
[TensorRT for RTX](https://developer.nvidia.com/tensorrt-rtx) to accelerate ONNX models on
RTX GPUs (Ampere and later).

## Test Matrix

| EP Version | Base ORT Version | Compatible ORT Versions | TRT RTX Version | Notes |
|------------|-------------------|-----------------------------------|------------------|-------|
| 0.1 | 1.24.0 | 1.24.0 only | 1.4.x.x | Initial **Windows** Support |
| 0.3 | 1.25.2 | 1.24.4 – 1.25.2 | 1.5.x.x | Linux Support, Weight Streaming, CIG Interop, ORT Version Negotiation |
| 0.4 | 1.27.1 | 1.25.0 – 1.27.1 | 1.6.x.x | Sync GPU Allocator, iGPU Zero-Copy, Vulkan/CiG Import, Compile-Only Mode, Weightless EP Context, MSFT Profiler Support, Windows-on-Arm Build Support |
| 0.5 | 1.30.0 | 1.25.0 – 1.30.0 | 1.7.x.x | Runtime Weight-Streaming Budget Updates, CUDA Graph Capture Stability Fixes, Sandbox-Safe CPU-Only AOT Compile, Opt-in Persistent Execution Context Memory, Deferred/Streamed Weight Loading, WebNN Sandboxed-Browser Crash Fix, Discrete-GPU Zero-Copy Fix |

## Documentation

Full documentation — installation, build-from-source, integration guide, API reference, and
samples — is published at:

- **https://nvidia.github.io/TensorRT-RTX-EP-ABI/**

## Contributing

This project is not currently accepting external contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
