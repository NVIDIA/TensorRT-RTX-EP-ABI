<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python tests and samples

The `tests/python_tests/` directory contains a pytest suite that exercises the EP
through the public ONNX Runtime Python API.

## Setup

Set up a virtual environment from the repo root:

```powershell
python -m venv .venv-pytests
.\.venv-pytests\Scripts\Activate.ps1
pip install -r tests\python_tests\requirements.txt
pip install path\to\onnxruntime_ep_nv_tensorrt_rtx-*.whl
```

Use the `onnxruntime` version from the [release notes](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases) for your EP release.

## Run

Run pytest from inside `tests\python_tests` so test artifacts land next to the
tests rather than in the repo root:

```powershell
cd tests\python_tests
python -m pytest -v
```

## Coverage

| Test file | What it covers |
|-----------|----------------|
| `test_registration.py` | EP library registration, device discovery, unregister and re-register |
| `test_sessions.py` | Session creation with explicit EP devices and provider options |
| `test_ep_context.py` | EPContext creation and reload via `ep.context_*` session options |
| `test_compile_model.py` | Compile API file and bytes coverage (requires `onnxruntime.ModelCompiler`) |
| `test_proto_preprocessing.py` | Proto-preprocessing passes exercised indirectly through compile |

## Notes

- Tests require an NVIDIA RTX GPU to run.
- The compile tests (`test_compile_model.py`) require `onnxruntime` 1.27+ which includes `onnxruntime.ModelCompiler`. No special build needed — `pip install onnxruntime>=1.27` is sufficient.
- Test artifacts (ONNX models, EPContext files, cache directories) are written
  to the current working directory and are not auto-cleaned between runs.
