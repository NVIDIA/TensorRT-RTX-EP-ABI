# Python TRT RTX EP ABI Tests

This directory ports the gtest coverage under `tests/` to pytest where the
behavior is reachable through the public ONNX Runtime Python API.

The pytest fixture registers the EP library from the ABI wheel:

```python
import onnxruntime_ep_nv_tensorrt_rtx as trt_rtx_ep

ort.register_execution_provider_library(
    trt_rtx_ep.get_ep_name(),
    trt_rtx_ep.get_library_path(),
)
```

## Running

Set up the venv from the repo root:

```powershell
python -m venv .venv-pytests
.\.venv-pytests\Scripts\Activate.ps1
python -m pip install -r tests\python_tests\requirements.txt
python -m pip install path\to\onnxruntime_ep_nv_tensorrt_rtx-*.whl
```

Then run pytest from inside `tests\python_tests` so the test artifacts land
next to the tests rather than in the repo root:

```powershell
cd tests\python_tests
python -m pytest -v
```

Use the ONNX Runtime package that matches the ABI wheel. The compile tests
require a Python build with `onnxruntime.ModelCompiler`.

Generated ONNX models, EPContext models, cache directories, and profiling
files are written into the current working directory (mirroring the C++ tests,
which write next to the model file). Running pytest from `tests\python_tests`
keeps every artifact under that directory and out of the repo root. Files are
not auto-cleaned between runs; each test calls `clear_path` on its own outputs
before producing them.

## Coverage Notes

Direct ports:

- EP library registration, device discovery, unregister and re-register.
- Session creation with explicit EP devices and provider options.
- EPContext creation and reload via `ep.context_*` session options.
- Parameterized input/output dtype smoke tests where Python can feed the dtype.
- Runtime cache path behavior.
- Output metadata behavior for multi-output and unused-node-output models.
- EPContext source attribute behavior.
- Split-graph asymmetric Q/DQ and DQ runtime checks with CPU golden output.
- Compile API file and bytes coverage when Python `ModelCompiler` is present.
- Indirect proto-preprocessing smoke tests through compile-to-EPContext.

Not direct ports:

- `GetSharedAllocator`, `CopyTensors`, raw `OrtEpDevice_MemoryInfo`, and CUDA
  stream/mempool internals are not exposed at the same level through Python.
- The C++ proto-preprocessing tests call internal source functions directly.
  Python can only exercise those passes indirectly through session or compile.
- BF16 Python feed support depends on the installed NumPy/ORT build, so the
  BF16 dtype case is skipped unless a normal Python feed path is available.
- Non-embedded compiled-model bytes output is skipped because Python
  `ModelCompiler` does not expose the C++ `SetEpContextBinaryInformation`
  hook needed to resolve the external engine binary.
- The three special-character path tests in `test_compile_model.py`
  (`test_context_input_path_special_chars`,
  `test_context_output_path_special_chars`,
  `test_runtime_cache_path_special_chars`) are marked `@pytest.mark.skip`.
  The EP receives provider-option paths as `std::string` and constructs
  `std::filesystem::path` from them, which on Windows decodes through the
  system ANSI codepage rather than UTF-8. Python ORT bindings send UTF-8,
  so non-ASCII path characters (e.g. `é`) get mis-decoded and the EP fails
  with "invalid argument". The C++ counterparts only pass on machines
  configured with the UTF-8 ACP toggle. Remove the skips once the EP
  accepts UTF-8 explicitly.
