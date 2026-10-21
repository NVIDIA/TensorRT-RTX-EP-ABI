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

The `resnet18_Opset18_timm.onnx` model is required. Place it in the directory
where pytest runs, in `tests\python_tests`, or in `tests`. The suite fails when
the model is missing so model-dependent coverage cannot be silently skipped.

Then run pytest from inside the tests/python_tests directory so local model
discovery and the suite configuration are predictable:

```powershell
cd tests\python_tests
python -m pytest -v
```

EP-backed tests require the public TensorRT RTX EP wheel, a visible NVIDIA EP
device, and the ORT device-registration APIs. Missing requirements fail those
tests instead of skipping them. The shared session helper disables CPU EP
fallback by default, so a successful session cannot silently pass on CPU.
Intentional split-graph tests must explicitly opt in to CPU fallback and assert
both CPU and `nv_tensorrt_rtx` assignments through the ORT profile.

Use the ONNX Runtime package that matches the ABI wheel. The compile tests
require a Python build with `onnxruntime.ModelCompiler`.

Generated ONNX models, EPContext models, cache directories, and profiling
files use pytest's per-test tmp_path directories. Pytest removes older
temporary-session roots according to its retention policy; it may retain recent
runs for debugging. Tests also call clear_path before reusing named outputs.

## Coverage Notes

Direct ports:

- EP library registration, device discovery, unregister and re-register.
- Session creation with explicit EP devices and provider options.
- EPContext creation and reload via `ep.context_*` session options.
- Parameterized input/output dtype tests with non-zero analytic output checks.
- Runtime cache path behavior.
- Runtime-cache recovery, unsafe partition names, and teardown write failures.
- Output metadata behavior for multi-output and unused-node-output models.
- EPContext source attribute behavior.
- Split-graph asymmetric Q/DQ and DQ runtime checks with CPU golden output.
- Compile API file and bytes coverage when Python `ModelCompiler` is present.
- Concurrent compiled-session creation, stream output, and in-memory external
  initializers.
- Profiler output naming, JSON structure, inference grouping, CUDA graph
  composition, and output-error handling.
- Weight-streaming budget parsing, output equivalence, EPContext reload, and
  runtime-cache behavior.
- Experimental weightless-engine refit and compile-only EPContext reload.
- User-facing provider option aliases, synchronous allocator mode, high-rank
  rejection, and automatic EP selection.
- Indirect proto-preprocessing tests with compile-to-EPContext and CPU-golden output checks.

Not direct ports:

- `GetSharedAllocator`, `CopyTensors`, raw `OrtEpDevice_MemoryInfo`, and CUDA
  stream/mempool internals are not exposed at the same level through Python.
- The C++ proto-preprocessing tests call internal source functions directly.
  Python can only exercise those passes indirectly through session or compile.
- The C++ large subgraph-dump test injects provider-internal _MEM_ADDR_ tensors
  to exercise external-data materialization. Public Python can only supply normal
  ONNX initializers, so that internal-memory conversion remains C++-only.
- BF16 Python feed support depends on the installed NumPy/ORT build, so the
  BF16 dtype case is skipped unless a normal Python feed path is available.
- Non-embedded compiled-model bytes output is skipped because Python
  `ModelCompiler` does not expose the C++ `SetEpContextBinaryInformation`
  hook needed to resolve the external engine binary.
- Special-character runtime-cache paths run on Windows. The context model for
  that test uses an ASCII path so the test isolates the EP's explicit UTF-8
  provider-option-to-native-path conversion.
- Special-character input and output model paths remain skipped on Windows
  because ONNX Runtime's Python `ModelCompiler` does not preserve those paths
  before the EP is invoked.
