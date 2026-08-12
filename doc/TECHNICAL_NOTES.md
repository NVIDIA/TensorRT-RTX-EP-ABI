# NvTensorRTRTX EP Technical Notes

In-depth notes on selected features and behaviors of the NvTensorRTRTX execution provider.
Each section is self-contained; new topics are appended as they come up.

- [Synchronous GPU allocation (`nv_use_sync_gpu_allocator`)](#synchronous-gpu-allocation-nv_use_sync_gpu_allocator)
- [Weightless EPContext refit (`nv_weight_stripped_engine_enable_experimental`) — experimental](#weightless-epcontext-refit-nv_weight_stripped_engine_enable_experimental--experimental)

---

## Synchronous GPU allocation (`nv_use_sync_gpu_allocator`)

How to disable TensorRT RTX's asynchronous CUDA memory allocation (`cudaMallocAsync`) in the
NvTensorRTRTX execution provider, why you would want to, and what it costs.

| Mode | Provider option | Allocation path |
|---|---|---|
| **Async (default)** | option omitted, or `"nv_use_sync_gpu_allocator": "0"` | TensorRT RTX default allocator: `cudaMallocAsync` when CUDA memory pools are supported, `cudaMalloc` otherwise |
| **Sync (opt-in)** | `"nv_use_sync_gpu_allocator": "1"` | `GpuSyncAllocator` installed via `setGpuAllocator()`: all TensorRT allocations go through the EP's cudaMalloc-backed BFC arena |

### Background

TensorRT RTX's default GPU allocator services stream-ordered allocation requests with
`cudaMallocAsync` (see `IRuntime::setGpuAllocator` documentation). Async allocation reduces
allocation overhead and is a prerequisite for capturing runtime-allocated code paths into CUDA
graphs. However, it has proven unreliable in some RTX deployment environments:

- Under **CiG (CUDA in Graphics)** — D3D12/Vulkan interop scenarios — `cudaMallocAsync` can fail
  outright.
- On Windows, the async pool can exhaust the process **virtual address space while VRAM is still
  available** (a known CUDA driver bug), producing out-of-memory failures that depend on
  process history rather than actual memory pressure.
- The resulting behavior differs across configurations (dedicated GPU vs. shared/iGPU, app VA
  layout), making failures nondeterministic and hard to reproduce.

For applications that hit these issues, the EP provides an opt-in fully synchronous allocation
path. The option defaults to **off** so existing (async) behavior is unchanged.

### What enabling it does

When `nv_use_sync_gpu_allocator=1`, the EP:

1. Wraps the factory-owned per-device **BFC arena** (cudaMalloc/cudaFree backed, pooling) in a
   `GpuSyncAllocator` (`src/gpu_sync_allocator.h`) and installs it on the TensorRT RTX **runtime**
   at EP creation and on the **builder** at first use. Every allocation TensorRT makes — engine
   deserialization, build-time scratch, and Myelin's runtime tensor allocations — then goes
   through the arena. Because the arena pools memory, repeated allocations amortize to zero CUDA
   API calls after warmup.
2. Routes the EP's own **execution context memory** (the `setDeviceMemoryV2` scratch buffer)
   through the same synchronous arena instead of the async CUDA mempool
   (`cudaMallocFromPoolAsync`).

No allocation path can reach `cudaMallocAsync` afterwards. On success the session log shows:

```
[NvTensorRTRTX EP] Using synchronous GPU allocator (GpuSyncAllocator); TensorRT RTX async allocation (cudaMallocAsync) is disabled.
```

If the device arena cannot be created, the EP logs a warning and falls back to the default
(async) allocator rather than failing the session:

```
[NvTensorRTRTX EP] nv_use_sync_gpu_allocator was requested but the device arena could not be created; falling back to the default allocator.
```

### Enabling / disabling

The switch is a per-session provider option (string `"0"`/`"1"`). There is no environment
variable; async mode is restored by omitting the option or passing `"0"`.

#### Python

```python
import onnxruntime as ort

ort.register_execution_provider_library("nv_tensorrt_rtx", ep_library_path)
devices = [d for d in ort.get_ep_devices() if d.ep_name == "nv_tensorrt_rtx"]

so = ort.SessionOptions()
# Disable async CUDA malloc (force synchronous allocation):
so.add_provider_for_devices(devices, {"nv_use_sync_gpu_allocator": "1"})
# Async mode (default): omit the key, or pass {"nv_use_sync_gpu_allocator": "0"}.

session = ort.InferenceSession("model.onnx", sess_options=so)
```

#### C++

```cpp
Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
env.RegisterExecutionProviderLibrary("nv_tensorrt_rtx", ep_library_path);

std::vector<Ort::ConstEpDevice> selected;   // filter env.GetEpDevices() by EpName()

Ort::KeyValuePairs ep_options;
ep_options.Add("nv_use_sync_gpu_allocator", "1");   // sync mode; omit for async (default)

Ort::SessionOptions so;
so.AppendExecutionProvider_V2(env, selected, ep_options);
Ort::Session session(env, model_path, so);
```

See `examples/cxx/10_ep-device-selection` for the full device-selection pattern.

### Performance guidance

- **Static-shape models** (and dynamic-shape models run with ORT free-dimension overrides): no
  measurable difference. TensorRT plans activation memory statically into the EP-managed scratch
  buffer, so there are no per-inference allocations for the mode to affect. In a ~1.5K-model
  sweep only 2 models regressed with sync allocation.
- **Dynamic-shape models**: models whose optimization profile spans a wide range can have
  tensors that Myelin allocates at *inference time* (runtime tensor allocation) instead of
  planning statically. Two effects apply under sync mode:
  1. Those per-inference allocations become synchronous. The arena pooling recovers most of the
     overhead after warmup, but the first allocation of each new high-water mark synchronizes the
     device.
  2. TensorRT RTX disables **CUDA graph capture** for runtime-allocated paths when the allocator
     is not asynchronous. Enqueue-bound models lose the graph-capture benefit (observed up to 4x
     latency on an extreme dynamic-range vision model).

  Mitigation: constrain the shape range. Use ORT **free-dimension overrides** to make symbolic
  dims static, or provide realistic ranges via `nv_profile_min_shapes` / `nv_profile_opt_shapes` /
  `nv_profile_max_shapes`. Without either, the EP builds dynamic inputs with an implicit profile
  of min 0 / opt 1 / max 32767 per dynamic dimension, which maximizes runtime allocation.

### When to enable

Enable `nv_use_sync_gpu_allocator` when:

- the application runs TensorRT RTX under **CiG** (D3D12/Vulkan graphics interop),
- you observe `cudaMallocAsync`/async-pool OOM failures while VRAM is available (VA-space
  exhaustion), or
- you require **deterministic, reproducible allocation behavior** across GPU configurations.

Keep the default (async) when none of the above apply, especially for dynamic-shape models that
benefit from CUDA graph capture.

### References

- `src/gpu_sync_allocator.h` — allocator implementation
- `tests/test_tensorrt_rtx_options.cpp` — option unit test

---

## Weightless EPContext refit (`nv_weight_stripped_engine_enable_experimental`) — experimental

> **⚠️ EXPERIMENTAL — opt-in, not enabled by default.** The behavior, the provider-option name, and
> the on-disk refit-table format may change or be removed in a future release. Enabling the option
> emits a warning at session creation. Test coverage is limited; validate on your own models before
> relying on it.

Ship a small **weight-stripped** TensorRT-RTX engine plus a compact refit recipe instead of a full
engine with baked-in weights. At load, the EP **refits** the weights back into the engine from the
model's own initializers — the original ONNX model is not needed at load. This shrinks the shipped
artifact, deduplicates weight storage, accelerates load, and supports models above the 2 GB protobuf
limit. The mechanism is IHV-agnostic (it uses generic ONNX Runtime hooks); the refit itself is done
with TensorRT-RTX's refitter.

| Mode | Provider option | Behavior |
|---|---|---|
| **Default (off)** | option omitted, or `"nv_weight_stripped_engine_enable_experimental": "0"` | Normal EPContext: a full engine with baked-in weights; nothing extra is needed at load. |
| **Weightless (opt-in)** | `"nv_weight_stripped_engine_enable_experimental": "1"` | Compile a weight-stripped engine + `ep_refit_table`; refit from the kept initializers at load. |

### Requirements

- **TensorRT-RTX 1.6 or newer — required.** The weightless capture/replay API and the weight-strip
  build capability are only available from 1.6. Building or running this feature against an older
  TensorRT-RTX is not supported.
- **ONNX Runtime 1.27 or newer** — the feature relies on ORT's external-initializer-location callback
  and EPContext model-editor APIs.
- **GPU-architecture floor.** Weight-stripping (`kSTRIP_PLAN`) is a build-time capability of the
  TensorRT-RTX builder (Myelin); whether it can be honored depends on the target GPU architecture
  together with the installed TensorRT-RTX version. Some architectures need a specific patch build —
  for example **SM120 / RTX 5090 requires TensorRT-RTX >= 1.6.1.106**. This floor is resolved when the
  engine is actually built, so an unsupported combination fails at build time (see *Failure behavior*).

### How it works

- **Compile (capture).** With the option on, the EP builds the engine with the `kSTRIP_PLAN` and
  `kREFIT_IDENTICAL` builder flags (a weight-stripped, refittable engine) and records a compact
  **`ep_refit_table`**: for each engine weight, which ONNX initializer(s) it comes from and any
  transform applied (identity, dtype cast, BatchNorm fold, constant). The refit-source initializers
  are preserved as inputs on the exported EPContext node, so they travel with the model and can be
  resolved by name at load.
- **Load (refit).** The EP reads the `ep_refit_table`, resolves each source weight from the model's
  kept initializers, and refits them into the deserialized engine via `IRefitter::setNamedWeights`
  followed by `refitCudaEngine()`. No original ONNX model is required. The weightless path is selected
  by the **presence of the `ep_refit_table` marker** on the node; a model without it loads normally.
  Weights are fully refit before execution, so **inference time is unchanged** — only load does extra
  work.

### Using it

Set the option on the EP at **both compile and load**:

```cpp
ep_options.Add("nv_weight_stripped_engine_enable_experimental", "1");
```

For large models, and to avoid duplicating weights, compile from the on-disk model and register a
per-initializer external-data location callback so the weights stay in the external file and the
emitted `.onnx` stays tiny:

- `ModelCompilationOptions::SetInputModelPath(...)` — so originally-external initializers arrive with
  their file, offset, and size populated.
- `SetOutputModelGetInitializerLocationFunc(...)` — a callback that, for an already-external
  initializer, re-references the **same** file at the **same** offset (zero copy); other initializers
  fall back to a sidecar file (above a size threshold) or inline (below it).

Worked examples: `examples/cxx/40_ep-context/sample_weightless.cpp` (compile/reload from a file path)
and `sample_weightless_buffer.cpp` (compile/reload from an in-memory buffer, with the weights file in a
different folder). Both accept `--verify` to compare outputs numerically against a weight-full
reference run.

On enable, the session log shows (level WARNING):

```
[NvTensorRTRTX EP] nv_weight_stripped_engine_enable_experimental is an EXPERIMENTAL feature (weightless EPContext refit / weight-stripped engine). It is opt-in, not enabled by default, and may change or be removed in a future release.
```

### Modes and weight location

`embed_mode` and where the **weights** live are independent axes:

- **`embed_mode` (0 or 1)** controls only where the compiled **engine** bytes go — embedded inside the
  `.onnx` (`1`) or written to a separate engine cache file (`0`).
- **Weight storage** is controlled by the initializer-location callback: inline in the model, a sidecar
  file, or a zero-copy reference to the original weights file.

> **Important — load-time weight dependency.** Because a weightless engine is refit at load, the
> **refit-source weights must be resolvable at load** (from inline data, an external file, or an
> in-memory buffer). This is a *new* load-time dependency that a normal full engine does not have. In
> particular, `embed_mode=1` with externalized weights is **not** a single self-contained file — the
> weights file (or buffer) must accompany the `.onnx`. Keeping weights inline yields a self-contained
> model but reintroduces the 2 GB `ModelProto` limit for large models.

### Failure behavior (no automatic fallback)

Because the feature is opt-in, failures surface as **clear errors** rather than silently degrading:

- **Weight-strip build fails** (unsupported architecture/SDK — e.g. SM120 below 1.6.1.106): the compile
  fails with `ORT_EP_FAIL` and a message naming the SDK floor. There is **no** automatic retry as a
  non-stripped engine.
- **A refit source cannot be resolved at load** (e.g. the external weights file is missing): the load
  fails with `ORT_EP_FAIL` ("source weight … was not supplied"). The weights are required in order to
  refit; there is no fallback.
- **Refit-table capture fails** (rare — the strip build succeeded but the table could not be produced):
  the EP logs a warning and emits a stripped engine **without** a refit table. Such a model then
  requires the **original ONNX** at load (the legacy original-ONNX refit path) and is not
  self-contained.
- **The option must be set at both compile and load.** If it is omitted at load, the kept refit-source
  initializers are pruned during partitioning and the refit fails with "source weight … not supplied".

### Known limitations (current version)

- **Opt-in only, not default** — the option must be set at compile *and* load.
- **TensorRT-RTX 1.6+ required**; older SDKs are not supported.
- **Not a single self-contained file** when weights are externalized — the weights file/buffer is
  needed at load.
- **No fallback** — an unsupported arch/SDK, a failed capture, or a missing weight is a hard, clear
  error, by design for an opt-in experimental path.
- **BatchNorm fold supports float32 sources only** — fp16/double BatchNorm sources are rejected with a
  clear error.
- **Quantized / MatMulNBits (INT4) weights are not refit-captured.** The refit table only records the
  weights the TensorRT-RTX parser reports as standard refittable weights; it has no transform for a
  `MatMulNBits`-style packed set (`{INT4 weight + scales + zero-points}`). When such an op is executed by
  a TensorRT plugin, its weights live in the plugin's own serialized state and are invisible to the
  capture — they stay baked into the "stripped" engine (so weightless yields no size/dedup benefit for
  them, and there is a risk of incorrect results if the SDK strips them without the capture re-reporting
  them). When the op is lowered to `DequantizeLinear` + `MatMul`, the scale/zero-point initializers may be
  pruned during partitioning, in which case the export aborts with a clear "refit source … was not
  preserved" error. Net: **INT4-quantized LLMs (e.g. `MatMulNBits` produced by WebNN/Olive) are not
  reliably supported by weightless in this version.**
- **Limited model coverage.** Validated bit-exact on selected fp16 models (including a model above
  2 GB). Other classes may fail — see the INT4/`MatMulNBits` note above, and note computer-vision models
  are still under investigation. Validate on your target models.

### Roadmap

The intent is to graduate weightless to the default engine format once it is robust and broadly
validated — mirroring how TensorRT introduces experimental engine-builder flags before promoting them
to defaults. Planned follow-ups include: enforcing the TensorRT-RTX 1.6 minimum at build configuration,
making loading fully format-driven (no option required at load), adding a build-time fallback if it
becomes the default, and adopting an ORT-side initializer-prune-suppression change to simplify the
compile path.

### References

- `examples/cxx/40_ep-context/sample_weightless.cpp`, `sample_weightless_buffer.cpp` — worked examples
  (file and buffer input; `--verify` for a numerical check).
- `tests/test_tensorrt_rtx_weightless_refit.cpp` — refit-table format unit tests and an end-to-end
  strip → reload → run test.
- `src/weightless_refit.h`, `src/weightless_refit.cc` — refit-table format (serialize/deserialize) and
  source-name collection.
- `src/tensorrt_rtx_execution_provider.cc` — capture (`CaptureWeightlessRefitTable`) and replay
  (`WeightlessRefitEngineImpl`).
- `src/onnx_ctx_model_helper.cc` — EPContext export and the load-time refit (`TryWeightlessRefit`).
