# TensorRT RTX EP performance benchmark

`trt_ep_perf_test` compiles an ONNX model into an EP-context model, loads the context into an ONNX Runtime session, and
runs inference. It emits one tab-indented JSON result on success, either to standard output or to the path specified by
`--output`.

## Example

~~~powershell
trt_ep_perf_test `
  --model model.onnx `
  --context context.onnx `
  --runtime-cache runtime-cache/ `
  --time-compile 3 `
  --time-deserialize 5 `
  --warmup-duration 2 `
  --run-duration 10 `
  --output result.json
~~~

Use `--skip-compile` with an existing `--context` to omit compilation. In that case the JSON compile value is {}.
The benchmark creates one deterministic random input set before warmup and reuses it for every warmup and measured
inference. Floating-point values are sampled from [-1, 1); integer values span their full representable range. Input
generation is excluded from inference timing.
By default, the benchmark creates and binds a fresh Ort::IoBinding for each inference run, so ONNX Runtime performs
input/output binding work again which also triggers a copy. `--disable-io` restores the prior behavior of reusing one binding for all.
`--async` adds `disable_synchronize_execution_providers=1` to every `Ort::RunOptions`. The benchmark still calls 
`IoBinding::SynchronizeOutputs`, so each measured inference completes before its timer stops.

Use `--shape NAME=DIM` (repeat as needed) to set named ONNX dimensions. By default, these values are used only for the
generated inference inputs. Add `--overwrite-shape` to apply every `--shape` value through ONNX Runtime's free-dimension
override before compilation, so the compiled model uses the specified dimensions. Shapes identify ONNX symbolic
dimensions by name, not input tensors.

## Result schema

~~~json
{
  "status": "ok",
  "compile" : {
    "first_us" : 0,
    "min_us" : 0,
    "max_us" : 0,
    "median_us" : 0,
    "memory" : {"cpu_private_delta_bytes" : 0, "cpu_resident_delta_bytes" : 0, "gpu_process_delta_bytes" : 0}
  }, 
  "load" : {
    "first_us" : 0,
    "min_us" : 0,
    "max_us" : 0,
    "median_us" : 0,
    "memory" : {"cpu_private_delta_bytes" : 0, "cpu_resident_delta_bytes" : 0, "gpu_process_delta_bytes" : 0}
  }, 
  "inference": {
    "first_us" : 0, "min_us" : 0, "max_us" : 0, "median_us" : 0,
    "first_enqueue_us" : 0, "enqueue_min_us" : 0, "enqueue_max_us" : 0, "enqueue_median_us" : 0,
    "memory" : {"cpu_private_delta_bytes" : 0, "cpu_resident_delta_bytes" : 0, "gpu_process_delta_bytes" : 0}
  }
}
~~~

All durations are microseconds. status is "ok" for a successful run. On failure, the process writes {"status":
"error","message":"..."} to standard output and exits nonzero.

### Stage timing fields

compile and load use the same fields:

- `first_us`: duration of the first warmup repetition
- `min_us`, `max_us`, `median_us`: statistics over repetitions after warmup With one repetition, these fields equal
  first_us. For an even number of included repetitions, median_us is the arithmetic mean of the two central values.

`--time-compile` controls `Ort::CompileModel` repetitions and defaults to 1.
`--time-deserialize` controls `Ort::Session` repetitions using an EP context and defaults to 1.

For warmup and timed inference, provide either an iteration count or a duration in seconds:

- `--warmup-iterations COUNT` (default: 5) or `--warmup-duration SECONDS`
- `--run-iterations COUNT` (default: 50) or `--run-duration SECONDS`

The iteration and duration options for the same phase are mutually exclusive. Duration-based phases always execute at
least one inference; a run already in progress is allowed to complete after the requested duration expires.

Inference has one memory-monitoring interval covering the full warmup and timed-run sequence. It has additional latency
measurements for CPU enqueue:

- `first_enqueue_us`: duration of the first warmup `Session::Run` before output synchronization.
- `enqueue_min_us`, `enqueue_max_us`, `enqueue_median_us`:
  statistics across the same timed inferences through `Session::Run` only, before output synchronization.

### Memory fields

Each stage's memory object contains the maximum observed delta for that stage. With multiple compile or load
repetitions, the reported value is the largest delta seen in any repetition;
the values are not summed.

        - `cpu_private_delta_bytes`
    : process private memory increase.- `cpu_resident_delta_bytes`: process resident / working -
        set increase.- `gpu_process_delta_bytes`
    : per -
      process GPU memory increase.This is measured with DXGI local - video - memory usage on Windows and
    NVML on Linux
        .

    The monitor samples memory every millisecond.CPU memory is measured with PrivateUsage and
    WorkingSetSize on Windows
    , and / proc / self / status on Linux.On Windows
    , DXGI samples the current process's local-video-memory usage for the CUDA adapter;
on Linux,
    NVML samples this process on every visible
            GPU.The CUDA primary context is established before the monitor baseline is taken.Very short allocation
                spikes can be missed by the one -
        millisecond sampling interval.

        ##Measurement boundaries

            Provider registration,
    EP - device discovery, session options,
    and construction of the memory monitor happen before the corresponding stage timer starts.

        - Compile : includes directory creation and Ort::CompileModel.-
        Load : includes only construction of Ort::Session from the context model.-
        Inference
    : input tensor allocation and output -
      memory selection happen before the stage
          starts.Each inference measures Session::Run and IoBinding::SynchronizeOutputs.
