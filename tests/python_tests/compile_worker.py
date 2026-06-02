# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegisteredEp:
    ort: object
    ep_name: str
    library_path: Path


def _bool_arg(value: str) -> bool:
    if value in ("0", "false", "False"):
        return False
    if value in ("1", "true", "True"):
        return True
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _register_ep() -> RegisteredEp:
    import onnxruntime as ort
    import onnxruntime_ep_nv_tensorrt_rtx as ep_pkg

    ep_name = ep_pkg.get_ep_name()
    library_path = Path(ep_pkg.get_library_path())
    if not ep_name:
        raise RuntimeError("onnxruntime_ep_nv_tensorrt_rtx.get_ep_name() returned an empty name")
    if not library_path.is_file():
        raise RuntimeError(f"TRT RTX EP library path does not exist: {library_path}")
    if not hasattr(ort, "register_execution_provider_library"):
        raise RuntimeError("onnxruntime build does not expose register_execution_provider_library")
    if not hasattr(ort, "get_ep_devices"):
        raise RuntimeError("onnxruntime build does not expose get_ep_devices")

    try:
        ort.register_execution_provider_library(ep_name, str(library_path))
    except Exception as exc:
        if not [d for d in ort.get_ep_devices() if getattr(d, "ep_name", None) == ep_name]:
            raise RuntimeError(f"Failed to register TRT RTX EP library: {exc}") from exc

    return RegisteredEp(ort=ort, ep_name=ep_name, library_path=library_path)


def _get_trt_rtx_devices(registered_ep: RegisteredEp):
    devices = [
        d for d in registered_ep.ort.get_ep_devices()
        if getattr(d, "ep_name", None) == registered_ep.ep_name
    ]
    if not devices:
        raise RuntimeError(f"No OrtEpDevice found for {registered_ep.ep_name}")
    return devices


def _make_session_options(registered_ep: RegisteredEp, cache_dir: Path):
    ort = registered_ep.ort
    so = ort.SessionOptions()
    if not hasattr(so, "add_provider_for_devices"):
        raise RuntimeError("onnxruntime.SessionOptions does not expose add_provider_for_devices")

    so.add_provider_for_devices(
        _get_trt_rtx_devices(registered_ep),
        {
            "enable_cuda_graph": "1",
            "nv_runtime_cache_path": str(cache_dir),
        },
    )
    return so


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embed-mode", type=_bool_arg, required=True)
    parser.add_argument("--input-from-buffer", type=_bool_arg, required=True)
    parser.add_argument("--output-to-buffer", type=_bool_arg, required=True)
    args = parser.parse_args()

    registered_ep = _register_ep()
    if not hasattr(registered_ep.ort, "ModelCompiler"):
        raise RuntimeError("onnxruntime build does not expose ModelCompiler")

    input_model = args.input_model.read_bytes() if args.input_from_buffer else str(args.input_model)
    so = _make_session_options(registered_ep, args.cache_dir)
    compiler = registered_ep.ort.ModelCompiler(
        so,
        input_model,
        embed_compiled_data_into_model=args.embed_mode,
    )
    if args.input_from_buffer:
        compiler._input_model_bytes_ref = input_model

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    if args.output_to_buffer:
        compiled = compiler.compile_to_bytes()
        if not compiled:
            raise RuntimeError("Output buffer is empty after compilation")
        args.output_model.write_bytes(compiled)
    else:
        compiler.compile_to_file(str(args.output_model))
        if not args.output_model.is_file():
            raise RuntimeError(f"Compiled model not found at: {args.output_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
