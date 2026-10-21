# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil

import numpy as np

from python_tests.compile_worker import _get_trt_rtx_devices, _register_ep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    registered_ep = _register_ep()
    so = registered_ep.ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(
        _get_trt_rtx_devices(registered_ep),
        {"nv_runtime_cache_path": str(args.cache_dir)},
    )
    session = registered_ep.ort.InferenceSession(str(args.model), sess_options=so)
    feeds = {}
    for model_input in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in model_input.shape]
        feeds[model_input.name] = np.zeros(shape, dtype=np.float32)
    session.run(None, feeds)

    if not args.cache_dir.is_dir():
        raise RuntimeError("Runtime cache directory was not created")
    shutil.rmtree(args.cache_dir)
    args.cache_dir.write_bytes(b"block final cache write")

    del session
    gc.collect()
    if not args.cache_dir.is_file():
        raise RuntimeError("Blocker file disappeared during session teardown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
