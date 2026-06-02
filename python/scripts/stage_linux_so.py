# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Copy the EP shared library and TensorRT RTX shared libraries into the Python package
directory before building a wheel.

The EP enforces loading TensorRT RTX SOs only from the same directory as the EP SO
(see secure_load logic). All must sit next to each other inside
onnxruntime_ep_nv_tensorrt_rtx/.

Environment variables (used when CLI args omitted):
  NV_TRT_RTX_EP_SO    Path to libonnxruntime_providers_nv_tensorrt_rtx.so
  NV_TRT_RTX_LIB_DIR  Directory containing libtensorrt_rtx*.so* and related SOs

Example:
  python scripts/stage_linux_so.py \\
    --ep-so ../build/libonnxruntime_providers_nv_tensorrt_rtx.so \\
    --trt-lib-dir /opt/TensorRT-RTX/lib
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _package_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "onnxruntime_ep_nv_tensorrt_rtx"


def _check_trt_rtx_pair(trt_lib_dir: Path) -> None:
    """Verify the expected TRT RTX SO pair is present (versioned or unversioned)."""
    # Versioned: libtensorrt_rtx.so.1.5, libtensorrt_onnxparser_rtx.so.1.5
    # Unversioned: libtensorrt_rtx.so, libtensorrt_onnxparser_rtx.so
    rtx_matches = [
        p for p in trt_lib_dir.glob("libtensorrt_rtx.so*")
        if p.is_file() and not p.is_symlink()
    ]
    parser_matches = [
        p for p in trt_lib_dir.glob("libtensorrt_onnxparser_rtx.so*")
        if p.is_file() and not p.is_symlink()
    ]

    if not rtx_matches:
        raise FileNotFoundError(
            f"No real libtensorrt_rtx.so* found under {trt_lib_dir}. "
            "Check that NV_TRT_RTX_LIB_DIR points to the TensorRT RTX lib directory."
        )
    if not parser_matches:
        raise FileNotFoundError(
            f"No real libtensorrt_onnxparser_rtx.so* found under {trt_lib_dir}. "
            "Check that NV_TRT_RTX_LIB_DIR points to the TensorRT RTX lib directory."
        )


def stage(ep_so: Path, trt_lib_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not ep_so.is_file():
        raise FileNotFoundError(f"EP shared library not found: {ep_so}")

    # Validate the directory contains the expected TRT RTX SO pair
    _check_trt_rtx_pair(trt_lib_dir)

    # Remove stale SOs from a previous staging run before copying new ones.
    # Without this, upgrading (e.g. .so.1.0 -> .so.1.5) leaves both in the
    # directory and both get packaged into the wheel.
    for stale in list(dest.glob("*.so")) + list(dest.glob("*.so.*")):
        stale.unlink()
        print(f"Removed stale {stale}")

    # Copy EP SO
    shutil.copy2(ep_so, dest / ep_so.name)
    print(f"Copied {ep_so} -> {dest / ep_so.name}")

    # Copy only realfiles (not symlinks) from trt_lib_dir.
    # The wheel ships only versioned realfiles (e.g. libtensorrt_rtx.so.1.5.0).
    # SONAME and unversioned symlinks are created at first import by
    # _ensure_soname_symlinks() in __init__.py. This avoids ZIP duplicate
    # copies (ZIP cannot store symlinks; bdist_wheel follows them).
    for src in sorted(list(trt_lib_dir.glob("*.so")) + list(trt_lib_dir.glob("*.so.*"))):
        if src.is_symlink():
            print(f"Skipped symlink {src.name}")
            continue
        dst = dest / src.name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ep-so",
        type=Path,
        default=None,
        help="Path to libonnxruntime_providers_nv_tensorrt_rtx.so (or set NV_TRT_RTX_EP_SO)",
    )
    parser.add_argument(
        "--trt-lib-dir",
        type=Path,
        default=None,
        help="TensorRT RTX SDK lib directory (or set NV_TRT_RTX_LIB_DIR)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination package directory (default: onnxruntime_ep_nv_tensorrt_rtx next to scripts/)",
    )
    args = parser.parse_args()

    ep_so = args.ep_so
    if ep_so is None:
        env_ep = os.environ.get("NV_TRT_RTX_EP_SO")
        if not env_ep:
            print("error: pass --ep-so or set NV_TRT_RTX_EP_SO", file=sys.stderr)
            return 1
        ep_so = Path(env_ep)

    trt_lib_dir = args.trt_lib_dir
    if trt_lib_dir is None:
        env_lib = os.environ.get("NV_TRT_RTX_LIB_DIR")
        if not env_lib:
            print("error: pass --trt-lib-dir or set NV_TRT_RTX_LIB_DIR", file=sys.stderr)
            return 1
        trt_lib_dir = Path(env_lib)

    dest = args.dest if args.dest is not None else _package_dir()

    try:
        stage(ep_so.resolve(), trt_lib_dir.resolve(), dest.resolve())
    except (FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
