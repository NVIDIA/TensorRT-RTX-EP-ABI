# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Copy the EP DLL, TensorRT RTX DLLs, and CUDA runtime into the Python package directory
before building a wheel.

The EP enforces loading TensorRT RTX DLLs only from the same directory as the EP DLL
(see secure_load.cc). All DLLs must sit next to each other inside
onnxruntime_ep_nv_tensorrt_rtx/.

Note: the EP links against nvcudart_hybrid64.dll (NVIDIA driver component), not
cudart64_*.dll directly. cudart64_*.dll is bundled as a redistribution requirement.

Environment variables (used when CLI args omitted):
  NV_TRT_RTX_EP_DLL   Path to onnxruntime_providers_nv_tensorrt_rtx.dll
  NV_TRT_RTX_LIB_DIR  Directory containing tensorrt_rtx_*.dll and tensorrt_onnxparser_rtx_*.dll
  NV_CUDA_BIN         CUDA bin directory containing cudart64_*.dll
                      (e.g. C:\\CUDA\\v13.1\\bin\\x64 or C:\\CUDA\\v13.1\\bin)

Example:
  python scripts/stage_windows_dlls.py ^
    --ep-dll ..\\trt-rtx-ep-abi\\build\\Release\\onnxruntime_providers_nv_tensorrt_rtx.dll ^
    --trt-lib-dir C:\\TensorRT-RTX\\bin ^
    --cuda-bin "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _package_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "onnxruntime_ep_nv_tensorrt_rtx"


def _glob_one(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {directory}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous: expected exactly one match for {pattern!r} under {directory}, "
            f"found: {[p.name for p in matches]}. Remove old versions and retry."
        )
    return matches[0]


def _resolve_trt_rtx_pair(trt_lib_dir: Path) -> tuple[Path, Path]:
    """Prefer versioned SDK names (tensorrt_rtx_1_1.dll); fall back to unversioned names."""
    try:
        return (
            _glob_one("tensorrt_rtx_*.dll", trt_lib_dir),
            _glob_one("tensorrt_onnxparser_rtx_*.dll", trt_lib_dir),
        )
    except FileNotFoundError:
        rtx = trt_lib_dir / "tensorrt_rtx.dll"
        onnx = trt_lib_dir / "tensorrt_onnxparser_rtx.dll"
        if rtx.is_file() and onnx.is_file():
            return rtx, onnx
        raise FileNotFoundError(
            f"Could not find TensorRT RTX DLL pair under {trt_lib_dir} "
            "(expected tensorrt_rtx_*.dll + tensorrt_onnxparser_rtx_*.dll, or unversioned names)."
        ) from None


def _find_cudart(cuda_bin: Path) -> Path:
    """Return cudart64_*.dll from cuda_bin, also checking bin\\x64 as a fallback."""
    for search in (cuda_bin, cuda_bin / "x64"):
        matches = sorted(search.glob("cudart64_*.dll"))
        if matches:
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple cudart64_*.dll found under {search}: "
                    f"{[p.name for p in matches]}. Remove old versions and retry."
                )
            return matches[0]
    raise FileNotFoundError(
        f"No cudart64_*.dll found under {cuda_bin} or {cuda_bin / 'x64'}. "
        "Pass the directory that contains cudart64_*.dll via --cuda-bin."
    )


def stage(ep_dll: Path, trt_lib_dir: Path, dest: Path, cuda_bin: Path | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if not ep_dll.is_file():
        raise FileNotFoundError(f"EP DLL not found: {ep_dll}")

    # Validate the directory contains the expected TRT RTX DLL pair
    _resolve_trt_rtx_pair(trt_lib_dir)

    # Remove stale DLLs from a previous staging run before copying new ones.
    # Without this, upgrading (e.g. tensorrt_rtx_1_0.dll -> 1_1.dll) leaves both
    # in the directory and both get packaged into the wheel.
    for stale in dest.glob("*.dll"):
        stale.unlink()
        print(f"Removed stale {stale}")

    # Copy EP DLL
    shutil.copy2(ep_dll, dest / ep_dll.name)
    print(f"Copied {ep_dll} -> {dest / ep_dll.name}")

    # Copy ALL DLLs from trt_lib_dir — includes tensorrt_rtx_*.dll,
    # tensorrt_onnxparser_rtx_*.dll, and any other runtime dependencies required
    # by the EP at engine compile time.
    # All must be co-located with the EP DLL (enforced by secure_load.cc).
    for src in sorted(trt_lib_dir.glob("*.dll")):
        dst = dest / src.name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    # Bundle the CUDA runtime alongside the EP DLL as a redistribution requirement.
    # Note: the EP itself links against nvcudart_hybrid64.dll (NVIDIA driver component)
    # rather than cudart64_*.dll directly, so this DLL is not loaded at runtime on
    # current builds. It is included per redistribution policy and as a safety net
    # for future builds that may switch to dynamic CUDA runtime linkage.
    if cuda_bin is not None:
        cudart = _find_cudart(cuda_bin)
        shutil.copy2(cudart, dest / cudart.name)
        print(f"Copied {cudart} -> {dest / cudart.name}")
    else:
        print("Warning: --cuda-bin not provided; cudart64_*.dll will NOT be bundled.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ep-dll",
        type=Path,
        default=None,
        help="Path to onnxruntime_providers_nv_tensorrt_rtx.dll (or set NV_TRT_RTX_EP_DLL)",
    )
    parser.add_argument(
        "--trt-lib-dir",
        type=Path,
        default=None,
        help="TensorRT RTX SDK lib directory (or set NV_TRT_RTX_LIB_DIR)",
    )
    parser.add_argument(
        "--cuda-bin",
        type=Path,
        default=None,
        help="CUDA bin directory containing cudart64_*.dll (or set NV_CUDA_BIN). "
             "Typically <CUDA_HOME>\\bin\\x64 on Windows.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination package directory (default: onnxruntime_ep_nv_tensorrt_rtx next to scripts/)",
    )
    args = parser.parse_args()

    ep_dll = args.ep_dll
    if ep_dll is None:
        env_ep = os.environ.get("NV_TRT_RTX_EP_DLL")
        if not env_ep:
            print("error: pass --ep-dll or set NV_TRT_RTX_EP_DLL", file=sys.stderr)
            return 1
        ep_dll = Path(env_ep)

    trt_lib_dir = args.trt_lib_dir
    if trt_lib_dir is None:
        env_lib = os.environ.get("NV_TRT_RTX_LIB_DIR")
        if not env_lib:
            print("error: pass --trt-lib-dir or set NV_TRT_RTX_LIB_DIR", file=sys.stderr)
            return 1
        trt_lib_dir = Path(env_lib)

    cuda_bin = args.cuda_bin
    if cuda_bin is None:
        env_cuda = os.environ.get("NV_CUDA_BIN")
        if env_cuda:
            cuda_bin = Path(env_cuda)

    dest = args.dest if args.dest is not None else _package_dir()

    try:
        stage(ep_dll.resolve(), trt_lib_dir.resolve(), dest.resolve(),
              cuda_bin.resolve() if cuda_bin is not None else None)
    except (FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
