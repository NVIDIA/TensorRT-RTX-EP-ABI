# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Copy the EP DLL, TensorRT RTX DLLs, CUDA runtime, docs, and optional plugin DLLs
into the Python package directory before building a wheel.

Conventional x64 packages keep TensorRT RTX DLLs beside the EP DLL. ARM64X
packages keep the two EP implementations beside the public forwarder and place
their NVIDIA dependencies in arm64ec/ and arm64/ subdirectories. The secure
loader enforces the corresponding layout.

Note: the EP links against nvcudart_hybrid64.dll (NVIDIA driver component), not
cudart64_*.dll directly. cudart64_*.dll is bundled as a redistribution requirement.

Environment variables (used when CLI args omitted):
  NV_TRT_RTX_EP_DLL   Path to onnxruntime_providers_nv_tensorrt_rtx.dll
  NV_TRT_RTX_LIB_DIR  Directory containing tensorrt_rtx_*.dll and tensorrt_onnxparser_rtx_*.dll
  NV_CUDA_BIN         CUDA bin directory containing cudart64_*.dll
                      (e.g. C:\\CUDA\\v13.1\\bin\\x64 or C:\\CUDA\\v13.1\\bin)
  NV_TRT_RTX_DOC_DIR  TensorRT RTX SDK doc directory (e.g. C:\\TensorRT-RTX\\doc)
  NV_PLUGIN_DIR       TensorRT plugin directory containing tensorrt_plugins.dll

Example:
  python scripts/stage_windows_dlls.py ^
    --ep-dll ..\\trt-rtx-ep-abi\\build\\Release\\onnxruntime_providers_nv_tensorrt_rtx.dll ^
    --trt-lib-dir C:\\TensorRT-RTX\\bin ^
    --cuda-bin "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64" ^
    --trt-doc-dir C:\\TensorRT-RTX\\doc ^
    --plugin-dir "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\TensorRT-RTX-Plugins\\RTX-1.5"

ARM64X bundle example:
  python scripts\\stage_windows_dlls.py ^
    --arm64x-forwarder build-arm64x\\Release\\onnxruntime_providers_nv_tensorrt_rtx.dll ^
    --arm64ec-ep-dll build-arm64ec\\Release\\onnxruntime_providers_nv_tensorrt_rtx_arm64ec.dll ^
    --arm64-ep-dll build-arm64\\Release\\onnxruntime_providers_nv_tensorrt_rtx_arm64.dll ^
    --arm64ec-trt-lib-dir C:\\TRT-ARM64EC\\bin ^
    --arm64-trt-lib-dir C:\\TRT-ARM64\\bin ^
    --arm64ec-cuda-bin C:\\CUDA-ARM64EC\\bin ^
    --arm64-cuda-bin C:\\CUDA-ARM64\\bin
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _package_dir() -> Path:
    """Return the default Python package directory used for DLL staging."""
    return Path(__file__).resolve().parent.parent / "onnxruntime_ep_nv_tensorrt_rtx"


def _glob_one(pattern: str, directory: Path) -> Path:
    """Return the unique file matching pattern under directory."""
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


def _clear_dlls(directory: Path) -> None:
    """Remove previously staged DLLs from directory when it exists."""
    if not directory.is_dir():
        return
    for stale in directory.glob("*.dll"):
        stale.unlink()
        print(f"Removed stale {stale}")


def _copy_runtime_tree(
    source: Path,
    destination: Path,
    cuda_bin: Path | None,
    plugin_dir: Path | None = None,
) -> None:
    """Stage one architecture's TensorRT RTX, CUDA, and optional plugin DLLs."""
    _resolve_trt_rtx_pair(source)
    destination.mkdir(parents=True, exist_ok=True)
    _clear_dlls(destination)

    for src in sorted(source.glob("*.dll")):
        dst = destination / src.name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

    if cuda_bin is not None:
        cudart = _find_cudart(cuda_bin)
        dst = destination / cudart.name
        shutil.copy2(cudart, dst)
        print(f"Copied {cudart} -> {dst}")
    else:
        print(f"Warning: CUDA bin not provided for {destination.name}; cudart will NOT be bundled.")

    if plugin_dir is not None:
        plugin_dll = plugin_dir / "tensorrt_plugins.dll"
        if not plugin_dll.is_file():
            raise FileNotFoundError(f"TensorRT plugin DLL not found: {plugin_dll}")
        dst = destination / plugin_dll.name
        shutil.copy2(plugin_dll, dst)
        print(f"Copied {plugin_dll} -> {dst}")
    else:
        print(
            f"Warning: plugin directory not provided for {destination.name}; "
            "tensorrt_plugins.dll will NOT be bundled."
        )


def stage(
    ep_dll: Path,
    trt_lib_dir: Path,
    dest: Path,
    cuda_bin: Path | None = None,
    trt_doc_dir: Path | None = None,
    plugin_dir: Path | None = None,
) -> None:
    """Stage a standalone EP and its flat runtime dependencies for a wheel."""
    dest.mkdir(parents=True, exist_ok=True)

    if not ep_dll.is_file():
        raise FileNotFoundError(f"EP DLL not found: {ep_dll}")

    # Validate the directory contains the expected TRT RTX DLL pair
    _resolve_trt_rtx_pair(trt_lib_dir)

    # Remove stale DLLs and staged TRT-RTX docs from a previous staging run.
    # Without this, upgrading (e.g. tensorrt_rtx_1_0.dll -> 1_1.dll) leaves both
    # in the directory and both get packaged into the wheel.
    # Use TRTRTX_*.txt rather than *.txt to avoid deleting any checked-in package files.
    for stale in dest.glob("TRTRTX_*.txt"):
        stale.unlink()
        print(f"Removed stale {stale}")
    _clear_dlls(dest)
    # Remove stale native-package contents when switching back to a wheel build.
    _clear_dlls(dest / "arm64ec")
    _clear_dlls(dest / "arm64")

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

    # Copy TensorRT RTX SDK docs (Acknowledgements.txt, README.txt) from doc/.
    # Renamed with TRTRTX_ prefix so wheel consumers can distinguish them from EP docs.
    if trt_doc_dir is not None:
        if not trt_doc_dir.is_dir():
            raise FileNotFoundError(f"TRT RTX doc directory not found: {trt_doc_dir}")
        doc_files = sorted(trt_doc_dir.glob("*.txt"))
        if not doc_files:
            print(f"Warning: no *.txt files found under {trt_doc_dir}; docs will not be bundled.")
        for src in doc_files:
            dst = dest / f"TRTRTX_{src.name}"
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")
    else:
        print("Warning: --trt-doc-dir not provided; TensorRT RTX docs will NOT be bundled.")

    # Copy TensorRT plugin DLL(s) from the plugin directory.
    if plugin_dir is not None:
        if not plugin_dir.is_dir():
            raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")
        plugin_dlls = sorted(plugin_dir.glob("*.dll"))
        if not plugin_dlls:
            print(f"Warning: no *.dll files found under {plugin_dir}.")
        for src in plugin_dlls:
            shutil.copy2(src, dest / src.name)
            print(f"Copied {src} -> {dest / src.name}")
    else:
        print("Warning: --plugin-dir not provided; TensorRT plugin DLLs will NOT be bundled.")


def stage_arm64x_bundle(
    forwarder: Path,
    arm64ec_ep_dll: Path,
    arm64_ep_dll: Path,
    arm64ec_trt_lib_dir: Path,
    arm64_trt_lib_dir: Path,
    dest: Path,
    arm64ec_cuda_bin: Path | None = None,
    arm64_cuda_bin: Path | None = None,
    arm64ec_plugin_dir: Path | None = None,
    arm64_plugin_dir: Path | None = None,
) -> None:
    """Stage one ARM64X public EP and its two architecture implementations."""
    expected_names = {
        forwarder: "onnxruntime_providers_nv_tensorrt_rtx.dll",
        arm64ec_ep_dll: "onnxruntime_providers_nv_tensorrt_rtx_arm64ec.dll",
        arm64_ep_dll: "onnxruntime_providers_nv_tensorrt_rtx_arm64.dll",
    }
    for source, expected_name in expected_names.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required ARM64X bundle DLL not found: {source}")
        if source.name.lower() != expected_name:
            raise ValueError(
                f"Unexpected DLL name {source.name!r}; ARM64X bundle requires {expected_name!r}"
            )

    dest.mkdir(parents=True, exist_ok=True)
    _clear_dlls(dest)

    for source in (forwarder, arm64ec_ep_dll, arm64_ep_dll):
        target = dest / source.name
        shutil.copy2(source, target)
        print(f"Copied {source} -> {target}")

    _copy_runtime_tree(
        arm64ec_trt_lib_dir,
        dest / "arm64ec",
        arm64ec_cuda_bin,
        arm64ec_plugin_dir,
    )
    _copy_runtime_tree(
        arm64_trt_lib_dir,
        dest / "arm64",
        arm64_cuda_bin,
        arm64_plugin_dir,
    )


def main() -> int:
    """Parse staging arguments and build either a standalone or ARM64X layout."""
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
        help="TensorRT RTX SDK bin directory containing DLLs (or set NV_TRT_RTX_LIB_DIR)",
    )
    parser.add_argument(
        "--cuda-bin",
        type=Path,
        default=None,
        help="CUDA bin directory containing cudart64_*.dll (or set NV_CUDA_BIN). "
             "Typically <CUDA_HOME>\\bin\\x64 on Windows.",
    )
    parser.add_argument(
        "--trt-doc-dir",
        type=Path,
        default=None,
        help="TensorRT RTX SDK doc directory containing Acknowledgements.txt and README.txt "
             "(or set NV_TRT_RTX_DOC_DIR). Typically <TRT_RTX_HOME>\\doc.",
    )
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        default=None,
        help="TensorRT plugin directory containing tensorrt_plugins.dll and its docs "
             "(or set NV_PLUGIN_DIR). E.g. C:\\TensorRT-RTX-Plugins\\RTX-1.5.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination package directory (default: onnxruntime_ep_nv_tensorrt_rtx next to scripts/)",
    )
    parser.add_argument(
        "--arm64x-forwarder",
        type=Path,
        default=None,
        help="Public ARM64X onnxruntime_providers_nv_tensorrt_rtx.dll",
    )
    parser.add_argument("--arm64ec-ep-dll", type=Path, default=None)
    parser.add_argument("--arm64-ep-dll", type=Path, default=None)
    parser.add_argument("--arm64ec-trt-lib-dir", type=Path, default=None)
    parser.add_argument("--arm64-trt-lib-dir", type=Path, default=None)
    parser.add_argument("--arm64ec-cuda-bin", type=Path, default=None)
    parser.add_argument("--arm64-cuda-bin", type=Path, default=None)
    parser.add_argument("--arm64ec-plugin-dir", type=Path, default=None)
    parser.add_argument("--arm64-plugin-dir", type=Path, default=None)
    args = parser.parse_args()

    dest = args.dest if args.dest is not None else _package_dir()

    if args.arm64x_forwarder is not None:
        required_bundle_args = {
            "--arm64ec-ep-dll": args.arm64ec_ep_dll,
            "--arm64-ep-dll": args.arm64_ep_dll,
            "--arm64ec-trt-lib-dir": args.arm64ec_trt_lib_dir,
            "--arm64-trt-lib-dir": args.arm64_trt_lib_dir,
        }
        missing = [name for name, value in required_bundle_args.items() if value is None]
        if missing:
            parser.error(
                "--arm64x-forwarder requires " + ", ".join(missing)
            )

        try:
            stage_arm64x_bundle(
                args.arm64x_forwarder.resolve(),
                args.arm64ec_ep_dll.resolve(),
                args.arm64_ep_dll.resolve(),
                args.arm64ec_trt_lib_dir.resolve(),
                args.arm64_trt_lib_dir.resolve(),
                dest.resolve(),
                args.arm64ec_cuda_bin.resolve() if args.arm64ec_cuda_bin else None,
                args.arm64_cuda_bin.resolve() if args.arm64_cuda_bin else None,
                args.arm64ec_plugin_dir.resolve() if args.arm64ec_plugin_dir else None,
                args.arm64_plugin_dir.resolve() if args.arm64_plugin_dir else None,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

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

    trt_doc_dir = args.trt_doc_dir
    if trt_doc_dir is None:
        env_doc = os.environ.get("NV_TRT_RTX_DOC_DIR")
        if env_doc:
            trt_doc_dir = Path(env_doc)

    plugin_dir = args.plugin_dir
    if plugin_dir is None:
        env_plugin = os.environ.get("NV_PLUGIN_DIR")
        if env_plugin:
            plugin_dir = Path(env_plugin)

    try:
        stage(
            ep_dll.resolve(),
            trt_lib_dir.resolve(),
            dest.resolve(),
            cuda_bin.resolve() if cuda_bin is not None else None,
            trt_doc_dir.resolve() if trt_doc_dir is not None else None,
            plugin_dir.resolve() if plugin_dir is not None else None,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
