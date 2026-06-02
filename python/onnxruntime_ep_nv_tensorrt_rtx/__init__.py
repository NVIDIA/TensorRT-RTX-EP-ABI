# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import pathlib
import re
import shutil
import sys

_logger = logging.getLogger(__name__)

if os.name == "nt":
    _ort_spec = importlib.util.find_spec("onnxruntime")
    if _ort_spec and _ort_spec.submodule_search_locations:
        _ort_loc = next(iter(_ort_spec.submodule_search_locations), None)
        if _ort_loc:
            _ort_dll = os.path.join(_ort_loc, "capi", "onnxruntime.dll")
            if os.path.isfile(_ort_dll):
                try:
                    ctypes.CDLL(_ort_dll)
                except OSError as e:
                    _logger.warning("Failed to preload ORT DLL '%s': %s", _ort_dll, e)

from onnxruntime_ep_nv_tensorrt_rtx._version import __version__

__all__ = [
    "__version__",
    "get_ep_name",
    "get_ep_names",
    "get_library_path",
]

_EP_DLL_NAMES = (
    "onnxruntime_providers_nv_tensorrt_rtx.dll",    # Windows
    "libonnxruntime_providers_nv_tensorrt_rtx.so",  # Linux
)
_KNOWN_EP_NAME = "nv_tensorrt_rtx"
_module_dir = pathlib.Path(__file__).parent.resolve()

_LINUX_PRELOAD_ORDER = (
    "libtensorrt_rtx.so.1",
    "libtensorrt_onnxparser_rtx.so.1",
)

_MIN_VALID_ELF_SIZE = 4096
_MAX_TEXT_SHIM_SIZE = 256


def _so_version_key(path: pathlib.Path) -> tuple[int, ...]:
    m = re.search(r'\.so\.(.+)$', path.name)
    if m:
        try:
            return tuple(int(p) for p in m.group(1).split('.'))
        except ValueError:
            pass
    return (0,)


def _derive_aliases(realfile: str, base: str) -> tuple[str, ...]:
    m = re.match(rf'^{re.escape(base)}\.so\.(\d+)', realfile)
    if not m:
        return ()
    soname = f"{base}.so.{m.group(1)}"
    unversioned = f"{base}.so"
    aliases = []
    if soname != realfile:
        aliases.append(soname)
    aliases.append(unversioned)
    return tuple(aliases)


def _discover_soname_links() -> tuple:
    result = []
    for base in ("libtensorrt_rtx", "libtensorrt_onnxparser_rtx"):
        candidates = sorted(_module_dir.glob(f"{base}.so.*.*"), key=_so_version_key, reverse=True)
        realfile = None
        for c in candidates:
            if c.is_file() and not c.is_symlink() and c.stat().st_size >= _MIN_VALID_ELF_SIZE:
                realfile = c.name
                break
        if realfile is None:
            candidates = sorted(_module_dir.glob(f"{base}.so.*"), key=_so_version_key, reverse=True)
            for c in candidates:
                if c.is_file() and not c.is_symlink() and c.stat().st_size >= _MIN_VALID_ELF_SIZE:
                    realfile = c.name
                    break
        if realfile:
            aliases = _derive_aliases(realfile, base)
            if aliases:
                result.append((realfile, aliases))
    return tuple(result)


if sys.platform == "linux" and _module_dir.is_dir():
    _LINUX_SONAME_LINKS = _discover_soname_links()
else:
    _LINUX_SONAME_LINKS = ()

_SONAME_TO_REALFILE = {
    link: real for real, links in _LINUX_SONAME_LINKS for link in links
}


def _is_valid_elf(path: pathlib.Path) -> bool:
    """Check if a file is large enough to be a real shared library (not a text shim)."""
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size >= _MIN_VALID_ELF_SIZE
    except OSError:
        return False


def _replace_with_link(link_path: pathlib.Path, realfile: str) -> bool:
    """Try to replace a file with a symlink, hard-link, or copy of the realfile.

    Attempts in order of preference:
      1. Symlink  (cheapest, preserves single inode)
      2. Hard-link (same inode as realfile, no extra disk space)
      3. Copy     (expensive but works across filesystems and with any permissions
                   the current user has write access to the directory)

    Returns True if any method succeeded.
    """
    real_path = link_path.parent / realfile
    if not real_path.is_file():
        return False

    backup = link_path.with_suffix(link_path.suffix + ".bak")
    try:
        link_path.rename(backup)
    except OSError:
        return False

    try:
        os.symlink(realfile, link_path)
        backup.unlink(missing_ok=True)
        return True
    except OSError:
        pass

    try:
        os.link(real_path, link_path)
        backup.unlink(missing_ok=True)
        return True
    except OSError:
        pass

    try:
        shutil.copy2(real_path, link_path)
        backup.unlink(missing_ok=True)
        return True
    except OSError:
        pass

    try:
        backup.rename(link_path)
    except OSError:
        pass

    return False


def get_library_path() -> str:
    """Return absolute path to the TensorRT RTX EP plugin DLL packaged with this module."""
    for name in _EP_DLL_NAMES:
        candidate = _module_dir / name
        if candidate.is_file():
            return str(candidate)
    names = ", ".join(_EP_DLL_NAMES)
    raise FileNotFoundError(
        f"No EP library found next to this package (looked for {names} under {_module_dir}). "
        "The package may be incomplete; try reinstalling: "
        "pip install --force-reinstall onnxruntime-ep-nv-tensorrt-rtx"
    )


def get_ep_name() -> str:
    """Return the ONNX Runtime execution provider name for this plugin."""
    return _KNOWN_EP_NAME


def get_ep_names() -> list[str]:
    """Return all execution provider names exposed by this plugin package."""
    return [get_ep_name()]


def _prepend_lib_dir_to_search_path() -> None:
    """Inject the package directory into LD_LIBRARY_PATH so SONAME lookups
    performed by the dynamic linker (after the initial RTLD_GLOBAL preload)
    can find bundled libraries.
    """
    if sys.platform != "linux":
        return
    lib_dir = str(_module_dir)
    if not os.path.isdir(lib_dir):
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = existing.split(os.pathsep) if existing else []
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([lib_dir, *parts]) if parts else lib_dir


def _preload_onnxruntime_core() -> None:
    """Preload libonnxruntime.so from the installed onnxruntime package
    so that the EP plugin .so (which has NEEDED libonnxruntime.so.1) can
    resolve it via the dynamic loader's already-loaded list.

    The loader registers the library's embedded SONAME on load, so
    preloading the fully-versioned file (e.g. libonnxruntime.so.1.25.1)
    is sufficient — no SONAME symlink is needed in ORT's directory.
    """
    if sys.platform != "linux":
        return
    try:
        import onnxruntime as _ort  # noqa: F401
    except ImportError:
        return
    if not getattr(_ort, "__file__", None):
        _logger.warning(
            "onnxruntime.__file__ is None (namespace/frozen package); "
            "cannot preload libonnxruntime.so — set LD_LIBRARY_PATH manually"
        )
        return
    ort_capi = pathlib.Path(_ort.__file__).resolve().parent / "capi"

    for soname in ("libonnxruntime.so.1", "libonnxruntime.so"):
        candidate = ort_capi / soname
        if candidate.is_file() or candidate.is_symlink():
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError as e:
                _logger.debug("Failed to preload '%s': %s", candidate, e)
                continue

    for candidate in sorted(ort_capi.glob("libonnxruntime.so.1.*"), key=_so_version_key, reverse=True):
        if candidate.is_file() and candidate.stat().st_size > _MIN_VALID_ELF_SIZE:
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError as e:
                _logger.debug("Failed to preload '%s': %s", candidate, e)
            break


def _ensure_soname_symlinks() -> None:
    """Create SONAME and unversioned symlinks pointing to the canonical
    versioned realfile.

    The wheel ships only the versioned realfile (e.g. libtensorrt_rtx.so.1.5.0).
    This function derives and creates the needed aliases:
      - SONAME alias:      libtensorrt_rtx.so.1  -> libtensorrt_rtx.so.1.5.0
      - Unversioned alias: libtensorrt_rtx.so     -> libtensorrt_rtx.so.1.5.0

    Both are required at runtime:
      - SONAME: satisfies NEEDED deps from other .so files
      - Unversioned: the TRT-RTX SDK internally calls dlopen("libtensorrt_rtx.so")
        during session creation (plugin system), which does NOT match SONAME

    Also handles legacy cases (duplicate realfiles, stale symlinks) from older
    wheel formats.

    Fallback chain per link: symlink -> hard-link -> copy.
    If all fail, _preload_bundled_libs() will still preload the realfile
    which covers SONAME resolution (but NOT unversioned dlopen lookups).
    """
    if sys.platform != "linux":
        return
    if not _module_dir.is_dir():
        return

    unresolved = []

    for realfile, links in _LINUX_SONAME_LINKS:
        real_path = _module_dir / realfile
        if not real_path.is_file():
            continue
        try:
            real_size = real_path.stat().st_size
        except OSError:
            continue
        for link_name in links:
            link_path = _module_dir / link_name
            try:
                if link_path.is_symlink():
                    if os.readlink(link_path) == realfile:
                        continue
                    link_path.unlink()
                    os.symlink(realfile, link_path)
                    continue
                elif link_path.exists():
                    link_size = link_path.stat().st_size
                    if link_size == real_size:
                        if link_path.stat().st_ino == real_path.stat().st_ino:
                            continue
                    if link_size == real_size or link_size < _MAX_TEXT_SHIM_SIZE:
                        if not _replace_with_link(link_path, realfile):
                            unresolved.append(link_name)
                        continue
                    continue
                os.symlink(realfile, link_path)
            except OSError:
                if not _replace_with_link(link_path, realfile):
                    unresolved.append(link_name)

    if unresolved:
        ep_dir = str(_module_dir)
        _logger.warning(
            "Could not fix library symlinks in '%s' for: %s. "
            "TRT-RTX session creation may fail with SIGSEGV or dlopen errors. "
            "Fix with: sudo bash -c 'cd \"%s\" && for f in %s; do "
            "t=$(cat \"$f\" 2>/dev/null || echo \"%s\"); "
            "rm -f \"$f\" && ln -s \"$t\" \"$f\"; done'",
            ep_dir,
            ", ".join(unresolved),
            ep_dir,
            " ".join(unresolved),
            next(iter(_LINUX_SONAME_LINKS))[0],
        )


def _preload_bundled_libs() -> None:
    """Preload bundled TRT-RTX libraries with RTLD_GLOBAL so their symbols
    are visible to the subsequent dlopen of the EP plugin .so performed by
    ORT's register_execution_provider_library().

    If the SONAME file is missing or not a valid ELF, falls back to loading
    the canonical versioned realfile directly.

    Order matters: libtensorrt_rtx must be loaded before
    libtensorrt_onnxparser_rtx.
    """
    if sys.platform != "linux":
        return
    if not _module_dir.is_dir():
        return
    for soname in _LINUX_PRELOAD_ORDER:
        candidate = _module_dir / soname
        if candidate.is_symlink() or _is_valid_elf(candidate):
            pass
        else:
            real = _SONAME_TO_REALFILE.get(soname)
            if real:
                fallback = _module_dir / real
                if fallback.is_file():
                    candidate = fallback
                else:
                    continue
            else:
                continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            real = _SONAME_TO_REALFILE.get(soname)
            if real:
                fallback = _module_dir / real
                if fallback.is_file() and str(fallback) != str(candidate):
                    try:
                        ctypes.CDLL(str(fallback), mode=ctypes.RTLD_GLOBAL)
                        continue
                    except OSError:
                        pass
            _logger.warning("Failed to preload '%s': %s", candidate, e)


if sys.platform == "linux":
    _prepend_lib_dir_to_search_path()
    _ensure_soname_symlinks()
    _preload_bundled_libs()
    _preload_onnxruntime_core()
