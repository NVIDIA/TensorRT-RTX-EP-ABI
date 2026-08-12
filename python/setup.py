# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setuptools entry: platform wheel tagged py3-none-<plat> for any Python 3."""

import os
import platform
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# ---------------------------------------------------------------------------
# CUDA major suffix — set NV_CUDA_MAJOR=13 to produce onnxruntime-ep-nv-tensorrt-rtx-cu13
# The base name in pyproject.toml is overridden in parse_config_files() below.
# ---------------------------------------------------------------------------
_cuda_major = os.environ.get("NV_CUDA_MAJOR", "").strip()

# ---------------------------------------------------------------------------
# Platform tag
# ---------------------------------------------------------------------------
_PLAT_TAG = {
    ("Windows", "x86_64"): "win_amd64",
    ("Windows", "AMD64"):  "win_amd64",
    ("Windows", "ARM64"):  "win_arm64",
    ("Linux",   "x86_64"): "linux_x86_64",
}


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def parse_config_files(self):
        super().parse_config_files()
        # Append CUDA suffix to the name read from pyproject.toml.
        # This runs after setuptools validates pyproject.toml, so the base name
        # satisfies the validator while the wheel gets the correct suffixed name.
        effective_cuda = _cuda_major or "12"
        if _cuda_major:
            self.metadata.name = f"onnxruntime-ep-nv-tensorrt-rtx-cu{_cuda_major}"
        # Substitute {CUDA_MAJOR} placeholder in README so each variant's PyPI
        # description shows the correct driver requirement.
        readme = Path(__file__).parent / "README.md"
        self.metadata.long_description = readme.read_text(encoding="utf-8").replace(
            "{CUDA_MAJOR}", effective_cuda
        )
        self.metadata.long_description_content_type = "text/markdown"


class bdist_wheel(_bdist_wheel):
    """No Python ABI coupling — only packaged binaries + pure Python helpers."""

    def get_tag(self):
        # NV_TARGET_PLAT lets build.bat override the tag when cross-compiling
        # (e.g. building win_arm64 on an x64 host where platform.machine()=="AMD64").
        env_plat = os.environ.get("NV_TARGET_PLAT", "").strip()
        if env_plat:
            return "py3", "none", env_plat
        system = platform.system()
        machine = platform.machine()
        plat = _PLAT_TAG.get((system, machine))
        if plat is None:
            # Fall back to whatever setuptools detects for unknown platforms
            _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
