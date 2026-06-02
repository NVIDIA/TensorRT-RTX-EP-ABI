# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setuptools entry for the meta package — injects the exact-versioned cu13 dependency."""

from pathlib import Path
from setuptools import setup

_version = (Path(__file__).parent / "_version.txt").read_text().strip()

setup(install_requires=[f"onnxruntime-ep-nv-tensorrt-rtx-cu13=={_version}"])
