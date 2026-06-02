# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent
TESTS_ROOT = PACKAGE_DIR.parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

RESNET18_MODEL_NAME = "resnet18_Opset18_timm.onnx"


@dataclass(frozen=True)
class RegisteredEp:
    ort: object
    ep_name: str
    library_path: Path
    registered_by_fixture: bool


@pytest.fixture
def work_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture(scope="session")
def resnet18_model_path() -> Path:
    candidates = [
        Path.cwd() / RESNET18_MODEL_NAME,
        PACKAGE_DIR / RESNET18_MODEL_NAME,
        TESTS_ROOT / RESNET18_MODEL_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    pytest.skip(
        f"{RESNET18_MODEL_NAME} was not found. Place it in the current "
        f"working directory or under {PACKAGE_DIR}."
    )


@pytest.fixture(scope="session")
def registered_ep() -> RegisteredEp:
    ort = pytest.importorskip("onnxruntime")
    ep_pkg = pytest.importorskip("onnxruntime_ep_nv_tensorrt_rtx")

    ep_name = ep_pkg.get_ep_name()
    library_path = Path(ep_pkg.get_library_path())
    if not ep_name:
        pytest.skip("onnxruntime_ep_nv_tensorrt_rtx.get_ep_name() returned an empty name")
    if not library_path.is_file():
        pytest.skip(f"TRT RTX EP library path does not exist: {library_path}")

    if not hasattr(ort, "register_execution_provider_library"):
        pytest.skip("onnxruntime build does not expose register_execution_provider_library")
    if not hasattr(ort, "get_ep_devices"):
        pytest.skip("onnxruntime build does not expose get_ep_devices")

    registered_by_fixture = False
    try:
        ort.register_execution_provider_library(ep_name, str(library_path))
        registered_by_fixture = True
    except Exception as exc:  # pragma: no cover - depends on installed ORT state
        devices = [
            d for d in getattr(ort, "get_ep_devices", lambda: [])()
            if getattr(d, "ep_name", None) == ep_name
        ]
        if not devices:
            pytest.skip(f"Failed to register TRT RTX EP library: {exc}")

    yield RegisteredEp(
        ort=ort,
        ep_name=ep_name,
        library_path=library_path,
        registered_by_fixture=registered_by_fixture,
    )

    if registered_by_fixture and hasattr(ort, "unregister_execution_provider_library"):
        try:
            ort.unregister_execution_provider_library(ep_name)
        except Exception:
            pass
