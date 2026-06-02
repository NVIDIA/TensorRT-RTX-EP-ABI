# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from python_tests.ort_helpers import get_trt_rtx_devices


def test_wheel_exports_library_path_and_ep_name(registered_ep):
    assert registered_ep.ep_name == "nv_tensorrt_rtx"
    assert registered_ep.library_path.is_file()


def test_registered_ep_device_is_visible(registered_ep):
    devices = get_trt_rtx_devices(registered_ep)
    assert len(devices) >= 1

    device = devices[0]
    assert getattr(device, "ep_name") == registered_ep.ep_name

    hardware = getattr(device, "device", None)
    if hardware is not None and hasattr(hardware, "metadata"):
        assert isinstance(hardware.metadata, dict)


def test_unregister_and_reregister_ep_library(registered_ep):
    ort = registered_ep.ort
    if not hasattr(ort, "unregister_execution_provider_library"):
        pytest.skip("onnxruntime build does not expose unregister_execution_provider_library")

    ort.unregister_execution_provider_library(registered_ep.ep_name)
    try:
        ort.register_execution_provider_library(
            registered_ep.ep_name,
            str(registered_ep.library_path),
        )
        assert len(get_trt_rtx_devices(registered_ep)) >= 1
    finally:
        if not [
            d for d in ort.get_ep_devices()
            if getattr(d, "ep_name", None) == registered_ep.ep_name
        ]:
            ort.register_execution_provider_library(
                registered_ep.ep_name,
                str(registered_ep.library_path),
            )
