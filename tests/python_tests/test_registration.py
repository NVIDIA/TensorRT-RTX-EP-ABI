# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

import pytest

from python_tests.ort_helpers import create_session, get_trt_rtx_devices, run_session_once, assert_base_model_output, assert_nv_ep_session


def test_wheel_exports_library_path_and_ep_name(registered_ep):
    assert registered_ep.ep_name == "NvTensorRTRTXExecutionProvider"
    assert registered_ep.library_path.is_file()


def test_registered_ep_device_is_visible(registered_ep):
    devices = get_trt_rtx_devices(registered_ep)
    assert len(devices) >= 1

    device = devices[0]
    assert getattr(device, "ep_name") == registered_ep.ep_name
    assert getattr(device, "ep_vendor") == "NVIDIA"

    hardware = getattr(device, "device", None)
    assert hardware is not None
    assert hardware.type == registered_ep.ort.OrtHardwareDeviceType.GPU
    assert hardware.vendor == "NVIDIA"
    assert hardware.vendor_id > 0
    assert hardware.device_id >= 0
    assert isinstance(hardware.metadata, dict)
    assert hardware.metadata


def test_custom_registration_name_works(registered_ep, resnet18_model_path):
    """The EP name is caller-assigned at registration time, not hardcoded by
    the DLL -- unregister the default name and re-register the same library
    under an arbitrary custom name (never two simultaneous registrations of
    the same library, mirroring test_unregister_and_reregister_ep_library)
    and confirm inference still works identically."""
    ort = registered_ep.ort
    if not hasattr(ort, "unregister_execution_provider_library"):
        pytest.skip("onnxruntime build does not expose unregister_execution_provider_library")

    custom_name = "my_custom_trt_rtx_ep_name"

    ort.unregister_execution_provider_library(registered_ep.ep_name)
    try:
        ort.register_execution_provider_library(custom_name, str(registered_ep.library_path))
        try:
            custom_ep = dataclasses.replace(registered_ep, ep_name=custom_name)
            devices = get_trt_rtx_devices(custom_ep)
            assert len(devices) >= 1

            session = create_session(custom_ep, str(resnet18_model_path))
            run_session_once(session)
        finally:
            ort.unregister_execution_provider_library(custom_name)
    finally:
        if not [
            d for d in ort.get_ep_devices()
            if getattr(d, "ep_name", None) == registered_ep.ep_name
        ]:
            ort.register_execution_provider_library(
                registered_ep.ep_name,
                str(registered_ep.library_path),
            )


def test_unregister_and_reregister_ep_library(registered_ep):
    ort = registered_ep.ort
    if not hasattr(ort, "unregister_execution_provider_library"):
        pytest.fail("onnxruntime build does not expose required unregister_execution_provider_library")

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


def test_auto_ep_prefer_gpu_selects_registered_ep(registered_ep, work_dir):
    ort = registered_ep.ort
    if not hasattr(ort.SessionOptions(), "set_provider_selection_policy"):
        pytest.fail("onnxruntime build does not expose required automatic EP selection")
    if not hasattr(ort, "OrtExecutionProviderDevicePolicy"):
        pytest.fail("onnxruntime build does not expose required EP device policies")

    from python_tests import model_builder
    from python_tests.ort_helpers import clear_path, read_profile_text

    model_path = work_dir / "nv_execution_provider_auto_ep.onnx"
    clear_path(model_path)
    model_builder.create_base_model(model_path)

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.enable_profiling = True
    so.profile_file_prefix = str(work_dir / "auto_ep_prefer_gpu_profile")
    so.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_GPU)
    session = ort.InferenceSession(str(model_path), sess_options=so)
    assert_nv_ep_session(session, registered_ep)
    assert_base_model_output(session)
    profile = read_profile_text(session)
    assert registered_ep.ep_name in profile
