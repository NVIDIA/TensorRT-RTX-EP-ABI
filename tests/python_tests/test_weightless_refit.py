# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import onnx

from python_tests import model_builder
from python_tests.ort_helpers import clear_path, create_session, make_session_options, require_model_compiler


def test_weight_stripped_compile_reload_matches_matmul(registered_ep, work_dir):
    model_path = work_dir / "weightless_e2e_matmul.onnx"
    ctx_path = work_dir / "weightless_e2e_matmul_ctx.onnx"
    clear_path(model_path)
    clear_path(ctx_path)
    model_builder.create_initializer_matmul_model(model_path)

    options = {"nv_weight_stripped_engine_enable_experimental": "1"}
    compiler_type = require_model_compiler(registered_ep.ort)
    compiler = compiler_type(
        make_session_options(registered_ep, provider_options=options),
        str(model_path),
        embed_compiled_data_into_model=True,
    )
    compiler.compile_to_file(str(ctx_path))
    assert ctx_path.is_file()

    compiled = onnx.load(str(ctx_path), load_external_data=False)
    ep_context_nodes = [node for node in compiled.graph.node if node.op_type == "EPContext"]
    assert len(ep_context_nodes) == 1
    attributes = {attr.name: attr for attr in ep_context_nodes[0].attribute}
    assert "ep_refit_table" in attributes
    assert attributes["ep_refit_table"].s
    assert "W" in ep_context_nodes[0].input

    session = create_session(
        registered_ep,
        str(ctx_path),
        provider_options=options,
    )
    output = session.run(
        ["Y"],
        {"X": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)},
    )[0]
    np.testing.assert_allclose(output, np.array([[70.0, 80.0, 90.0]], dtype=np.float32), rtol=0, atol=1e-2)
