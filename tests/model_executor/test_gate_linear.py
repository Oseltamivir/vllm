# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear
import vllm.model_executor.layers.linear as linear
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


def test_gfx942_m3_fp32_router_eligibility(monkeypatch) -> None:
    monkeypatch.setattr(gate_linear, "_on_gfx942", lambda: True)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)

    layer = GateLinear(
        6144,
        128,
        bias=False,
        params_dtype=torch.float32,
        out_dtype=torch.float32,
    )
    assert layer.allow_rocm_fp32_router_gemm

    unsupported_shape = GateLinear(
        3072,
        256,
        bias=False,
        params_dtype=torch.float32,
        out_dtype=torch.float32,
    )
    assert not unsupported_shape.allow_rocm_fp32_router_gemm

    biased = GateLinear(
        6144,
        128,
        bias=True,
        params_dtype=torch.float32,
        out_dtype=torch.float32,
    )
    assert not biased.allow_rocm_fp32_router_gemm
