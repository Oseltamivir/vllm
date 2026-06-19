# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only router kernel", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.router.gate_linear import (  # noqa: E402
    rocm_fp32_router_gemm_dispatch_impl,
)
from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (  # noqa: E402
    rocm_fp32_router_gemm,
)
from vllm.platforms.rocm import on_gfx942  # noqa: E402

if not on_gfx942():
    pytest.skip("The router kernel is tuned for gfx942", allow_module_level=True)


@pytest.mark.parametrize("num_tokens", [1, 16])
def test_rocm_fp32_router_gemm_matches_linear(num_tokens: int) -> None:
    torch.manual_seed(20260619 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        6144,
        device="cuda",
        dtype=torch.bfloat16,
    )
    router_weight = torch.randn(128, 6144, device="cuda", dtype=torch.float32) * 0.01

    reference = torch.nn.functional.linear(hidden_states.float(), router_weight)
    output = rocm_fp32_router_gemm(hidden_states, router_weight)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)
    output_top4 = output.topk(4, dim=-1).indices.sort(dim=-1).values
    reference_top4 = reference.topk(4, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(output_top4, reference_top4)


def test_rocm_fp32_router_gemm_graph_capture() -> None:
    torch.manual_seed(7)
    hidden_states = torch.randn(
        16,
        6144,
        device="cuda",
        dtype=torch.bfloat16,
    )
    router_weight = torch.randn(128, 6144, device="cuda", dtype=torch.float32) * 0.01
    rocm_fp32_router_gemm(hidden_states, router_weight)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.ops.vllm.rocm_fp32_router_gemm_dispatch(
            hidden_states,
            router_weight,
        )
    graph.replay()
    torch.cuda.synchronize()

    reference = torch.nn.functional.linear(hidden_states.float(), router_weight)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=2e-6)


def test_rocm_fp32_router_gemm_large_batch_falls_back() -> None:
    torch.manual_seed(9)
    hidden_states = torch.randn(
        256,
        6144,
        device="cuda",
        dtype=torch.bfloat16,
    )
    router_weight = torch.randn(128, 6144, device="cuda", dtype=torch.float32) * 0.01

    output = rocm_fp32_router_gemm_dispatch_impl(hidden_states, router_weight)
    reference = torch.nn.functional.linear(hidden_states.float(), router_weight)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
