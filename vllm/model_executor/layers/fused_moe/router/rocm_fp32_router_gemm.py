# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch BF16 activation by FP32 router projection for gfx942."""

import torch

from vllm.triton_utils import tl, triton

_HIDDEN_SIZE = 6144
_NUM_EXPERTS = 128
_BLOCK_M = 4
_BLOCK_K = 256


@triton.jit
def _rocm_fp32_router_gemm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    HIDDEN_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    token_offsets = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for hidden_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_K):
        hidden_states = tl.load(
            x_ptr
            + token_offsets[:, None] * HIDDEN_SIZE
            + hidden_start
            + hidden_offsets[None, :],
            mask=token_offsets[:, None] < num_tokens,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + expert * HIDDEN_SIZE + hidden_start + hidden_offsets,
        ).to(tl.float32)
        accumulator += tl.sum(hidden_states * weight[None, :], axis=1)

    tl.store(
        output_ptr + token_offsets * NUM_EXPERTS + expert,
        accumulator,
        mask=token_offsets < num_tokens,
    )


def rocm_fp32_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != _HIDDEN_SIZE:
        raise ValueError("hidden_states must have shape [num_tokens, 6144]")
    if router_weight.shape != (_NUM_EXPERTS, _HIDDEN_SIZE):
        raise ValueError("router_weight must have shape [128, 6144]")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be bfloat16")
    if router_weight.dtype != torch.float32:
        raise ValueError("router_weight must be float32")
    if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
        raise ValueError("hidden_states and router_weight must be contiguous")

    num_tokens = hidden_states.shape[0]
    output = torch.empty(
        (num_tokens, _NUM_EXPERTS),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    _rocm_fp32_router_gemm_kernel[(_NUM_EXPERTS, triton.cdiv(num_tokens, _BLOCK_M))](
        hidden_states,
        router_weight,
        output,
        num_tokens,
        HIDDEN_SIZE=_HIDDEN_SIZE,
        NUM_EXPERTS=_NUM_EXPERTS,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return output
