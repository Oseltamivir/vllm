# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)


@pytest.mark.parametrize("apply_weights", [True, False])
def test_moe_fused_mul_sum_skips_remote_routes(apply_weights: bool) -> None:
    num_tokens, top_k, hidden_size = 17, 4, 256
    inputs = torch.randn(
        num_tokens,
        top_k,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        device="cuda",
        dtype=torch.float32,
    )
    topk_ids = torch.randint(
        0,
        8,
        (num_tokens, top_k),
        device="cuda",
        dtype=torch.int32,
    )
    expert_map = torch.tensor(
        [0, -1, 1, -1, 2, -1, 3, -1],
        device="cuda",
        dtype=torch.int32,
    )

    local_mask = expert_map[topk_ids] >= 0
    expected = inputs.float()
    if apply_weights:
        expected = expected * topk_weights[..., None]
    expected = (expected * local_mask[..., None]).sum(dim=1).to(inputs.dtype)

    actual = moe_fused_mul_sum(
        inputs,
        topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        apply_weights=apply_weights,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
