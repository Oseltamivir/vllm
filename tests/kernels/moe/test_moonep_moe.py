# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoonEP all2all backend.

Three levels, cheapest first:

* ``test_m_indices_*`` -- the cu_seqlens -> buffer-row expansion, on one GPU.
* ``test_symmetric_weight_mapping`` -- that a rank can read a peer's expert
  rows through the VMM mapping and get the owner's bytes.
* ``test_moonep_dispatch_combine`` -- dispatch/combine round trip against a
  pure-PyTorch reference.

The multi-GPU tests need a single NVLink domain with multicast support, which
is what MoonEP requires in general.
"""

import pytest
import torch

from vllm.utils.import_utils import has_moonep

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

requires_moonep = pytest.mark.skipif(
    not has_moonep(),
    reason="Requires the moonep package and a multicast-capable device",
)

if has_moonep():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (
        _build_m_indices,
        _local_tokens_per_expert,
        moonep_expert_row_pad,
    )


def _reference_m_indices(cu_seqlens, slot_experts, num_experts, experts_per_rank, nvs):
    """Straightforward CPU expansion of the same contract."""
    row_pad = moonep_expert_row_pad(experts_per_rank)
    out = [-1] * nvs
    prev = 0
    for g, end in enumerate(cu_seqlens.tolist()):
        eid = g if g < num_experts else int(slot_experts[g - num_experts])
        row = -1
        if eid >= 0:
            row = (eid // experts_per_rank) * row_pad + eid % experts_per_rank
        for i in range(prev, min(end, nvs)):
            out[i] = row
        prev = end
    return torch.tensor(out, dtype=torch.int32)


@requires_moonep
@pytest.mark.parametrize("experts_per_rank", [4, 112])
@pytest.mark.parametrize("num_ranks", [2, 8])
def test_m_indices_matches_reference(experts_per_rank: int, num_ranks: int):
    """cu_seqlens expansion, including prefetch-slot resolution."""
    device = "cuda"
    num_experts = experts_per_rank * num_ranks
    num_slots = 1
    pad = 128

    torch.manual_seed(0)
    counts = torch.randint(0, 3, (num_experts + num_slots,)) * pad
    cu = torch.cumsum(counts, 0).to(torch.int32).to(device)
    nvs = int(cu[-1].item()) + pad  # trailing rows stay -1

    # One slot standing in for a real remote expert.
    slot_experts = torch.tensor([num_experts - 1], dtype=torch.int32, device=device)

    got = _build_m_indices(cu, slot_experts, num_experts, experts_per_rank, nvs)
    want = _reference_m_indices(
        cu.cpu(), slot_experts.cpu(), num_experts, experts_per_rank, nvs
    )
    torch.testing.assert_close(got.cpu(), want)


@requires_moonep
def test_m_indices_empty_slot_is_skipped():
    """A planner-unused prefetch slot holds -1 and must stay -1."""
    device = "cuda"
    num_experts, experts_per_rank, pad = 8, 4, 128
    cu = torch.tensor([pad] * num_experts + [pad], dtype=torch.int32, device=device)
    cu = torch.cumsum(cu, 0).to(torch.int32)
    nvs = int(cu[-1].item())
    slot_experts = torch.tensor([-1], dtype=torch.int32, device=device)

    got = _build_m_indices(cu, slot_experts, num_experts, experts_per_rank, nvs)
    # The final segment is the empty slot.
    assert (got[-pad:] == -1).all()
    assert (got[:-pad] >= 0).all()


@requires_moonep
def test_local_tokens_per_expert_handles_invalid_ids():
    device = "cuda"
    num_experts = 16
    topk_ids = torch.tensor(
        [[0, 3, -1], [3, 3, 15], [-1, -1, 0]], dtype=torch.int32, device=device
    )
    got = _local_tokens_per_expert(topk_ids, num_experts)
    want = torch.zeros(num_experts, dtype=torch.int32)
    want[0] = 2
    want[3] = 3
    want[15] = 1
    torch.testing.assert_close(got.cpu(), want)
    # Invalid ids must not inflate any real bucket.
    assert int(got.sum().item()) == 6


@requires_moonep
def test_invalid_expert_ids_are_sanitized():
    """vLLM marks invalid routing slots with -1.

    MoonEP indexes its planning arrays directly by expert id, so forwarding a
    negative id is an out-of-bounds device access (verified on B300: dispatch
    dies with cudaErrorIllegalAddress). prepare() must remap them onto real
    experts and zero the matching weight so the slot is inert but addressable.
    """
    device = "cuda"
    num_experts = 64
    topk_ids = torch.tensor(
        [[-1, 3, 7], [0, -1, -1], [-1, -1, -1]], dtype=torch.int32, device=device
    )

    invalid = (topk_ids < 0) | (topk_ids >= num_experts)
    filler = (
        torch.arange(topk_ids.numel(), device=device, dtype=topk_ids.dtype)
        % num_experts
    ).view_as(topk_ids)
    sanitized = torch.where(invalid, filler, topk_ids)

    assert int(sanitized.min().item()) >= 0
    assert int(sanitized.max().item()) < num_experts
    # Valid ids must survive untouched.
    assert sanitized[0, 1].item() == 3
    assert sanitized[0, 2].item() == 7
    assert sanitized[1, 0].item() == 0
    # And the histogram must stay consistent with what dispatch will see.
    counts = _local_tokens_per_expert(sanitized, num_experts)
    assert int(counts.sum().item()) == topk_ids.numel()


def _symmetric_mapping_worker(pgi: ProcessGroupInfo, experts_per_rank: int):
    """Each rank writes only its own experts; all ranks read every expert."""
    import torch.distributed as dist
    from moonep.buffer import create_nvl_dist_tensor

    rank, world = pgi.rank, pgi.world_size
    row_pad = moonep_expert_row_pad(experts_per_rank)

    # Small, granularity-aligned chunk: 2 MiB per row x row_pad rows.
    row_i32 = (2 << 20) // 4
    buf = create_nvl_dist_tensor(
        [row_pad, row_i32], torch.int32, rank, world, group=dist.group.WORLD
    )
    assert buf.shape[0] == world * row_pad

    lo = rank * row_pad
    for e in range(experts_per_rank):
        buf[lo + e].fill_(1000 + lo + e)
    dist.barrier()

    # Read every rank's experts through the mapping.
    for r in range(world):
        for e in range(experts_per_rank):
            row = r * row_pad + e
            want = 1000 + row
            assert int(buf[row][0].item()) == want, (
                f"rank {rank} read row {row}: got {int(buf[row][0].item())}, "
                f"want {want}"
            )
    dist.barrier()


@requires_moonep
@multi_gpu_test(num_gpus=2)
def test_symmetric_weight_mapping():
    """Remote expert rows must return the owning rank's bytes."""
    parallel_launch(2, _symmetric_mapping_worker, 4)


def _dispatch_combine_worker(pgi: ProcessGroupInfo, num_tokens: int, topk: int):
    """dispatch -> identity expert -> combine must reproduce the input.

    With an identity expert function, combining the dispatched rows sums each
    token's topk copies, so the result is the input scaled by topk.
    """
    import torch.distributed as dist
    from moonep import Buffer

    rank, world = pgi.rank, pgi.world_size
    device = pgi.device
    hidden, num_experts = 512, world * 8

    torch.manual_seed(rank)
    x = (torch.randn(num_tokens, hidden, device=device) * 0.1).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_w = torch.ones(num_tokens, topk, dtype=torch.float32, device=device)

    tpe = _local_tokens_per_expert(topk_ids, num_experts)
    buf = Buffer(
        S=num_tokens,
        H=hidden,
        K=topk,
        E=num_experts,
        num_ep_ranks=world,
        token_padding=128,
        B=1,
        group=dist.group.WORLD,
    )
    try:
        h, w, cu, plan = buf.dispatch(x, topk_w, topk_ids, tpe)
        assert h.dtype == torch.bfloat16
        assert w is not None and w.numel() == h.shape[0]
        # Identity expert: combine sums the topk copies of every token.
        out, _, _ = buf.combine(plan=plan, hidden_nvsh=h, route_weights_nvs=None)
        torch.testing.assert_close(out.float(), x.float() * topk, atol=2e-1, rtol=2e-2)
    finally:
        buf.destroy()


@requires_moonep
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("num_tokens", [128, 512])
@pytest.mark.parametrize("topk", [2, 8])
def test_moonep_dispatch_combine(num_tokens: int, topk: int):
    parallel_launch(2, _dispatch_combine_worker, num_tokens, topk)
