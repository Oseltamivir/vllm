# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/Finalize for the MoonEP all2all backend.

MoonEP dispatches tokens straight into expert-grouped positions on the
destination rank, so unlike the DeepEP backends there is no permutation left
for the experts kernel to do: ``prepare`` hands back a contiguous ``[NvS, H]``
buffer whose rows are already sorted by expert, plus the per-row expert id.

The layout is described by ``cu_seqlens[E + B]``, an inclusive prefix sum of
padded segment ends. Segment ``g`` occupies rows
``[cu_seqlens[g - 1], cu_seqlens[g])`` and is served by:

* ``g`` itself, for ``g < E`` -- the global expert id, whose weight row lives
  in the symmetric mapping (possibly in another rank's HBM).
* ``plan.experts_to_copy[rank, g - E]``, for ``g >= E`` -- MoonEP's weight
  prefetch slots. We never call ``prefetch_weight``, so instead of reading the
  (unpopulated) slot we resolve the slot back to its global expert id and read
  that expert's real row. ``experts_to_copy`` is a device tensor, so this
  costs one gather and no host synchronization.

Both cases therefore reduce to "expert id per row", which is exactly the
``m_indices`` argument DeepGEMM's contiguous grouped GEMM wants.

MoonEP's combine is an unweighted fp32 accumulation, so the routing weights
must be applied to the expert output before combining; ``dispatch`` scatters
them alongside the tokens as ``route_weights_nvs[NvS]``.
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


@triton.jit
def _moonep_m_indices_kernel(
    cu_seqlens_ptr,  # [E + B] int32, inclusive padded segment ends
    slot_experts_ptr,  # [B] int32, global expert id per prefetch slot (-1 empty)
    m_indices_ptr,  # [NvS] int32, written
    num_experts: tl.constexpr,
    nvs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Expand ``cu_seqlens`` into a per-row expert id.

    One program per segment. Rows past the last segment keep the ``-1`` the
    caller pre-filled, which DeepGEMM treats as a block to skip.
    """
    g = tl.program_id(0)

    start = tl.where(g == 0, 0, tl.load(cu_seqlens_ptr + g - 1, mask=g > 0, other=0))
    end = tl.load(cu_seqlens_ptr + g)

    # Prefetch-slot segments carry the global id of the expert they stand in
    # for; a slot the planner left unused holds -1 and is always empty.
    expert_id = tl.where(
        g < num_experts,
        g,
        tl.load(
            slot_experts_ptr + (g - num_experts),
            mask=g >= num_experts,
            other=-1,
        ),
    )

    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        tl.store(m_indices_ptr + idx, expert_id, mask=(idx < end) & (idx < nvs))


def _build_m_indices(
    cu_seqlens: torch.Tensor,
    experts_to_copy_local: torch.Tensor,
    num_experts: int,
    nvs: int,
) -> torch.Tensor:
    """Build the ``[NvS]`` int32 per-row expert id for DeepGEMM."""
    m_indices = torch.full((nvs,), -1, dtype=torch.int32, device=cu_seqlens.device)
    num_groups = cu_seqlens.numel()
    _moonep_m_indices_kernel[(num_groups,)](
        cu_seqlens,
        experts_to_copy_local,
        m_indices,
        num_experts=num_experts,
        nvs=nvs,
        BLOCK=256,
    )
    return m_indices


def _local_tokens_per_expert(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Histogram this rank's (token, k) pairs over the global expert space.

    ``torch.bincount`` would synchronize to size its output, so scatter into a
    pre-sized buffer instead. Negative ids (padded/invalid routing slots) are
    dropped.
    """
    flat = topk_ids.flatten()
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    # Fold invalid ids into a trailing bucket that is then discarded.
    safe = torch.where(flat < 0, num_experts, flat).to(torch.int64)
    counts.scatter_add_(0, safe, torch.ones_like(safe, dtype=torch.int32))
    return counts[:num_experts].contiguous()


class MoonEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using MoonEP's load-balanced NVLink dispatch."""

    def __init__(
        self,
        buffer,
        num_dispatchers: int,
        dp_size: int,
        rank: int,
        num_experts: int,
        num_topk: int,
        max_num_tokens: int,
        token_padding: int,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank = rank
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.max_num_tokens = max_num_tokens
        self.token_padding = token_padding

        # dispatch returns a plan that combine needs back, plus the routing
        # weights in dispatched order. MoonEP's combine does not apply them.
        self._plan = None
        self._route_weights_nvs: torch.Tensor | None = None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        # combine sums each token's topk contributions across all EP ranks.
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        # Standard (non-batched) format.
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        # MoonEP's planning kernel consumes int32 expert ids.
        return torch.int32

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        assert expert_map is None, (
            "MoonEP addresses experts in the global expert space through its "
            "symmetric weight mapping and does not use an expert_map."
        )
        assert a1.dtype == torch.bfloat16, (
            f"MoonEP dispatches bf16 activations, got {a1.dtype}."
        )

        tokens_per_expert = _local_tokens_per_expert(topk_ids, num_experts)

        # MoonEP always moves bf16; activations are quantized after dispatch.
        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = self.buffer.dispatch(
            a1,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            tokens_per_expert,
        )

        self._plan = plan
        self._route_weights_nvs = route_weights_nvs

        return lambda: self._receiver(
            hidden_nvsh=hidden_nvsh,
            route_weights_nvs=route_weights_nvs,
            cu_seqlens=cu_seqlens,
            plan=plan,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        plan,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        nvs = hidden_nvsh.size(0)

        m_indices = _build_m_indices(
            cu_seqlens=cu_seqlens,
            experts_to_copy_local=plan.experts_to_copy[self.rank],
            num_experts=self.num_experts,
            nvs=nvs,
        )

        expert_x = hidden_nvsh
        expert_x_scale = None
        if not defer_input_quant:
            expert_x, expert_x_scale = moe_kernel_quantize_input(
                hidden_nvsh,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
            )

        # In dispatched space each row belongs to exactly one expert with
        # exactly one routing weight, so the topk dimension is 1. The experts
        # kernel reads column 0 as DeepGEMM's m_indices.
        expert_topk_ids = m_indices.view(nvs, 1)
        expert_topk_weights = route_weights_nvs.view(nvs, 1)

        # Segment lengths are only known on device and the buffer is sized for
        # the worst case, so there is no exact per-expert count to report.
        # The experts kernel sizes its workspaces from NvS instead, which is
        # static and therefore cudagraph friendly.
        return (expert_x, expert_x_scale, None, expert_topk_ids, expert_topk_weights)

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert self._plan is not None, "finalize called before prepare"
        # The routing weights are applied below, before MoonEP's unweighted
        # combine. An experts kernel that already applied them (anything other
        # than a no-op reduction) would make that a double application, so
        # reject the pairing rather than silently returning wrong numbers.
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP), (
            "MoonEP applies routing weights during finalize and expects the "
            "experts kernel to leave them unapplied, but it requested "
            f"{type(weight_and_reduce_impl).__name__}."
        )
        assert fused_expert_output.dtype == torch.bfloat16, (
            "MoonEP combine accumulates bf16 expert output, got "
            f"{fused_expert_output.dtype}."
        )

        route_weights_nvs = self._route_weights_nvs
        assert route_weights_nvs is not None

        # MoonEP's combine is an unweighted fp32 accumulation of each token's
        # topk contributions, so scale by the routing weight here. When the
        # weight was already folded into the input we must not apply it twice.
        if not apply_router_weight_on_input:
            fused_expert_output = fused_expert_output * route_weights_nvs.unsqueeze(
                1
            ).to(fused_expert_output.dtype)

        combined, _, _ = self.buffer.combine(
            plan=self._plan,
            hidden_nvsh=fused_expert_output.contiguous(),
            route_weights_nvs=None,
        )
        output.copy_(combined[: output.size(0)], non_blocking=True)

        self._plan = None
        self._route_weights_nvs = None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        # MoonEP's dispatch/combine are already device-side and asynchronous
        # with respect to the host; running them eagerly and handing back a
        # trivial receiver is enough to unlock shared-expert overlap.
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        return lambda: None
