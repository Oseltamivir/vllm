# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM MXFP4 experts over a MoonEP-dispatched activation buffer."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmFP4Experts,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    mk_alignment_scope,
)


class MoonEPDeepGemmFP4Experts(DeepGemmFP4Experts):
    """MXFP4 experts for the ``moonep`` all2all backend.

    Numerically identical to :class:`DeepGemmFP4Experts` -- the same pair of
    ``m_grouped_fp8_fp4_gemm_nt_contiguous`` calls with the same recipes --
    but without the permute/unpermute pair around them, because MoonEP
    already delivers what those two functions exist to produce:

    * ``deepgemm_moe_permute`` sorts tokens into per-expert contiguous
      segments and builds ``m_indices``. MoonEP's dispatch writes tokens
      directly into their expert-grouped destination, and the prepare step
      hands the per-row expert id over in ``topk_ids[:, 0]``.
    * ``deepgemm_unpermute_and_reduce`` scatters rows back to token order,
      applying the routing weights while reducing. MoonEP's combine performs
      that reduction across ranks, and the routing weights are applied in the
      prepare/finalize object just before it.

    Consequently ``apply`` writes its second GEMM straight to ``output`` and
    reports :class:`TopKWeightAndReduceNoOP` (inherited), leaving weighting
    and reduction to finalize.

    The expert ids are global: ``w1``/``w2`` are the full ``[E, ...]``
    symmetric-memory mappings, so a segment whose expert is owned by another
    rank is served by reading that rank's HBM over NVLink.
    """

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_moonep_kernels

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        # Every per-expert segment MoonEP produces is padded up to this, so
        # segment starts are multiples of it. DeepGEMM reads m_indices once
        # per BLOCK_M rows, so BLOCK_M must not exceed it. The all2all factory
        # sizes the MoonEP buffer from the same source, so the two agree by
        # construction.
        self.token_padding = get_mk_alignment_for_contiguous_layout()[0]

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # M is MoonEP's NvS: the buffer is already expert-grouped and padded,
        # and its size is fixed by (S, K, E, R, token_padding) rather than by
        # the routing of any particular step. No alignment maths and no
        # dependence on expert_tokens_meta, which keeps this cudagraph-safe.
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, max(activation_out_dim, K))
        workspace2 = (M, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        assert expert_map is None, (
            "MoonEP addresses experts globally; expert_map must be None."
        )

        a1q = hidden_states
        _, N, _ = w1.size()
        # K comes from the activations: w1 is FP4 packed as (E, N, K//2).
        K = a1q.size(1)
        M_sum = a1q.size(0)
        # FC2 writes straight into output, so it must already be the full
        # dispatched buffer rather than the per-token output shape.
        assert output.shape == (M_sum, K), (
            f"expected output {(M_sum, K)} for the MoonEP dispatched buffer, "
            f"got {tuple(output.shape)}."
        )

        # prepare() put the per-row expert id here; in dispatched space each
        # row belongs to exactly one expert, so topk == 1. Rows past the last
        # segment are -1, which DeepGEMM skips.
        assert topk_ids.size(1) == 1, (
            "MoonEP-dispatched activations carry one expert per row, got "
            f"topk={topk_ids.size(1)}."
        )
        expert_ids = topk_ids[:, 0].contiguous()

        with mk_alignment_scope(self.token_padding):
            # FC1: FP8 activations x FP4 weights.
            mm1_out = _resize_cache(workspace2, (M_sum, N))
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a1q, a1q_scale),
                (w1.view(torch.int8), self.w1_scale),
                mm1_out,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )

            # Gated activation + FP8 requant.
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn),
                (M_sum, activation_out_dim),
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            # FC2 straight into the output buffer: no unpermute, and the
            # routing weights and cross-rank reduction are finalize's job.
            m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a2q, a2q_scale),
                (w2.view(torch.int8), self.w2_scale),
                output,
                expert_ids,
                recipe_a=(1, self._ACT_BLOCK_K),
                recipe_b=(1, self._WEIGHT_BLOCK_K),
            )
