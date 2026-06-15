# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 (1x32 block, E8M0 scale) MoE experts on Triton.

``Mxfp8TritonExpertsBase`` stashes E8M0 weight scales for checkpoint layout.
``Mxfp8EmulationTritonExperts`` dequantizes to BF16 and runs ``TritonExperts``
for devices without a fused MXFP8 MoE kernel.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_gated_triton_kernel,
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl

logger = init_logger(__name__)

_MINIMAX_M3_MI300X_EP_BF16_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 2,
}


def _is_minimax_m3_mi300x_ep8(moe_config: FusedMoEConfig) -> bool:
    """Match the profiled MiniMax-M3 EP8 shape on gfx94x."""
    return (
        current_platform.is_fp8_fnuz()
        and moe_config.ep_size == 8
        and moe_config.has_shared_experts
        and moe_config.num_experts == 128
        and moe_config.experts_per_token == 4
        and moe_config.hidden_dim == 6144
        and moe_config.intermediate_size == 3072
        and moe_config.max_model_len > 0
    )


class Mxfp8TritonExpertsBase(TritonExperts):
    """Shared MXFP8 MoE setup: stash E8M0 scales, clear scales on ``quant_config``."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation) -> bool:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return True
        return TritonExperts._supports_activation(activation)


class Mxfp8EmulationTritonExperts(Mxfp8TritonExpertsBase):
    """Dequantize MXFP8 weights to BF16 on the fly and run ``TritonExperts``."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.use_sparse_mi300x_ep = _is_minimax_m3_mi300x_ep8(moe_config)
        logger.warning_once(
            "Using Mxfp8EmulationTritonExperts MoE backend. Weights are "
            "dequantized to BF16 on the fly; this is slower than a native "
            "MXFP8 MoE kernel and is intended for devices without one."
        )

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        # BF16 fallback: do not MXFP8-quantize activations in ``TritonExperts``.
        return None

    @property
    def block_shape(self) -> list[int] | None:
        return None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    def activation(
        self,
        activation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ):
        """Apply GEMM1 activation with quant-config alpha/beta/clamp."""
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
            apply_moe_activation,
        )

        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            limit = self.quant_config.gemm1_clamp_limit
            if limit is None:
                raise ValueError("SWIGLUOAI_UNINTERLEAVE requires gemm1_clamp_limit")
            alpha = self.quant_config.gemm1_alpha
            alpha = 1.702 if alpha is None else float(alpha)
            beta = self.quant_config.gemm1_beta
            beta = 1.0 if beta is None else float(beta)
            apply_moe_activation(
                activation,
                output,
                input,
                clamp_limit=float(limit),
                alpha=alpha,
                beta=beta,
            )
            return
        super().activation(activation, output, input)

    def _apply_sparse_mi300x_ep(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> None:
        """Run only local EP routes with a fused BF16 GEMM1 SwiGLU epilogue."""
        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = expert_map.numel()
        config = _MINIMAX_M3_MI300X_EP_BF16_CONFIG
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                ignore_invalid_experts=True,
                num_local_experts=E,
            )
        )
        assert sorted_token_ids is not None

        activation_dim = N // 2
        intermediate_activation = _resize_cache(
            workspace13,
            (num_tokens * top_k_num, activation_dim),
        )
        intermediate_output = _resize_cache(
            workspace2,
            (num_tokens, top_k_num, K),
        )

        alpha = self.quant_config.gemm1_alpha
        alpha = 1.702 if alpha is None else float(alpha)
        beta = self.quant_config.gemm1_beta
        beta = 1.0 if beta is None else float(beta)
        limit = self.quant_config.gemm1_clamp_limit
        limit = None if limit is None else float(limit)

        invoke_fused_moe_gated_triton_kernel(
            hidden_states,
            w1,
            intermediate_activation,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k_num,
            config,
            alpha,
            beta,
            limit,
        )
        invoke_fused_moe_triton_kernel(
            intermediate_activation,
            w2,
            intermediate_output,
            None,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        moe_fused_mul_sum(
            intermediate_output,
            topk_weights,
            outputs=output,
            topk_ids=topk_ids,
            expert_map=expert_map,
            apply_weights=False,
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        # If the weights were already dequantized to BF16 at load time
        # (process_weights_after_loading on devices without a native MXFP8 MoE
        # kernel), use them directly -- no per-step dequant. MXFP8 weights are
        # 1-byte FP8 (element_size 1); BF16/FP16 are >= 2 bytes.
        if w1.element_size() >= 2:
            # tl.dot requires w and activations share a dtype; .to() is a no-op
            # when they already match (e.g. both BF16).
            w1_bf16 = w1.to(hidden_states.dtype)
            w2_bf16 = w2.to(hidden_states.dtype)
        else:
            w1_bf16 = dequant_mxfp8_to_bf16(w1, self.w1_scale_val).to(
                hidden_states.dtype
            )
            w2_bf16 = dequant_mxfp8_to_bf16(w2, self.w2_scale_val).to(
                hidden_states.dtype
            )

        use_sparse_ep = (
            self.use_sparse_mi300x_ep
            and hidden_states.dtype == torch.bfloat16
            and activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
            and expert_map is not None
            and not apply_router_weight_on_input
            and getattr(self, "_lora_context", None) is None
        )
        if use_sparse_ep:
            self._apply_sparse_mi300x_ep(
                output=output,
                hidden_states=hidden_states,
                w1=w1_bf16,
                w2=w2_bf16,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                workspace13=workspace13,
                workspace2=workspace2,
            )
            return

        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_bf16,
            w2=w2_bf16,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
