# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual fusion of tensor-parallel all-reduce with the following GemmaRMSNorm.

Under tensor parallelism a ``RowParallelLinear`` (e.g. attention ``o_proj``)
produces a per-rank partial sum that is all-reduced, and the result is then fed
into a ``GemmaRMSNorm`` that adds the residual and normalizes. flashinfer ships a
kernel that fuses all-reduce + residual-add + RMSNorm into a single launch; this
helper drives it directly (no torch.compile pass) for models that run eager.

When the platform-specific fast path is not applicable it falls back to
``all_reduce`` + ``GemmaRMSNorm``, which is numerically identical to the
unfused model path.
"""

import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform

logger = init_logger(__name__)

MiB = 1024 * 1024

# flashinfer fused all-reduce + RMSNorm is wired as a registered custom op in
# allreduce_rms_fusion; both that op and the workspace helpers only exist when
# flashinfer.comm.allreduce_fusion is importable.
try:
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        flashinfer_trtllm_fused_allreduce_norm,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        flashinfer_comm,
        get_fi_ar_workspace,
    )

    _AR_RESIDUAL_RMS_NORM = (
        flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm
        if flashinfer_comm is not None
        else None
    )
except ImportError:
    flashinfer_trtllm_fused_allreduce_norm = None  # type: ignore[assignment]
    get_fi_ar_workspace = None  # type: ignore[assignment]
    _AR_RESIDUAL_RMS_NORM = None


_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_MI300X_M3_AITER_MAX_TOKENS = 1536


def _aiter_gemma_shape_is_profitable(
    num_tokens: int,
    hidden_size: int,
    tp_size: int,
    *,
    on_gfx942: bool,
) -> bool:
    # TP8 MiniMax M3 crosses over between 1536 and 2048 tokens on MI300X.
    # Large chunked-prefill batches are faster with NCCL plus the native norm.
    return not (
        on_gfx942
        and tp_size == 8
        and hidden_size == 6144
        and num_tokens > _MI300X_M3_AITER_MAX_TOKENS
    )


def initialize_aiter_fused_allreduce_gemma_rms_norm() -> bool:
    """Initialize the opt-in AITER communicator before graph capture."""
    if not current_platform.is_rocm() or get_tensor_model_parallel_world_size() == 1:
        return False

    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_fused_allreduce_gemma_rmsnorm_enabled():
        return False
    if rocm_aiter_ops.get_aiter_allreduce() is None:
        device_index = torch.accelerator.current_device_index()
        device = torch.device("cuda", 0 if device_index is None else device_index)
        rocm_aiter_ops.initialize_aiter_allreduce(get_tp_group().cpu_group, device)

    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    if aiter_ar is None or aiter_ar.disabled:
        logger.warning_once(
            "AITER fused all-reduce + Gemma RMSNorm was requested but its "
            "communicator could not be initialized; using the unfused path."
        )
        return False
    return True


def _can_use_aiter(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> bool:
    if (
        not current_platform.is_rocm()
        or not hidden_states.is_cuda
        or hidden_states.dtype not in _FI_SUPPORTED_DTYPES
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or residual.shape != hidden_states.shape
        or residual.dtype != hidden_states.dtype
        or not residual.is_contiguous()
        or norm.weight.shape != (hidden_states.shape[-1],)
        or norm.weight.dtype != hidden_states.dtype
        or not norm.weight.is_contiguous()
    ):
        return False

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx942

    if not rocm_aiter_ops.is_fused_allreduce_gemma_rmsnorm_enabled():
        return False
    if not _aiter_gemma_shape_is_profitable(
        hidden_states.shape[0],
        hidden_states.shape[1],
        get_tensor_model_parallel_world_size(),
        on_gfx942=on_gfx942(),
    ):
        return False
    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    return bool(
        aiter_ar is not None
        and not aiter_ar.disabled
        and aiter_ar.should_custom_ar(hidden_states)
    )


def _max_token_num(tp_size: int, hidden_size: int, dtype: torch.dtype) -> int | None:
    """Workspace token budget for flashinfer fused all-reduce, or None if the
    current world size / device is unsupported. Mirrors ``FlashInferAllReduce``."""
    from vllm.config.compilation import PassConfig

    max_size_mb = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(tp_size)
    if not max_size_mb:
        return None
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(max_size_mb * MiB) // (hidden_size * element_size)


def _can_use_flashinfer(hidden_states: torch.Tensor, tp_size: int) -> tuple[bool, int]:
    """Whether the flashinfer fused path applies; returns (ok, max_token_num)."""
    if (
        flashinfer_trtllm_fused_allreduce_norm is None
        or get_fi_ar_workspace is None
        or _AR_RESIDUAL_RMS_NORM is None
    ):
        return False, 0
    if (
        not hidden_states.is_cuda
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.dtype not in _FI_SUPPORTED_DTYPES
    ):
        return False, 0

    num_tokens, hidden_size = hidden_states.shape
    max_token_num = _max_token_num(tp_size, hidden_size, hidden_states.dtype)
    if max_token_num is None or num_tokens > max_token_num:
        return False, 0

    # Lazily create / fetch the (globally cached) workspace; returns None on
    # GPUs without NVSwitch, in which case we fall back gracefully.
    workspace = get_fi_ar_workspace(
        world_size=tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=hidden_states.dtype,
        group=get_tp_group().device_group,
    )
    if workspace is None:
        return False, 0
    return True, max_token_num


def fused_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce ``hidden_states`` + add ``residual`` + GemmaRMSNorm, fused.

    ``hidden_states`` is the per-rank *partial* (un-reduced) output of a
    row-parallel linear; ``norm`` is the GemmaRMSNorm applied right after.
    Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        # No all-reduce needed; identical to the unfused path.
        return norm(hidden_states, residual)

    if _can_use_aiter(hidden_states, residual, norm):
        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.get_fused_allreduce_gemma_rmsnorm_op()(
            input_=hidden_states,
            residual=residual,
            weight=norm.weight,
            epsilon=norm.variance_epsilon,
        )

    ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
    if ok:
        norm_out = torch.empty_like(hidden_states)
        # With norm_out provided, the kernel writes the new residual
        # (all_reduce(hidden_states) + residual) into the hidden_states buffer
        # and the normalized result into norm_out, leaving `residual` untouched.
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=hidden_states,
            residual=residual,
            rms_gamma=norm.weight,
            rms_eps=norm.variance_epsilon,
            world_size=tp_size,
            weight_bias=1.0,  # GemmaRMSNorm-style
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
            pattern_code=_AR_RESIDUAL_RMS_NORM,
            norm_out=norm_out,
        )
        return norm_out, hidden_states

    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
