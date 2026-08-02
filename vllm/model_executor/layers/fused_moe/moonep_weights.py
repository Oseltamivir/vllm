# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 expert weights in MoonEP symmetric memory.

MoonEP balances token load by reassigning an overloaded expert owner's surplus
onto underloaded ranks, so a rank routinely computes experts it does not own.
Those experts are reached by mapping every rank's shard into one contiguous
virtual address range with CUDA VMM, which makes ``w[row]`` valid for any
global expert -- local HBM when the rank owns it, a peer's HBM over NVLink
otherwise. Each rank still allocates only its own ``E / R`` experts, so the
resident footprint is unchanged; only the address space is shared.

Two layout constraints shape everything here:

* A VMM chunk must be an exact multiple of the allocation granularity, and
  DeepGEMM separately rejects a padded scale group stride
  (``sf.stride(-3) == sf.stride(-1) * sf.size(-1)``). So the *expert* extent
  absorbs the alignment: each rank reserves :func:`expert_row_pad` rows of
  which the first ``E / R`` are real. Groups stay tightly packed, and the
  prepare step emits buffer rows rather than bare expert ids, so the padding
  never reaches the GEMM.
* DeepGEMM's weight-scale transform is per-expert independent (verified on
  B300: transforming a whole group equals stacking per-expert transforms),
  which is what lets a remote expert's scales be addressed by row at all.

These helpers are shared by every MXFP4 MoE method, since which checkpoint
format a model uses is independent of whether it runs on the moonep backend.
"""

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# Experts per rank are padded up to a multiple of this. With a 2 MiB VMM
# granularity, 128 rows align any tensor whose per-expert size is a multiple
# of 16 KiB, which covers every MXFP4 payload and transformed-scale shape.
EXPERT_ROW_PAD = 128


def expert_row_pad(num_local_experts: int) -> int:
    """Rows reserved per rank in the symmetric expert buffers."""
    return round_up(num_local_experts, EXPERT_ROW_PAD)


def vmm_granularity() -> int:
    from moonep._C import get_vmm_granularity  # type: ignore[import-not-found]

    return get_vmm_granularity()


def _alloc_symmetric(
    chunk_shape: list[int],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    group,
) -> torch.Tensor:
    """Map one ``chunk_shape`` per rank into a single ``[R*chunk0, ...]`` VA.

    ``create_nvl_dist_tensor`` silently pads dim 0 when a chunk is not
    granularity aligned, which would break the "row == expert" invariant every
    caller depends on and produce wrong numbers rather than an error. Assert
    instead.
    """
    from moonep.buffer import (  # type: ignore[import-not-found]
        create_nvl_dist_tensor,
    )

    nbytes = dtype.itemsize
    for d in chunk_shape:
        nbytes *= d
    gran = vmm_granularity()
    assert nbytes % gran == 0, (
        f"MoonEP symmetric chunk {chunk_shape} of {dtype} is {nbytes} bytes, "
        f"not a multiple of the {gran}-byte VMM granularity; dim 0 would be "
        f"padded and global expert indexing would break."
    )

    full = create_nvl_dist_tensor(
        list(chunk_shape), dtype, rank, world_size, group=group
    )
    assert full.shape[0] == world_size * chunk_shape[0]
    return full


def alloc_symmetric_uint8(
    chunk_shape: list[int], rank: int, world_size: int, group
) -> torch.Tensor:
    """Allocate a uint8 symmetric tensor.

    MoonEP sizes chunks from a small dtype table that omits uint8, so allocate
    int32 with a quarter-width last dim and reinterpret. Byte layout and
    alignment are identical, which keeps this free of any MoonEP-side change.
    """
    assert chunk_shape[-1] % 4 == 0, (
        f"last dim {chunk_shape[-1]} must be divisible by 4 to alias int32"
    )
    i32_shape = list(chunk_shape[:-1]) + [chunk_shape[-1] // 4]
    return _alloc_symmetric(i32_shape, torch.int32, rank, world_size, group).view(
        torch.uint8
    )


class MoonEPExpertWeights:
    """Owns the symmetric MXFP4 weight mappings for one MoE layer."""

    def __init__(self, ep_rank: int, ep_size: int, ep_device_group):
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ep_device_group = ep_device_group
        self.w13: torch.Tensor | None = None
        self.w2: torch.Tensor | None = None

    def own_slice(self, full: torch.Tensor, num_local_experts: int) -> torch.Tensor:
        """This rank's real experts inside its padded row block."""
        lo = self.ep_rank * expert_row_pad(num_local_experts)
        return full[lo : lo + num_local_experts]

    def create_payloads(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate the FP4 payload mappings; returns this rank's slices.

        The checkpoint loader writes ``param.data[local_expert_id]``, so the
        registered Parameter must be the rank's slice. The full mappings are
        bound onto the layer once loading finishes.
        """
        e_pad = expert_row_pad(num_local_experts)
        args = (self.ep_rank, self.ep_size, self.ep_device_group)

        self.w13 = alloc_symmetric_uint8(
            [e_pad, 2 * intermediate_size, hidden_size // 2], *args
        )
        self.w2 = alloc_symmetric_uint8(
            [e_pad, hidden_size, intermediate_size // 2], *args
        )
        logger.info_once(
            "MoonEP symmetric expert weights: w13 %s, w2 %s (%d real experts "
            "per rank in %d padded rows).",
            tuple(self.w13.shape),
            tuple(self.w2.shape),
            num_local_experts,
            e_pad,
        )
        return (
            self.own_slice(self.w13, num_local_experts),
            self.own_slice(self.w2, num_local_experts),
        )

    def _symmetrize_scales(
        self, local_transformed: torch.Tensor, num_local_experts: int
    ) -> torch.Tensor:
        """Publish per-rank transformed scales into a symmetric buffer.

        ``local_transformed`` is ``[E_local, mn, k]`` int32 laid out MN-major,
        i.e. per expert the backing memory is ``[k, mn]`` contiguous. Groups
        stay tightly packed; only the expert extent is padded.
        """
        e_local, mn, k = local_transformed.shape
        assert local_transformed.dtype == torch.int32
        # The permute round trip below assumes the backend transform returned
        # MN-major, tightly packed scales, i.e. per expert the memory is
        # [k, mn] contiguous. If it were K-major, or MN-major with a
        # TMA-padded mn stride, the round trip would silently transpose or
        # drop the padding -- and DeepGEMM's own
        # stride(-3) == stride(-1)*size(-1) check passes either way, so it
        # would not catch it. Assert the premise instead.
        assert local_transformed.stride() == (mn * k, 1, mn), (
            f"expected MN-major tight scales with stride {(mn * k, 1, mn)}, "
            f"got {local_transformed.stride()} for shape {(e_local, mn, k)}."
        )
        assert e_local == num_local_experts

        e_pad = expert_row_pad(e_local)
        gran = vmm_granularity()
        per_expert = k * mn * local_transformed.element_size()
        assert (e_pad * per_expert) % gran == 0, (
            f"scale chunk {e_pad}x{per_expert} bytes is not a multiple of the "
            f"{gran}-byte VMM granularity"
        )

        buf = _alloc_symmetric(
            [e_pad, k, mn],
            torch.int32,
            self.ep_rank,
            self.ep_size,
            self.ep_device_group,
        )
        self.own_slice(buf, e_local).copy_(local_transformed.permute(0, 2, 1))
        # (E_pad_global, mn, k) with the tight group stride k*mn DeepGEMM wants.
        return buf.permute(0, 2, 1)

    def publish_converted(
        self,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        num_local_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Globalize the weights after vLLM's own backend conversion.

        Called with the scales that ``convert_weight_to_mxfp4_moe_kernel_format``
        already transformed into DeepGEMM's packed UE8M0 layout for this rank's
        shard -- that transform is per-expert independent, so publishing its
        output symmetrically keeps every expert addressable by row.

        The FP4 payloads were allocated symmetrically up front and the
        conversion passes them through untouched, so they only need swapping
        from this rank's slice back to the global mapping.

        Returns ``(w13, w2, w13_scale, w2_scale)`` in the global expert space.
        """
        assert self.w13 is not None and self.w2 is not None

        # No rank may read a peer's rows until every rank has written its own.
        if dist.is_initialized():
            dist.barrier(group=self.ep_device_group)

        w13_scale_full = self._symmetrize_scales(w13_scale, num_local_experts)
        w2_scale_full = self._symmetrize_scales(w2_scale, num_local_experts)

        if dist.is_initialized():
            dist.barrier(group=self.ep_device_group)

        return self.w13, self.w2, w13_scale_full, w2_scale_full
