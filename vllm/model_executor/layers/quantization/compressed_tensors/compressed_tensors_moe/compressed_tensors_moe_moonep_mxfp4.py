# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 MoE weights held in MoonEP symmetric memory.

The ``moonep`` all2all backend balances token load by reassigning an
overloaded expert owner's surplus onto underloaded ranks, so a rank routinely
has to compute experts it does not own. Those experts' weights are reached by
mapping every rank's shard into one contiguous virtual address range with
CUDA VMM, which makes ``w[e]`` valid for any global expert id -- backed by
local HBM when the rank owns ``e`` and by a peer's HBM over NVLink otherwise.

Each rank still allocates and owns exactly ``E / R`` experts, so the resident
footprint is unchanged; only the address space is shared.

Two layout constraints drive the code below:

* A VMM chunk must be an exact multiple of the allocation granularity. The
  FP4 payloads happen to be exact at Kimi-K3's shapes but the
  DeepGEMM-transformed scales are not, and DeepGEMM rejects a padded scale
  group stride outright (``sf.stride(-3) == sf.stride(-1) * sf.size(-1)``).
  So the *expert* extent absorbs the alignment: each rank reserves
  ``expert_row_pad(E/R)`` rows of which the first ``E/R`` are real.
  Rows stay tightly packed, and the prepare step emits buffer rows rather
  than bare expert ids so the padding is invisible to the GEMM.
* DeepGEMM's scale transform is per-expert independent (verified on B300:
  transforming the whole group equals stacking per-expert transforms), which
  is what lets a remote expert's scales be addressed by row at all.
"""

import torch
import torch.distributed as dist

from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.experts.moonep_deep_gemm_moe import (
    MoonEPDeepGemmFP4Experts,
)
from vllm.model_executor.layers.fused_moe.moonep_weights import (
    alloc_symmetric,
    alloc_symmetric_uint8,
    expert_row_pad,
    vmm_granularity,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_weight_scale_block,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

# MXFP4 scale group along the reduction dim.
_MXFP4_GROUP = 32


class MoonEPCompressedTensorsMxfp4MoEMethod(CompressedTensorsW4A4Mxfp4MoEMethod):
    """compressed-tensors MXFP4 MoE backed by MoonEP symmetric memory."""

    def __init__(self, moe):
        super().__init__(moe)
        # The parent picks CUTLASS/Marlin from device support alone. MoonEP
        # pairs with the DeepGEMM FP8xFP4 grouped GEMM, which is also the only
        # backend whose activation layout matches a MoonEP-dispatched buffer.
        self.use_cutlass_mxfp4 = False
        self.mxfp4_backend = Mxfp4MoeBackend.DEEPGEMM_MXFP4
        self.experts_cls = MoonEPDeepGemmFP4Experts

        ep = get_ep_group()
        self.ep_rank = ep.rank_in_group
        self.ep_size = ep.world_size
        self.ep_device_group = ep.device_group

        # Full [E_global, ...] mappings, bound onto the layer once loading is
        # complete. Keyed by the final (post-rename) parameter name.
        self._symmetric: dict[str, torch.Tensor] = {}

        logger.info_once(
            "Using MoonEP MXFP4 MoE method: expert weights in NVLink "
            "symmetric memory, DeepGEMM FP8xFP4 grouped GEMM, EP size %d.",
            self.ep_size,
        )

    def _own_slice(self, full: torch.Tensor, num_local_experts: int) -> torch.Tensor:
        """This rank's real experts inside its padded row block."""
        lo = self.ep_rank * expert_row_pad(num_local_experts)
        return full[lo : lo + num_local_experts]

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # `num_experts` here is this rank's shard (E / R), not the global E.
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        rank, world, group = self.ep_rank, self.ep_size, self.ep_device_group
        n13 = 2 * intermediate_size_per_partition
        # Rows reserved per rank; only the first `num_experts` are real.
        e_pad = expert_row_pad(num_experts)

        # FP4 payloads: two values per byte along the reduction dim.
        w13_full = alloc_symmetric_uint8(
            [e_pad, n13, hidden_size // 2], rank, world, group
        )
        w2_full = alloc_symmetric_uint8(
            [e_pad, hidden_size, intermediate_size_per_partition // 2],
            rank,
            world,
            group,
        )
        self._symmetric["w13_weight"] = w13_full
        self._symmetric["w2_weight"] = w2_full

        # The checkpoint loader writes `param.data[local_expert_id]`, so the
        # registered Parameter is this rank's slice of the mapping. Writes
        # land in the rank's own physical pages; the full view is bound in
        # process_weights_after_loading.
        w13_weight = torch.nn.Parameter(
            self._own_slice(w13_full, num_experts), requires_grad=False
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            self._own_slice(w2_full, num_experts), requires_grad=False
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Raw e8m0 scales stay in ordinary memory: they are consumed by
        # DeepGEMM's transform during post-load and never read remotely in
        # this form. The transformed result is what gets a symmetric buffer.
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                n13,
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def _symmetrize_scales(
        self, local_transformed: torch.Tensor, num_local_experts: int
    ) -> torch.Tensor:
        """Copy per-rank transformed scales into a symmetric buffer.

        ``local_transformed`` is ``[E_local, mn, k]`` int32 laid out MN-major,
        i.e. per expert the memory is ``[k, mn]`` contiguous. Groups stay
        tightly packed -- DeepGEMM asserts
        ``sf.stride(-3) == sf.stride(-1) * sf.size(-1)`` -- so alignment is
        absorbed by reserving padded expert rows instead.
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

        buf = alloc_symmetric(
            [e_pad, k, mn],
            torch.int32,
            self.ep_rank,
            self.ep_size,
            self.ep_device_group,
        )
        # local_transformed is (mn, k) per expert with stride (1, mn), so its
        # backing memory is (k, mn); copy that directly into the real rows.
        self._own_slice(buf, e_local).copy_(local_transformed.permute(0, 2, 1))

        # (E_pad_global, mn, k) with the tight group stride k*mn DeepGEMM wants.
        return buf.permute(0, 2, 1)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        num_local_experts = layer.num_experts

        # Every rank must have finished writing its own shard before any rank
        # reads a peer's rows through the mapping.
        if dist.is_initialized():
            dist.barrier(group=self.ep_device_group)

        # Bind the full mappings under the names the experts kernel uses. The
        # storage is unchanged; only the visible expert extent grows from the
        # local shard to the global space, which is what makes m_indices
        # global expert ids valid.
        layer.w13_weight = torch.nn.Parameter(
            self._symmetric["w13_weight"], requires_grad=False
        )
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = torch.nn.Parameter(
            self._symmetric["w2_weight"], requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Transform this rank's raw e8m0 scales into DeepGEMM's packed
        # UE8M0 layout, then publish them symmetrically. The transform is
        # per-expert independent, so a peer's rows stay addressable by id.
        hidden_size = layer.w2_weight.shape[1]
        intermediate = layer.w2_weight.shape[2] * 2
        n13 = layer.w13_weight.shape[1]

        w13_local = deepgemm_post_process_weight_scale_block(
            ws=layer.w13_weight_scale.data,
            mn=n13,
            k=hidden_size,
            quant_block_shape=(1, _MXFP4_GROUP),
            num_groups=num_local_experts,
        )
        w2_local = deepgemm_post_process_weight_scale_block(
            ws=layer.w2_weight_scale.data,
            mn=hidden_size,
            k=intermediate,
            quant_block_shape=(1, _MXFP4_GROUP),
            num_groups=num_local_experts,
        )

        w13_scale_full = self._symmetrize_scales(w13_local, num_local_experts)
        w2_scale_full = self._symmetrize_scales(w2_local, num_local_experts)

        if dist.is_initialized():
            dist.barrier(group=self.ep_device_group)

        layer.w13_weight_scale = torch.nn.Parameter(w13_scale_full, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale_full, requires_grad=False)

        self.moe_quant_config = make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            layer=layer,
        )
        assert self.moe_quant_config is not None
        self.moe_kernel = make_mxfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            mxfp4_backend=self.mxfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
