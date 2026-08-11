# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.nvfp4.lut_b import (
    LUT_B_BLOCK_K,
    LUT_B_BLOCK_N,
    dequantize_lut_b,
    quantize_lut_b,
    quantize_lut_b_calibration_free,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter


def _fake_quantize_moe_weight_lut_b(
    weight: torch.Tensor, algorithm: str | None
) -> None:
    """Quantize stacked MoE expert weights ``(E, N, K)`` to LUT-B and
    reconstruct them in place in the original dtype.

    LUT-B tiles are 8x64 and never cross expert boundaries when the expert
    dimension is folded into N, so all experts are fit in a single call.
    """
    assert weight.dim() == 3, f"expected 3D expert weights, got {weight.shape}"
    num_experts, n, k = weight.shape
    if n % LUT_B_BLOCK_N != 0 or k % LUT_B_BLOCK_K != 0:
        raise ValueError(
            "LUT-B MoE requires sharded expert weights divisible by "
            f"({LUT_B_BLOCK_N}, {LUT_B_BLOCK_K}), got ({n}, {k}); adjust "
            "tensor parallelism so the sharded dims stay divisible"
        )
    flat = weight.reshape(num_experts * n, k)
    if algorithm is None:
        packed, codebooks = quantize_lut_b(flat)
        reconstructed = dequantize_lut_b(packed, codebooks, out_dtype=weight.dtype)
    else:
        (
            packed,
            codebooks,
            output_scale,
            residual_position,
            residual_value,
        ) = quantize_lut_b_calibration_free(flat, algorithm=algorithm)
        reconstructed = dequantize_lut_b(
            packed,
            codebooks,
            out_dtype=weight.dtype,
            output_scale=output_scale,
            residual_position=residual_position,
            residual_value=residual_value,
        )
    flat.copy_(reconstructed)


class LutBOnlineMoEMethod(UnquantizedFusedMoEMethod):
    """Online calibration-free LUT-B fake quantization for MoE expert weights.

    Fits LUT-B (3-bit indices into per-8x64-tile E4M3 codebooks) to each
    expert weight at load time and reconstructs the weights in place in the
    model dtype. E4M3 codebook entries are exactly representable in bf16, so
    the reconstructed weights are value-identical to what a Rubin LUT-B MMA
    would consume, while serving runs the standard unquantized fused-MoE
    kernels.
    """

    def __init__(
        self, *, layer: torch.nn.Module, algorithm: str | None = None
    ) -> None:
        super().__init__(layer.moe_config)
        self.algorithm = algorithm

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_lut_b_fake_quantized", False):
            _fake_quantize_moe_weight_lut_b(layer.w13_weight.data, self.algorithm)
            _fake_quantize_moe_weight_lut_b(layer.w2_weight.data, self.algorithm)
            layer._lut_b_fake_quantized = True
        super().process_weights_after_loading(layer)


class LutBOnlineLinearMethod(LinearMethodBase):
    """Online calibration-free LUT-B weight quantization."""

    uses_meta_device: bool = True

    def __init__(self, algorithm: str | None = None) -> None:
        self.algorithm = algorithm

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % LUT_B_BLOCK_N != 0:
            raise ValueError(
                "The sharded LUT-B output width must be divisible by "
                f"{LUT_B_BLOCK_N}, got {output_size_per_partition}"
            )
        if input_size_per_partition % LUT_B_BLOCK_K != 0:
            raise ValueError(
                "The sharded LUT-B input width must be divisible by "
                f"{LUT_B_BLOCK_K}, got {input_size_per_partition}"
            )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if self.algorithm is None:
            packed, codebooks = quantize_lut_b(layer.weight)
            output_scale = None
            residual_position = None
            residual_value = None
        else:
            (
                packed,
                codebooks,
                output_scale,
                residual_position,
                residual_value,
            ) = quantize_lut_b_calibration_free(
                layer.weight,
                algorithm=self.algorithm,
            )
        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_codebook", codebooks)
        if output_scale is not None:
            replace_parameter(layer, "weight_output_scale", output_scale)
        if residual_position is not None and residual_value is not None:
            replace_parameter(layer, "weight_residual_position", residual_position)
            replace_parameter(layer, "weight_residual_value", residual_value)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = dequantize_lut_b(
            layer.weight,
            layer.weight_codebook,
            out_dtype=x.dtype,
            output_scale=getattr(layer, "weight_output_scale", None),
            residual_position=getattr(layer, "weight_residual_position", None),
            residual_value=getattr(layer, "weight_residual_value", None),
        )
        return F.linear(x, weight, bias)
