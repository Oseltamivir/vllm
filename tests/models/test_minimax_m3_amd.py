# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from torch import nn

import vllm.models.minimax_m3.amd.model as minimax_m3


@pytest.mark.parametrize("is_sequence_parallel", [False, True])
def test_mlp_parallelism_mode(monkeypatch, is_sequence_parallel: bool) -> None:
    linear_kwargs: list[dict] = []

    class FakeLinear(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            linear_kwargs.append(kwargs)

    monkeypatch.setattr(minimax_m3, "MergedColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(minimax_m3, "RowParallelLinear", FakeLinear)

    config = SimpleNamespace(
        hidden_size=16,
        hidden_act="swigluoai",
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=7.0,
    )
    minimax_m3.MiniMaxM3MLP(
        config=config,
        intermediate_size=32,
        is_sequence_parallel=is_sequence_parallel,
    )

    assert len(linear_kwargs) == 2
    assert all(kwargs["disable_tp"] is is_sequence_parallel for kwargs in linear_kwargs)


def test_moe_enables_sequence_parallel_for_hybrid_tp_dp(monkeypatch) -> None:
    fused_moe_kwargs: dict = {}
    shared_mlp_kwargs: dict = {}

    class FakeGate(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.out_dtype = kwargs["out_dtype"]

    class FakeSharedMLP(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            shared_mlp_kwargs.update(kwargs)

    class FakeExperts(nn.Module):
        pass

    def fake_fused_moe(*args, **kwargs):
        fused_moe_kwargs.update(kwargs)
        return FakeExperts()

    monkeypatch.setattr(minimax_m3, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        minimax_m3,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=True)
        ),
    )
    monkeypatch.setattr(minimax_m3, "GateLinear", FakeGate)
    monkeypatch.setattr(minimax_m3, "MiniMaxM3MLP", FakeSharedMLP)
    monkeypatch.setattr(minimax_m3, "FusedMoE", fake_fused_moe)

    config = SimpleNamespace(
        num_local_experts=32,
        num_experts_per_tok=2,
        hidden_size=64,
        intermediate_size=16,
        scoring_func="sigmoid",
        swiglu_limit=7.0,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        routed_scaling_factor=1.0,
        n_shared_experts=1,
        use_routing_bias=False,
    )
    layer = minimax_m3.MiniMaxM3MoE(config=config, layer_id=0)

    assert layer.is_sequence_parallel
    assert shared_mlp_kwargs["is_sequence_parallel"] is True
    assert fused_moe_kwargs["is_sequence_parallel"] is True
