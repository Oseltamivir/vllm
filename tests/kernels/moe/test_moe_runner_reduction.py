# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner as runner_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


class _Kernel:
    def __init__(self, output_is_reduced: bool) -> None:
        self._output_is_reduced = output_is_reduced

    def output_is_reduced(self) -> bool:
        return self._output_is_reduced


def _make_runner(
    *,
    reduce_results: bool,
    kernel_output_is_reduced: bool = False,
    is_sequence_parallel: bool = False,
    routed_scaling_factor: float = 1.0,
) -> MoERunner:
    runner = object.__new__(MoERunner)
    runner.__dict__["reduce_results"] = reduce_results
    runner.__dict__["moe_config"] = SimpleNamespace(
        is_sequence_parallel=is_sequence_parallel,
        tp_size=8,
        ep_size=8,
    )
    runner.__dict__["routed_experts"] = SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=_Kernel(kernel_output_is_reduced),
        )
    )
    runner.__dict__["routed_scaling_factor"] = routed_scaling_factor
    runner.__dict__["routed_output_transform"] = None
    return runner


def test_output_is_reduced_tracks_deferred_reduction() -> None:
    assert _make_runner(reduce_results=True).output_is_reduced
    assert not _make_runner(reduce_results=False).output_is_reduced
    assert _make_runner(
        reduce_results=False, kernel_output_is_reduced=True
    ).output_is_reduced
    assert _make_runner(
        reduce_results=False, is_sequence_parallel=True
    ).output_is_reduced


def test_deferred_final_reduction_skips_allreduce(monkeypatch) -> None:
    calls = 0

    def _allreduce(states: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return states + 1

    monkeypatch.setattr(
        runner_module,
        "tensor_model_parallel_all_reduce",
        _allreduce,
    )
    states = torch.zeros(2, 4)

    deferred = _make_runner(reduce_results=False)
    torch.testing.assert_close(
        deferred._maybe_reduce_final_output(states, None), states
    )
    assert calls == 0

    reduced = _make_runner(reduce_results=True)
    torch.testing.assert_close(
        reduced._maybe_reduce_final_output(states, None),
        states + 1,
    )
    assert calls == 1


def test_gfx942_fuses_bf16_shared_add_with_scale(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "_use_gfx942_fused_shared_routed_add",
        lambda: True,
    )
    runner = _make_runner(reduce_results=True, routed_scaling_factor=2.0)
    shared = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    routed = torch.tensor([[0.75, 1.5]], dtype=torch.bfloat16)
    routed_before = routed.clone()

    actual = runner._combine_shared_and_routed_outputs(shared, routed)
    expected = shared + routed_before * 2.0

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(routed, routed_before, rtol=0, atol=0)
