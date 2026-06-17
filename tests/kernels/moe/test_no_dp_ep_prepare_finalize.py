# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
    MoEPrepareAndFinalizeNoDPEPMonolithic,
)


def test_no_dp_ep_modular_supports_dbo() -> None:
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPModular()

    assert prepare_finalize.supports_dbo()
    assert not prepare_finalize.supports_async()


def test_no_dp_ep_monolithic_does_not_support_dbo() -> None:
    prepare_finalize = MoEPrepareAndFinalizeNoDPEPMonolithic()

    assert not prepare_finalize.supports_dbo()
