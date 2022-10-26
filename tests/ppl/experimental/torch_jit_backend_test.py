# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
import torch

from beanmachine.ppl.experimental.torch_jit_backend import get_backend, TorchJITBackend

from ..inference.inference_test import SampleModel


def test_get_backend():
    with pytest.warns(
        UserWarning, match="The support of TorchInductor is experimental"
    ):
        # test if switching to inductor triggers the warning
        backend = get_backend(nnc_compile=False, experimental_inductor_compile=True)
        assert backend is TorchJITBackend.INDUCTOR

    backend = get_backend(nnc_compile=True, experimental_inductor_compile=False)
    assert backend is TorchJITBackend.NNC

    backend = get_backend(nnc_compile=False, experimental_inductor_compile=False)
    assert backend is TorchJITBackend.NONE


@pytest.mark.skip(reason="The CPU backend of TorchInductor isn't working in fbcode yet")
def test_inductor_compile():
    model = SampleModel()
    queries = [model.foo()]
    observations = {model.bar(): torch.tensor(0.5)}
    num_samples = 30
    num_chains = 2
    # verify that Inductor can run through
    samples = bm.GlobalNoUTurnSampler(experimental_inductor_compile=True).infer(
        queries,
        observations,
        num_samples,
        num_adaptive_samples=num_samples,
        num_chains=num_chains,
    )
    # sanity check: make sure that the samples are valid
    assert not torch.isnan(samples[model.foo()]).any()
