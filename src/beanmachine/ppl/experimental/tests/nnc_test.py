# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist


class SampleModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(0.0, 1.0)

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), 1.0)


@pytest.mark.parametrize(
    "algorithm",
    [
        bm.GlobalNoUTurnSampler(nnc_compile=True),
        bm.GlobalHamiltonianMonteCarlo(trajectory_length=1.0, nnc_compile=True),
    ],
)
def test_nnc_compile(algorithm):
    model = SampleModel()
    queries = [model.foo()]
    observations = {model.bar(): torch.tensor(0.5)}
    num_samples = 30
    num_chains = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # verify that NNC can run through
        samples = algorithm.infer(
            queries,
            observations,
            num_samples,
            num_adaptive_samples=num_samples,
            num_chains=num_chains,
        )
    # sanity check: make sure that the samples are valid
    assert not torch.isnan(samples[model.foo()]).any()
