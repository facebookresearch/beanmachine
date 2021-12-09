# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.world.initialize_fn import (
    init_from_prior,
    init_to_uniform,
)


@pytest.mark.parametrize("init_fn", [init_from_prior, init_to_uniform])
@pytest.mark.parametrize(
    "distribution",
    [
        dist.Uniform(0.0, 1.0),
        dist.Normal(0.0, 1.0).expand((3,)),
        dist.Bernoulli(0.5),
        dist.Exponential(1.0),
        dist.Dirichlet(torch.tensor([0.5, 0.5])),
        dist.Categorical(logits=torch.randn(5, 10)),
        dist.Bernoulli(0.5).expand((3, 5, 7)),
        dist.Poisson(rate=2.0),
    ],
)
def test_initialize_validness(init_fn, distribution):
    value = init_fn(distribution)
    # make sure values are initialize within the constraint
    assert torch.all(distribution.support.check(value))
    assert not torch.any(torch.isnan(distribution.log_prob(value)))
    assert value.size() == distribution.sample().size()
