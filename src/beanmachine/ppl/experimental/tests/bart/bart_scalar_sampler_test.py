# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from beanmachine.ppl.experimental.causal_inference.models.bart.scalar_samplers import (
    NoiseStandardDeviation,
)


@pytest.fixture
def X():
    return torch.Tensor([[3.0, 1.0], [4.0, 1.0], [1.5, 1.0], [-1.0, 1.0]])


@pytest.fixture
def residual(X):
    return X * 0.1


@pytest.fixture
def sigma():
    return NoiseStandardDeviation(prior_concentration=0.1, prior_rate=0.2)


def test_sigma_sampling(sigma, X, residual):
    prev_val = sigma.val
    sample = sigma.sample(X=X, residual=residual)
    assert not prev_val == sigma.val
    assert sigma.val == sample
