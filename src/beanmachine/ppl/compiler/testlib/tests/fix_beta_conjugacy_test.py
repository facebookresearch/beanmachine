# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Parameterized test to compare samples from original
   and conjugate prior transformed models"""

import random

import pytest
import scipy
import torch
from beanmachine.ppl.compiler.testlib.conjugate_models import (
    BetaBernoulliBasicModel,
    BetaBernoulliOpsModel,
    BetaBernoulliScaleHyperParameters,
)
from beanmachine.ppl.inference.bmg_inference import BMGInference


_alpha = 2.0
_beta = 2.0

test_models = [
    (BetaBernoulliBasicModel, "BetaBernoulliConjugateFixer"),
    (BetaBernoulliOpsModel, "BetaBernoulliConjugateFixer"),
    (BetaBernoulliScaleHyperParameters, "BetaBernoulliConjugateFixer"),
]


@pytest.mark.parametrize("model, opt", test_models)
def test_samples_with_ks(model, opt):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)

    num_samples = 3000
    num_obs = 4
    bmg = BMGInference()
    model = model(_alpha, _beta)
    observations = model.gen_obs(num_obs)
    queries = [model.theta()]

    # Generate samples from model when opt is disabled
    skip_optimizations = {opt}
    posterior_original = bmg.infer(queries, observations, num_samples)
    graph_original = bmg.to_dot(
        queries, observations, skip_optimizations=skip_optimizations
    )
    theta_samples_original = posterior_original[model.theta()][0]

    # Generate samples from model when opt is enabled
    skip_optimizations = set()
    posterior_transformed = bmg.infer(
        queries, observations, num_samples, 1, skip_optimizations=skip_optimizations
    )
    graph_transformed = bmg.to_dot(
        queries, observations, skip_optimizations=skip_optimizations
    )
    theta_samples_transformed = posterior_transformed[model.theta()][0]

    assert (
        graph_original.strip() != graph_transformed.strip()
    ), "Original and transformed graph should not be identical."
    assert type(theta_samples_original) == type(
        theta_samples_transformed
    ), "Sample type of original and transformed model should be the same."
    assert len(theta_samples_original) == len(
        theta_samples_transformed
    ), "Sample size of original and transformed model should be the same."

    assert (
        scipy.stats.ks_2samp(theta_samples_original, theta_samples_transformed).pvalue
        >= 0.05
    )
