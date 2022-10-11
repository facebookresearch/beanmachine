# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from beanmachine.ppl.examples.conjugate_models import BetaBernoulliModel
from beanmachine.ppl.examples.primitive_models import DirichletModel
from beanmachine.ppl.inference.proposer.nmc import SingleSiteSimplexSpaceNMCProposer
from beanmachine.ppl.inference.single_site_nmc import SingleSiteNewtonianMonteCarlo
from beanmachine.ppl.world import World
from torch import tensor

from ....utils.fixtures import (
    approx,
    parametrize_inference,
    parametrize_model,
    parametrize_proposer,
)


@parametrize_model([DirichletModel(tensor([[0.5, 0.5], [0.5, 0.5]]))])
@parametrize_proposer([SingleSiteSimplexSpaceNMCProposer])
def test_alpha_for_dirichlet(model, proposer):
    world_ = World()
    with world_:
        model.x()
    prop = proposer(model.x())
    is_valid, predicted_alpha = prop.compute_alpha(world_)
    assert is_valid
    assert approx(model.alpha.sum(), predicted_alpha.sum(), 1e-4)


@parametrize_model([BetaBernoulliModel(tensor(2.0), tensor(2.0))])
@parametrize_inference([SingleSiteNewtonianMonteCarlo()])
def test_coin_flip(model, inference):
    prior_heads, prior_tails = model.alpha_, model.beta_
    heads_observed = 5
    conjugate_posterior_mean = (prior_heads + heads_observed) / (
        prior_heads + prior_tails + heads_observed
    )
    samples = inference.infer(
        queries=[model.theta()],
        observations={model.x(0): torch.ones(heads_observed)},
        num_samples=100,
        num_chains=1,
    ).get_chain(0)
    assert approx(samples[model.theta()].mean(), conjugate_posterior_mean, 5e-2)
