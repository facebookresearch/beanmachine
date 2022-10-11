# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.autograd
import torch.distributions as dist
from beanmachine.ppl.examples.conjugate_models import MV_NormalNormalModel
from beanmachine.ppl.examples.hierarchical_models import LogisticRegressionModel
from beanmachine.ppl.inference.proposer.nmc.single_site_real_space_nmc_proposer import (
    SingleSiteRealSpaceNMCProposer as SingleSiteRealSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.world import World
from beanmachine.ppl.world.variable import Variable
from torch import tensor

from ....utils.fixtures import (
    approx,
    approx_all,
    parametrize_model,
    parametrize_model_value_expected,
    parametrize_proposer,
    parametrize_value_expected,
)


pytestmark = parametrize_proposer([SingleSiteRealSpaceNewtonianMonteCarloProposer])


@parametrize_model_value_expected(
    [
        (
            MV_NormalNormalModel(
                tensor([1.0, 1.0]),
                tensor([[1.0, 0.8], [0.8, 1]]),
                tensor([[1.0, 0.8], [0.8, 1.0]]),
            ),
            tensor([2.0, 2.0]),
            ([1.5, 1.5], torch.linalg.cholesky(tensor([[0.5, 0.4], [0.4, 0.5]]))),
        ),
        (
            MV_NormalNormalModel(torch.zeros(2), torch.eye(2), torch.eye(2)),
            (
                dist.MultivariateNormal(
                    tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1.0]])
                ),
                tensor([2.0, 2.0]),
            ),
            ([1.0, 1.0], torch.linalg.cholesky(tensor([[1.0, 0.8], [0.8, 1]]))),
        ),
        (
            MV_NormalNormalModel(torch.zeros(2), torch.eye(2), torch.eye(2)),
            (
                dist.Normal(
                    tensor([[1.0, 1.0], [1.0, 1.0]]), tensor([[1.0, 1.0], [1.0, 1.0]])
                ),
                tensor([[2.0, 2.0], [2.0, 2.0]]),
            ),
            ([1.0, 1.0, 1.0, 1.0], torch.eye(4)),
        ),
    ]
)
def test_mean_scale_tril(model, proposer, value, expected):
    # set latents
    if torch.is_tensor(value):
        queries = [model.theta(), model.x()]
        observations = {model.x(): value}
    else:
        queries = [model.theta()]
        observations = {}
    world = World.initialize_world(queries, observations)
    if torch.is_tensor(value):
        world._variables[model.theta] = value
    else:
        dist, val = value
        val.requires_grad_(True)
        world._variables[model.theta()] = Variable(value=val, distribution=dist)

    # evaluate proposer
    prop = proposer(model.theta())
    prop.learning_rate_ = 1.0
    prop_dist = prop.get_proposal_distribution(world).base_dist
    mean, scale_tril = prop_dist.mean, prop_dist.scale_tril
    expected_mean, expected_scale_tril = expected
    assert approx_all(mean, expected_mean)
    assert approx_all(scale_tril, expected_scale_tril)


@parametrize_model([LogisticRegressionModel()])
def test_multi_mean_scale_tril_computation_in_inference(model, proposer):
    theta_0_key = model.theta_0()
    theta_1_key = model.theta_1()

    x_0_key = model.x(0)
    x_1_key = model.x(1)
    y_0_key = model.y(0)
    y_1_key = model.y(1)

    theta_0_value = tensor(1.5708)
    theta_0_value.requires_grad_(True)
    x_0_value = tensor(0.7654)
    x_1_value = tensor(-6.6737)
    theta_1_value = tensor(-0.4459)

    theta_0_distribution = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
    queries = [theta_0_key, theta_1_key]
    observations = {}
    world = World.initialize_world(queries, observations)
    world_vars = world._variables
    world_vars[theta_0_key] = Variable(
        value=theta_0_value,
        distribution=theta_0_distribution,
        children=set({y_0_key, y_1_key}),
    )
    world_vars[theta_1_key] = Variable(
        value=theta_1_value,
        distribution=theta_0_distribution,
        children=set({y_0_key, y_1_key}),
    )

    x_distribution = dist.Normal(torch.tensor(0.0), torch.tensor(5.0))
    world_vars[x_0_key] = Variable(
        value=x_0_value,
        distribution=x_distribution,
        children=set({y_0_key, y_1_key}),
    )
    world_vars[x_1_key] = Variable(
        value=x_1_value,
        distribution=x_distribution,
        children=set({y_0_key, y_1_key}),
    )

    y = theta_0_value + theta_1_value * x_0_value
    probs_0 = 1 / (1 + (y * -1).exp())
    y_0_distribution = dist.Bernoulli(probs_0)

    world_vars[y_0_key] = Variable(
        value=tensor(1.0),
        distribution=y_0_distribution,
        parents=set({theta_0_key, theta_1_key, x_0_key}),
    )
    y = theta_0_value + theta_1_value * x_1_value
    probs_1 = 1 / (1 + (y * -1).exp())
    y_1_distribution = dist.Bernoulli(probs_1)
    world_vars[y_1_key] = Variable(
        value=tensor(1.0),
        distribution=y_1_distribution,
        parents=set({theta_0_key, theta_1_key, x_1_key}),
    )

    prop = proposer(theta_0_key)
    prop.learning_rate_ = 1.0
    prop_dist = prop.get_proposal_distribution(world).base_dist
    mean, scale_tril = prop_dist.mean, prop_dist.scale_tril

    score = theta_0_distribution.log_prob(theta_0_value)
    score += (1 / (1 + (-1 * (theta_0_value + theta_1_value * x_0_value)).exp())).log()
    score += (1 / (1 + (-1 * (theta_0_value + theta_1_value * x_1_value)).exp())).log()

    expected_first_gradient = torch.autograd.grad(
        score, theta_0_value, create_graph=True
    )[0]
    expected_second_gradient = torch.autograd.grad(
        expected_first_gradient, theta_0_value
    )[0]

    expected_covar = expected_second_gradient.reshape(1, 1).inverse() * -1
    expected_scale_tril = torch.linalg.cholesky(expected_covar)
    assert approx(scale_tril, expected_scale_tril, 1e-3)

    expected_first_gradient = expected_first_gradient.unsqueeze(0)
    expected_mean = (
        theta_0_value.unsqueeze(0)
        + expected_first_gradient.unsqueeze(0).mm(expected_covar)
    ).squeeze(0)
    assert approx(mean, expected_mean, 1e-3)

    proposal_value = (
        dist.MultivariateNormal(mean, scale_tril=scale_tril)
        .sample()
        .reshape(theta_0_value.shape)
    )
    proposal_value.requires_grad_(True)
    world_vars[theta_0_key].value = proposal_value

    y = proposal_value + theta_1_value * x_0_value
    probs_0 = 1 / (1 + (y * -1).exp())
    y_0_distribution = dist.Bernoulli(probs_0)
    world_vars[y_0_key].distribution = y_0_distribution
    world_vars[y_0_key].log_prob = y_0_distribution.log_prob(tensor(1.0))
    y = proposal_value + theta_1_value * x_1_value
    probs_1 = 1 / (1 + (y * -1).exp())
    y_1_distribution = dist.Bernoulli(probs_1)
    world_vars[y_1_key].distribution = y_1_distribution

    prop.learning_rate_ = 1.0
    prop_dist = prop.get_proposal_distribution(world).base_dist
    mean, scale_tril = prop_dist.mean, prop_dist.scale_tril

    score = tensor(0.0)

    score = theta_0_distribution.log_prob(proposal_value)
    score += (1 / (1 + (-1 * (proposal_value + theta_1_value * x_0_value)).exp())).log()
    score += (1 / (1 + (-1 * (proposal_value + theta_1_value * x_1_value)).exp())).log()

    expected_first_gradient = torch.autograd.grad(
        score, proposal_value, create_graph=True
    )[0]
    expected_second_gradient = torch.autograd.grad(
        expected_first_gradient, proposal_value
    )[0]
    expected_covar = expected_second_gradient.reshape(1, 1).inverse() * -1
    expected_scale_tril = torch.linalg.cholesky(expected_covar)
    assert approx(scale_tril, expected_scale_tril, 1e-3)

    expected_first_gradient = expected_first_gradient.unsqueeze(0)
    expected_mean = (
        proposal_value.unsqueeze(0)
        + expected_first_gradient.unsqueeze(0).mm(expected_covar)
    ).squeeze(0)
    assert approx(mean, expected_mean, 1e-3)
    assert approx(scale_tril, expected_scale_tril, 1e-3)


@parametrize_model([LogisticRegressionModel()])
@parametrize_value_expected(
    [
        (
            tensor([0.0416, 0.079658, 0.0039118]).to(torch.float64),
            [0.07863, 0.00384, 1.40321, 16.44271],
        ),
        (
            tensor([[0.0416, 0.0583], [0.079658, 0.089861], [0.0039118, 0.0041231]]).to(
                torch.float64
            ),
            [
                [0.07863, 0.08901],
                [0.00384, 0.00403],
                [1.40321, 1.69839],
                [16.44271, 17.38294],
            ],
        ),
    ]
)
def test_adaptive_alpha_beta_computation(model, proposer, value, expected):
    theta_0_key = model.theta_0()
    prop = proposer(theta_0_key)
    prop.learning_rate_, prop.running_mean_, prop.running_var_ = value
    prop.accepted_samples_ = 37
    alpha, beta = prop.compute_beta_priors_from_accepted_lr()
    results = [prop.running_mean_, prop.running_var_, alpha, beta]
    assert approx_all(
        (torch.hstack if results[0].ndim == 0 else torch.vstack)(results),
        expected,
        1e-5,
    )
