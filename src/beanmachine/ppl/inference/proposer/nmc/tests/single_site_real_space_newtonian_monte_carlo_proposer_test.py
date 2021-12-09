# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.autograd
import torch.distributions as dist
from beanmachine.ppl.legacy.inference.proposer.single_site_real_space_newtonian_monte_carlo_proposer import (
    SingleSiteRealSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.world import Variable, World
from torch import tensor


class SingleSiteRealSpaceNewtonianMonteCarloProposerTest(unittest.TestCase):
    class SampleNormalModel:
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleLogisticRegressionModel:
        @bm.random_variable
        def theta_0(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def theta_1(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def x(self, i):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def y(self, i):
            y = self.theta_1() * self.x(i) + self.theta_0()
            probs = 1 / (1 + (y * -1).exp())
            return dist.Bernoulli(probs)

    def test_mean_scale_tril_for_node_with_child(self):
        model = self.SampleNormalModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer()
        foo_key = model.foo()
        bar_key = model.bar()
        val = tensor([2.0, 2.0])
        val.requires_grad_(True)
        distribution = dist.MultivariateNormal(
            tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1]])
        )
        log = distribution.log_prob(val).sum()
        world_vars = nw.world_.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=log,
            children=set({bar_key}),
            transformed_value=val,
            jacobian=tensor(0.0),
        )

        distribution = dist.MultivariateNormal(val, tensor([[1.0, 0.8], [0.8, 1.0]]))
        observed_val = tensor([2.0, 2.0])
        world_vars[bar_key] = Variable(
            distribution=distribution,
            value=observed_val,
            log_prob=distribution.log_prob(observed_val).sum(),
            parent=set({foo_key}),
            transformed_value=observed_val,
            jacobian=tensor(0.0),
        )

        prop_dist = nw_proposer.get_proposal_distribution(
            foo_key, world_vars[foo_key], nw.world_, {}
        )[0]
        mean, scale_tril = parse_arguments(prop_dist.arguments)
        expected_mean = tensor([1.5, 1.5])
        expected_scale_tril = torch.linalg.cholesky(
            tensor([[0.5000, 0.4000], [0.4000, 0.5000]])
        )
        self.assertTrue(torch.isclose(mean, expected_mean).all())
        self.assertTrue(torch.isclose(scale_tril, expected_scale_tril).all())

    def test_mean_scale_tril(self):
        model = self.SampleNormalModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer()
        foo_key = model.foo()
        val = tensor([2.0, 2.0])
        val.requires_grad_(True)
        distribution = dist.MultivariateNormal(
            tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1]])
        )
        log = distribution.log_prob(val)
        world_vars = nw.world_.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=log,
            transformed_value=val,
            jacobian=tensor(0.0),
        )

        prop_dist = nw_proposer.get_proposal_distribution(
            foo_key, world_vars[foo_key], nw.world_, {}
        )[0]
        mean, scale_tril = parse_arguments(prop_dist.arguments)

        expected_mean = tensor([1.0, 1.0])
        expected_scale_tril = torch.linalg.cholesky(tensor([[1.0, 0.8], [0.8, 1]]))
        self.assertTrue(torch.isclose(mean, expected_mean).all())
        self.assertTrue(torch.isclose(scale_tril, expected_scale_tril).all())

    def test_mean_scale_tril_for_iids(self):
        model = self.SampleNormalModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer()
        foo_key = model.foo()
        val = tensor([[2.0, 2.0], [2.0, 2.0]])
        val.requires_grad_(True)
        distribution = dist.Normal(
            tensor([[1.0, 1.0], [1.0, 1.0]]), tensor([[1.0, 1.0], [1.0, 1.0]])
        )
        log = distribution.log_prob(val).sum()
        world_vars = nw.world_.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=log,
            transformed_value=val,
            jacobian=tensor(0.0),
        )

        prop_dist = nw_proposer.get_proposal_distribution(
            foo_key, world_vars[foo_key], nw.world_, {}
        )[0]
        mean, scale_tril = parse_arguments(prop_dist.arguments)

        expected_mean = tensor([1.0, 1.0, 1.0, 1.0])
        expected_scale_tril = torch.eye(4)
        self.assertTrue(torch.isclose(mean, expected_mean).all())
        self.assertTrue(torch.isclose(scale_tril, expected_scale_tril).all())

    def test_multi_mean_scale_tril_computation_in_inference(self):
        model = self.SampleLogisticRegressionModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer()

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
        world_vars = nw.world_.variables_.vars()
        world_vars[theta_0_key] = Variable(
            distribution=theta_0_distribution,
            value=theta_0_value,
            log_prob=theta_0_distribution.log_prob(theta_0_value),
            children=set({y_0_key, y_1_key}),
            transformed_value=theta_0_value,
            jacobian=tensor(0.0),
        )

        world_vars[theta_1_key] = Variable(
            distribution=theta_0_distribution,
            value=theta_1_value,
            log_prob=theta_0_distribution.log_prob(theta_1_value),
            children=set({y_0_key, y_1_key}),
            transformed_value=theta_1_value,
            jacobian=tensor(0.0),
        )

        x_distribution = dist.Normal(torch.tensor(0.0), torch.tensor(5.0))
        world_vars[x_0_key] = Variable(
            distribution=x_distribution,
            value=x_0_value,
            log_prob=x_distribution.log_prob(x_0_value),
            children=set({y_0_key, y_1_key}),
            transformed_value=x_0_value,
            jacobian=tensor(0.0),
        )

        world_vars[x_1_key] = Variable(
            distribution=x_distribution,
            value=x_1_value,
            log_prob=x_distribution.log_prob(x_1_value),
            children=set({y_0_key, y_1_key}),
            transformed_value=x_1_value,
            jacobian=tensor(0.0),
        )

        y = theta_0_value + theta_1_value * x_0_value
        probs_0 = 1 / (1 + (y * -1).exp())
        y_0_distribution = dist.Bernoulli(probs_0)

        world_vars[y_0_key] = Variable(
            distribution=y_0_distribution,
            value=tensor(1.0),
            log_prob=y_0_distribution.log_prob(tensor(1.0)),
            parent=set({theta_0_key, theta_1_key, x_0_key}),
            transformed_value=tensor(1.0),
            jacobian=tensor(0.0),
        )

        y = theta_0_value + theta_1_value * x_1_value
        probs_1 = 1 / (1 + (y * -1).exp())
        y_1_distribution = dist.Bernoulli(probs_1)

        world_vars[y_1_key] = Variable(
            distribution=y_1_distribution,
            value=tensor(1.0),
            log_prob=y_1_distribution.log_prob(tensor(1.0)),
            parent=set({theta_0_key, theta_1_key, x_1_key}),
            transformed_value=tensor(1.0),
            jacobian=tensor(0.0),
        )

        prop_dist = nw_proposer.get_proposal_distribution(
            theta_0_key, world_vars[theta_0_key], nw.world_, {}
        )[0]
        mean, scale_tril = parse_arguments(prop_dist.arguments)

        score = theta_0_distribution.log_prob(theta_0_value)
        score += (
            1 / (1 + (-1 * (theta_0_value + theta_1_value * x_0_value)).exp())
        ).log()
        score += (
            1 / (1 + (-1 * (theta_0_value + theta_1_value * x_1_value)).exp())
        ).log()

        expected_first_gradient = torch.autograd.grad(
            score, theta_0_value, create_graph=True
        )[0]
        expected_second_gradient = torch.autograd.grad(
            expected_first_gradient, theta_0_value
        )[0]

        expected_covar = expected_second_gradient.reshape(1, 1).inverse() * -1
        expected_scale_tril = torch.linalg.cholesky(expected_covar)
        self.assertAlmostEqual(
            expected_scale_tril.item(), scale_tril.item(), delta=0.001
        )
        expected_first_gradient = expected_first_gradient.unsqueeze(0)
        expected_mean = (
            theta_0_value.unsqueeze(0)
            + expected_first_gradient.unsqueeze(0).mm(expected_covar)
        ).squeeze(0)
        self.assertAlmostEqual(mean.item(), expected_mean.item(), delta=0.001)

        proposal_value = (
            dist.MultivariateNormal(mean, scale_tril=scale_tril)
            .sample()
            .reshape(theta_0_value.shape)
        )
        proposal_value.requires_grad_(True)
        world_vars[theta_0_key].value = proposal_value
        world_vars[theta_0_key].transformed_value = proposal_value
        world_vars[theta_0_key].log_prob = theta_0_distribution.log_prob(proposal_value)

        y = proposal_value + theta_1_value * x_0_value
        probs_0 = 1 / (1 + (y * -1).exp())
        y_0_distribution = dist.Bernoulli(probs_0)
        world_vars[y_0_key].distribution = y_0_distribution
        world_vars[y_0_key].log_prob = y_0_distribution.log_prob(tensor(1.0))

        y = proposal_value + theta_1_value * x_1_value
        probs_1 = 1 / (1 + (y * -1).exp())
        y_1_distribution = dist.Bernoulli(probs_1)
        world_vars[y_1_key].distribution = y_1_distribution
        world_vars[y_1_key].log_prob = y_1_distribution.log_prob(tensor(1.0))

        prop_dist = nw_proposer.get_proposal_distribution(
            theta_0_key, world_vars[theta_0_key], nw.world_, {}
        )[0]
        mean, scale_tril = parse_arguments(prop_dist.arguments)

        score = tensor(0.0)

        score = theta_0_distribution.log_prob(proposal_value)
        score += (
            1 / (1 + (-1 * (proposal_value + theta_1_value * x_0_value)).exp())
        ).log()
        score += (
            1 / (1 + (-1 * (proposal_value + theta_1_value * x_1_value)).exp())
        ).log()

        expected_first_gradient = torch.autograd.grad(
            score, proposal_value, create_graph=True
        )[0]
        expected_second_gradient = torch.autograd.grad(
            expected_first_gradient, proposal_value
        )[0]
        expected_covar = expected_second_gradient.reshape(1, 1).inverse() * -1
        expected_scale_tril = torch.linalg.cholesky(expected_covar)
        self.assertAlmostEqual(
            expected_scale_tril.item(), scale_tril.item(), delta=0.001
        )
        expected_first_gradient = expected_first_gradient.unsqueeze(0)

        expected_mean = (
            proposal_value.unsqueeze(0)
            + expected_first_gradient.unsqueeze(0).mm(expected_covar)
        ).squeeze(0)
        self.assertAlmostEqual(mean.item(), expected_mean.item(), delta=0.001)

        self.assertAlmostEqual(
            scale_tril.item(), expected_scale_tril.item(), delta=0.001
        )

    def test_adaptive_alpha_beta_computation(self):
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer()
        nw_proposer.learning_rate_ = tensor(0.0416, dtype=torch.float64)
        nw_proposer.running_mean_, nw_proposer.running_var_ = (
            tensor(0.079658),
            tensor(0.0039118),
        )
        nw_proposer.accepted_samples_ = 37
        alpha, beta = nw_proposer.compute_beta_priors_from_accepted_lr()
        self.assertAlmostEqual(nw_proposer.running_mean_.item(), 0.0786, delta=0.0001)
        self.assertAlmostEqual(nw_proposer.running_var_.item(), 0.00384, delta=0.00001)
        self.assertAlmostEqual(alpha.item(), 1.4032, delta=0.001)
        self.assertAlmostEqual(beta.item(), 16.4427, delta=0.001)

    def test_adaptive_vectorized_alpha_beta_computation(self):
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer()
        nw_proposer.learning_rate_ = tensor([0.0416, 0.0583], dtype=torch.float64)
        nw_proposer.running_mean_, nw_proposer.running_var_ = (
            tensor([0.079658, 0.089861]),
            tensor([0.0039118, 0.0041231]),
        )
        nw_proposer.accepted_samples_ = 37
        alpha, beta = nw_proposer.compute_beta_priors_from_accepted_lr()
        self.assertListEqual(
            [round(x.item(), 4) for x in list(nw_proposer.running_mean_)],
            [0.0786, 0.089],
        )
        self.assertListEqual(
            [round(x.item(), 4) for x in list(nw_proposer.running_var_)],
            [0.0038, 0.004],
        )
        self.assertListEqual(
            [round(x.item(), 4) for x in list(alpha)], [1.4032, 1.6984]
        )
        self.assertListEqual(
            [round(x.item(), 4) for x in list(beta)], [16.4427, 17.3829]
        )


# simple function to read the arguments from a proposal distribution object
# and reconstruct the mean and scale_tril
def parse_arguments(_arguments):
    if "scale_tril" in _arguments:
        covar = _arguments["scale_tril"]
    else:
        (eig_vals, eig_vecs) = _arguments["eig_decomp"]
        covar = eig_vecs @ (torch.eye(len(eig_vals)) * eig_vals) @ eig_vecs.T
    mean = (_arguments["distance"] + _arguments["node_val_reshaped"]).squeeze(0)
    return mean, covar
