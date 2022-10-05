# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.autograd
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.nmc.single_site_real_space_nmc_proposer import (
    SingleSiteRealSpaceNMCProposer as SingleSiteRealSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.world import World
from beanmachine.ppl.world.variable import Variable
from torch import tensor


class SingleSiteRealSpaceNewtonianMonteCarloProposerTest(unittest.TestCase):
    class SampleNormalModel:
        @bm.random_variable
        def foo(self):
            return dist.MultivariateNormal(torch.zeros(2), torch.eye(2))

        @bm.random_variable
        def bar(self):
            return dist.MultivariateNormal(self.foo(), torch.eye(2))

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
        foo_key = bm.random_variable(
            lambda: dist.MultivariateNormal(
                tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1]])
            )
        )
        bar_key = bm.random_variable(
            lambda: dist.MultivariateNormal(
                foo_key(),
                tensor([[1.0, 0.8], [0.8, 1.0]]),
            )
        )
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer(foo_key())
        val = tensor([2.0, 2.0])
        queries = [foo_key(), bar_key()]
        observed_val = tensor([2.0, 2.0])
        observations = {bar_key(): observed_val}
        world = World.initialize_world(queries, observations)
        world_vars = world._variables
        world_vars[foo_key] = val

        nw_proposer.learning_rate_ = 1.0
        prop_dist = nw_proposer.get_proposal_distribution(world).base_dist
        mean, scale_tril = prop_dist.mean, prop_dist.scale_tril
        expected_mean = tensor([1.5, 1.5])
        expected_scale_tril = torch.linalg.cholesky(
            tensor([[0.5000, 0.4000], [0.4000, 0.5000]])
        )
        self.assertTrue(torch.isclose(mean, expected_mean).all())
        self.assertTrue(torch.isclose(scale_tril, expected_scale_tril).all())

    def test_mean_scale_tril(self):
        model = self.SampleNormalModel()
        foo_key = model.foo()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer(foo_key)
        val = tensor([2.0, 2.0])
        val.requires_grad_(True)
        distribution = dist.MultivariateNormal(
            tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1]])
        )
        queries = [foo_key]
        observations = {}
        world = World.initialize_world(queries, observations)
        world_vars = world._variables
        world_vars[foo_key] = Variable(
            value=val,
            distribution=distribution,
        )

        nw_proposer.learning_rate_ = 1.0
        prop_dist = nw_proposer.get_proposal_distribution(world).base_dist
        mean, scale_tril = prop_dist.mean, prop_dist.scale_tril

        expected_mean = tensor([1.0, 1.0])
        expected_scale_tril = torch.linalg.cholesky(tensor([[1.0, 0.8], [0.8, 1]]))
        self.assertTrue(torch.isclose(mean, expected_mean).all())
        self.assertTrue(torch.isclose(scale_tril, expected_scale_tril).all())

    def test_mean_scale_tril_for_iids(self):
        model = self.SampleNormalModel()
        foo_key = model.foo()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer(foo_key)
        val = tensor([[2.0, 2.0], [2.0, 2.0]])
        val.requires_grad_(True)
        distribution = dist.Normal(
            tensor([[1.0, 1.0], [1.0, 1.0]]), tensor([[1.0, 1.0], [1.0, 1.0]])
        )
        queries = [foo_key]
        observations = {}
        world = World.initialize_world(queries, observations)
        world_vars = world._variables
        world_vars[foo_key] = Variable(
            value=val,
            distribution=distribution,
        )

        nw_proposer.learning_rate_ = 1.0
        prop_dist = nw_proposer.get_proposal_distribution(world).base_dist
        mean, scale_tril = prop_dist.mean, prop_dist.scale_tril

        expected_mean = tensor([1.0, 1.0, 1.0, 1.0])
        expected_scale_tril = torch.eye(4)
        self.assertTrue(torch.isclose(mean, expected_mean).all())
        self.assertTrue(torch.isclose(scale_tril, expected_scale_tril).all())

    def test_multi_mean_scale_tril_computation_in_inference(self):
        model = self.SampleLogisticRegressionModel()
        theta_0_key = model.theta_0()
        theta_1_key = model.theta_1()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer(theta_0_key)

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

        nw_proposer.learning_rate_ = 1.0
        prop_dist = nw_proposer.get_proposal_distribution(world).base_dist
        mean, scale_tril = prop_dist.mean, prop_dist.scale_tril

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

        y = proposal_value + theta_1_value * x_0_value
        probs_0 = 1 / (1 + (y * -1).exp())
        y_0_distribution = dist.Bernoulli(probs_0)
        world_vars[y_0_key].distribution = y_0_distribution
        world_vars[y_0_key].log_prob = y_0_distribution.log_prob(tensor(1.0))
        y = proposal_value + theta_1_value * x_1_value
        probs_1 = 1 / (1 + (y * -1).exp())
        y_1_distribution = dist.Bernoulli(probs_1)
        world_vars[y_1_key].distribution = y_1_distribution

        nw_proposer.learning_rate_ = 1.0
        prop_dist = nw_proposer.get_proposal_distribution(world).base_dist
        mean, scale_tril = prop_dist.mean, prop_dist.scale_tril

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
        model = self.SampleLogisticRegressionModel()
        theta_0_key = model.theta_0()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer(theta_0_key)
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
        model = self.SampleLogisticRegressionModel()
        theta_0_key = model.theta_0()
        nw_proposer = SingleSiteRealSpaceNewtonianMonteCarloProposer(theta_0_key)
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
