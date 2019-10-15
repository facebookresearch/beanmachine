import unittest

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


class SingleSiteNewtonianMonteCarloProposerTest(unittest.TestCase):
    class SampleNormalModel(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleLogisticRegressionModel(object):
        @sample
        def theta_0(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def theta_1(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def x(self, i):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def y(self, i):
            y = self.theta_1() * self.x(i) + self.theta_0()
            probs = 1 / (1 + (y * -1).exp())
            return dist.Bernoulli(probs)

    def test_mean_covariance_for_node_with_child(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteNewtonianMonteCarloProposer(nw.world_)
        foo_key = model.foo()
        bar_key = model.bar()
        val = tensor([2.0, 2.0])
        val.requires_grad_(True)
        distribution = dist.MultivariateNormal(
            tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1]])
        )
        log = distribution.log_prob(val).sum()
        nw.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=log,
            parent=set(),
            children=set({bar_key}),
            mean=None,
            covariance=None,
        )

        distribution = dist.MultivariateNormal(val, tensor([[1.0, 0.8], [0.8, 1.0]]))
        observed_val = tensor([2.0, 2.0])
        nw.world_.variables_[bar_key] = Variable(
            distribution=distribution,
            value=observed_val,
            log_prob=distribution.log_prob(observed_val).sum(),
            parent=set({foo_key}),
            children=set(),
            mean=None,
            covariance=None,
        )

        mean, covariance = nw_proposer.compute_normal_mean_covar(
            nw.world_.variables_[foo_key]
        )
        expected_mean = tensor([1.5, 1.5])
        expected_covariance = tensor([[0.5000, 0.4000], [0.4000, 0.5000]])
        self.assertAlmostEqual(
            abs((mean - expected_mean).sum().item()), 0.0, delta=0.01
        )
        self.assertAlmostEqual(
            abs((covariance - expected_covariance).sum().item()), 0.0, delta=0.01
        )

    def test_mean_covariance(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteNewtonianMonteCarloProposer(nw.world_)
        foo_key = model.foo()
        val = tensor([2.0, 2.0])
        val.requires_grad_(True)
        distribution = dist.MultivariateNormal(
            tensor([1.0, 1.0]), tensor([[1.0, 0.8], [0.8, 1]])
        )
        log = distribution.log_prob(val)
        nw.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=log,
            parent=set(),
            children=set(),
            mean=None,
            covariance=None,
        )

        mean, covariance = nw_proposer.compute_normal_mean_covar(
            nw.world_.variables_[foo_key]
        )
        expected_mean = tensor([1.0, 1.0])
        expected_covariance = tensor([[1.0, 0.8], [0.8, 1]])
        self.assertAlmostEqual((mean - expected_mean).sum().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(
            (covariance - expected_covariance).sum().item(), 0.0, delta=0.01
        )

    def test_mean_covariance_for_iids(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteNewtonianMonteCarloProposer(nw.world_)
        foo_key = model.foo()
        val = tensor([[2.0, 2.0], [2.0, 2.0]])
        val.requires_grad_(True)
        distribution = dist.Normal(
            tensor([[1.0, 1.0], [1.0, 1.0]]), tensor([[1.0, 1.0], [1.0, 1.0]])
        )
        log = distribution.log_prob(val).sum()
        nw.world_.variables_[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=log,
            parent=set(),
            children=set(),
            mean=None,
            covariance=None,
        )

        mean, covariance = nw_proposer.compute_normal_mean_covar(
            nw.world_.variables_[foo_key]
        )
        expected_mean = tensor([1.0, 1.0, 1.0, 1.0])
        expected_covariance = torch.eye(4)
        self.assertAlmostEqual((mean - expected_mean).sum().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(
            (covariance - expected_covariance).sum().item(), 0.0, delta=0.01
        )

    def test_multi_mean_covariance_computation_in_inference(self):
        model = self.SampleLogisticRegressionModel()
        nw = SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
        nw_proposer = SingleSiteNewtonianMonteCarloProposer(nw.world_)

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
        nw.world_.variables_[theta_0_key] = Variable(
            distribution=theta_0_distribution,
            value=theta_0_value,
            log_prob=theta_0_distribution.log_prob(theta_0_value),
            parent=set(),
            children=set({y_0_key, y_1_key}),
            mean=None,
            covariance=None,
        )

        nw.world_.variables_[theta_1_key] = Variable(
            distribution=theta_0_distribution,
            value=theta_1_value,
            log_prob=theta_0_distribution.log_prob(theta_1_value),
            parent=set(),
            children=set({y_0_key, y_1_key}),
            mean=None,
            covariance=None,
        )

        x_distribution = dist.Normal(torch.tensor(0.0), torch.tensor(5.0))
        nw.world_.variables_[x_0_key] = Variable(
            distribution=x_distribution,
            value=x_0_value,
            log_prob=x_distribution.log_prob(x_0_value),
            parent=set(),
            children=set({y_0_key, y_1_key}),
            mean=None,
            covariance=None,
        )

        nw.world_.variables_[x_1_key] = Variable(
            distribution=x_distribution,
            value=x_1_value,
            log_prob=x_distribution.log_prob(x_1_value),
            parent=set(),
            children=set({y_0_key, y_1_key}),
            mean=None,
            covariance=None,
        )

        y = theta_0_value + theta_1_value * x_0_value
        probs_0 = 1 / (1 + (y * -1).exp())
        y_0_distribution = dist.Bernoulli(probs_0)

        nw.world_.variables_[y_0_key] = Variable(
            distribution=y_0_distribution,
            value=tensor(1.0),
            log_prob=y_0_distribution.log_prob(tensor(1.0)),
            parent=set({theta_0_key, theta_1_key, x_0_key}),
            children=set(),
            mean=None,
            covariance=None,
        )

        y = theta_0_value + theta_1_value * x_1_value
        probs_1 = 1 / (1 + (y * -1).exp())
        y_1_distribution = dist.Bernoulli(probs_1)

        nw.world_.variables_[y_1_key] = Variable(
            distribution=y_1_distribution,
            value=tensor(1.0),
            log_prob=y_1_distribution.log_prob(tensor(1.0)),
            parent=set({theta_0_key, theta_1_key, x_1_key}),
            children=set(),
            mean=None,
            covariance=None,
        )

        mean, covariance = nw_proposer.compute_normal_mean_covar(
            nw.world_.variables_[theta_0_key]
        )

        score = theta_0_distribution.log_prob(theta_0_value)
        score += (
            1 / (1 + (-1 * (theta_0_value + theta_1_value * x_0_value)).exp())
        ).log()
        score += (
            1 / (1 + (-1 * (theta_0_value + theta_1_value * x_1_value)).exp())
        ).log()

        score.backward(create_graph=True)
        expected_first_gradient = theta_0_value.grad.clone()

        expected_first_gradient.index_select(0, tensor([0])).backward(create_graph=True)
        expected_second_gradient = (
            theta_0_value.grad - expected_first_gradient
        ).unsqueeze(0)
        expected_covariance = (expected_second_gradient.unsqueeze(0)).inverse() * -1
        self.assertAlmostEqual(
            expected_covariance.item(), covariance.item(), delta=0.001
        )
        expected_first_gradient = expected_first_gradient.unsqueeze(0)
        expected_mean = (
            theta_0_value.unsqueeze(0)
            + expected_first_gradient.unsqueeze(0).mm(expected_covariance)
        ).squeeze(0)
        self.assertAlmostEqual(mean.item(), expected_mean.item(), delta=0.001)

        proposal_value = (
            dist.MultivariateNormal(mean, covariance)
            .sample()
            .reshape(theta_0_value.shape)
        )
        proposal_value.requires_grad_(True)
        nw.world_.variables_[theta_0_key].value = proposal_value
        nw.world_.variables_[theta_0_key].log_prob = theta_0_distribution.log_prob(
            proposal_value
        )

        y = proposal_value + theta_1_value * x_0_value
        probs_0 = 1 / (1 + (y * -1).exp())
        y_0_distribution = dist.Bernoulli(probs_0)
        nw.world_.variables_[y_0_key].distribution = y_0_distribution
        nw.world_.variables_[y_0_key].log_prob = y_0_distribution.log_prob(tensor(1.0))

        y = proposal_value + theta_1_value * x_1_value
        probs_1 = 1 / (1 + (y * -1).exp())
        y_1_distribution = dist.Bernoulli(probs_1)
        nw.world_.variables_[y_1_key].distribution = y_1_distribution
        nw.world_.variables_[y_1_key].log_prob = y_1_distribution.log_prob(tensor(1.0))

        mean, covariance = nw_proposer.compute_normal_mean_covar(
            nw.world_.variables_[theta_0_key]
        )

        score = tensor(0.0)

        score = theta_0_distribution.log_prob(proposal_value)
        score += (
            1 / (1 + (-1 * (proposal_value + theta_1_value * x_0_value)).exp())
        ).log()
        score += (
            1 / (1 + (-1 * (proposal_value + theta_1_value * x_1_value)).exp())
        ).log()

        score.backward(create_graph=True)
        expected_first_gradient = proposal_value.grad.clone()

        expected_first_gradient.index_select(0, tensor([0])).backward(create_graph=True)
        expected_second_gradient = (
            proposal_value.grad - expected_first_gradient
        ).unsqueeze(0)
        expected_covariance = (expected_second_gradient.unsqueeze(0)).inverse() * -1
        self.assertAlmostEqual(
            expected_covariance.item(), covariance.item(), delta=0.001
        )
        expected_first_gradient = expected_first_gradient.unsqueeze(0)
        expected_mean = (
            proposal_value.unsqueeze(0)
            + expected_first_gradient.unsqueeze(0).mm(expected_covariance)
        ).squeeze(0)
        self.assertAlmostEqual(mean.item(), expected_mean.item(), delta=0.001)

        self.assertAlmostEqual(
            covariance.item(), expected_covariance.item(), delta=0.001
        )
