# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


class SingleSiteNewtonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel(object):
        @sample
        def foo(self):
            return dist.Normal(torch.tensor(2.0), torch.tensor(2.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_single_site_newtonian_monte_carlo(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        foo_key = model.foo()
        bar_key = model.bar()
        nw.queries_ = [model.foo()]
        nw.observations_ = {model.bar(): torch.tensor(0.0)}
        nw._infer(10)

        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        self.assertEqual(foo_key in nw.world_.variables_, True)
        self.assertEqual(bar_key in nw.world_.variables_, True)
        self.assertEqual(foo_key in nw.world_.variables_[bar_key].parent, True)
        self.assertEqual(bar_key in nw.world_.variables_[foo_key].children, True)

    def test_mean_covariance(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
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

        mean, covariance = nw.compute_normal_mean_covar(nw.world_.variables_[foo_key])
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

        mean, covariance = nw.compute_normal_mean_covar(nw.world_.variables_[foo_key])
        expected_mean = tensor([1.0, 1.0, 1.0, 1.0])
        expected_covariance = torch.eye(4)
        self.assertAlmostEqual((mean - expected_mean).sum().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(
            (covariance - expected_covariance).sum().item(), 0.0, delta=0.01
        )

    def test_mean_covariance_for_node_with_child(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        nw.world_ = World()
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

        mean, covariance = nw.compute_normal_mean_covar(nw.world_.variables_[foo_key])
        expected_mean = tensor([1.5, 1.5])
        expected_covariance = tensor([[0.5000, 0.4000], [0.4000, 0.5000]])
        self.assertAlmostEqual(
            abs((mean - expected_mean).sum().item()), 0.0, delta=0.01
        )
        self.assertAlmostEqual(
            abs((covariance - expected_covariance).sum().item()), 0.0, delta=0.01
        )
