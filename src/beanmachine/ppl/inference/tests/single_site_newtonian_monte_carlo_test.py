# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
    SingleSiteRealSpaceNewtonianMonteCarloProposer,
    SingleSiteSimplexNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.world import TransformType
from beanmachine.ppl.world.utils import BetaDimensionTransform


class SingleSiteNewtonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleTransformModel(object):
        @sample
        def realspace(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def halfspace(self):
            return dist.Gamma(tensor(2.0), tensor(2.0))

        @sample
        def simplex(self):
            return dist.Dirichlet(tensor([0.1, 0.9]))

        @sample
        def interval(self):
            return dist.Uniform(tensor(1.0), tensor(3.0))

        @sample
        def beta(self):
            return dist.Beta(tensor(1.0), tensor(1.0))

    class SampleShapeModel(object):
        @sample
        def realspace(self):
            return dist.Normal(torch.zeros(2, 4), tensor(1.0))

        @sample
        def halfspace(self):
            return dist.Gamma(torch.zeros(1, 2, 4) + tensor(2.0), tensor(2.0))

        @sample
        def simplex(self):
            return dist.Dirichlet(tensor([0.1, 0.9]))

        @sample
        def interval(self):
            return dist.Uniform(tensor(1.0), tensor(3.0))

        @sample
        def beta(self):
            return dist.Beta(tensor([1.0, 2.0, 3.0]), tensor([1.0, 2.0, 3.0]))

    def test_single_site_newtonian_monte_carlo(self):
        model = self.SampleNormalModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        foo_key = model.foo()
        bar_key = model.bar()
        nw.queries_ = [model.foo()]
        nw.observations_ = {model.bar(): torch.tensor(0.0)}
        nw._infer(10)

        world_vars = nw.world_.variables_.vars()
        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        self.assertEqual(foo_key in world_vars, True)
        self.assertEqual(bar_key in world_vars, True)
        self.assertEqual(foo_key in world_vars[bar_key].parent, True)
        self.assertEqual(bar_key in world_vars[foo_key].children, True)

    def test_single_site_newtonian_monte_carlo_default_transform(self):
        model = self.SampleTransformModel()
        nw = SingleSiteNewtonianMonteCarlo(transform_type=TransformType.DEFAULT)

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()

        nw.queries_ = [
            model.realspace(),
            model.halfspace(),
            model.simplex(),
            model.interval(),
            model.beta(),
        ]
        nw.observations_ = {}
        nw.initialize_world()
        var_dict = nw.world_.variables_.vars()

        # test that transforms in variable are correct
        self.assertTrue(real_key in var_dict)
        self.assertEqual(var_dict[real_key].transforms, [])

        self.assertTrue(half_key in var_dict)
        lower_bound_zero = dist.AffineTransform(0.0, 1.0)
        log_transform = dist.ExpTransform().inv
        expected_transforms = [lower_bound_zero, log_transform]
        self.assertEqual(var_dict[half_key].transforms, expected_transforms)

        self.assertTrue(simplex_key in var_dict)
        self.assertEqual(
            var_dict[simplex_key].transforms, [dist.StickBreakingTransform().inv]
        )

        self.assertTrue(interval_key in var_dict)
        lower_bound_zero = dist.AffineTransform(-1.0, 1.0)
        upper_bound_one = dist.AffineTransform(0, 1.0 / 2.0)
        beta_dimension = BetaDimensionTransform()
        stick_breaking = dist.StickBreakingTransform().inv
        expected_transforms = [
            lower_bound_zero,
            upper_bound_one,
            beta_dimension,
            stick_breaking,
        ]
        self.assertEqual(var_dict[interval_key].transforms, expected_transforms)

        self.assertTrue(beta_key in var_dict)
        lower_bound_zero = dist.AffineTransform(0.0, 1.0)
        upper_bound_one = dist.AffineTransform(0.0, 1.0)
        beta_dimension = BetaDimensionTransform()
        stick_breaking = dist.StickBreakingTransform().inv
        expected_transforms = [
            lower_bound_zero,
            upper_bound_one,
            beta_dimension,
            stick_breaking,
        ]
        self.assertEqual(var_dict[beta_key].transforms, expected_transforms)

        # test that correct proposer was used
        # and resulting shapes of proposed values are correct
        proposer = nw.find_best_single_site_proposer(real_key)
        proposed_value = proposer.propose(real_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[real_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(half_key)
        proposed_value = proposer.propose(half_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[half_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(simplex_key)
        proposed_value = proposer.propose(simplex_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[simplex_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([2]))

        proposer = nw.find_best_single_site_proposer(interval_key)
        proposed_value = proposer.propose(interval_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[interval_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(beta_key)
        proposed_value = proposer.propose(beta_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[beta_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

    def test_single_site_newtonian_monte_carlo_no_transform(self):
        model = self.SampleTransformModel()
        nw = SingleSiteNewtonianMonteCarlo(transform_type=TransformType.NONE)

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()

        nw.queries_ = [
            model.realspace(),
            model.halfspace(),
            model.simplex(),
            model.interval(),
            model.beta(),
        ]
        nw.observations_ = {}
        nw.initialize_world()
        var_dict = nw.world_.variables_.vars()

        self.assertTrue(real_key in var_dict)
        self.assertEqual(var_dict[real_key].transforms, [])

        self.assertTrue(half_key in var_dict)
        self.assertEqual(var_dict[half_key].transforms, [])

        self.assertTrue(simplex_key in var_dict)
        self.assertEqual(var_dict[simplex_key].transforms, [])

        self.assertTrue(interval_key in var_dict)
        self.assertEqual(var_dict[interval_key].transforms, [])

        self.assertTrue(beta_key in var_dict)
        self.assertEqual(var_dict[beta_key].transforms, [BetaDimensionTransform()])

        # test that resulting shapes of proposed values are correct
        proposer = nw.find_best_single_site_proposer(real_key)
        proposed_value = proposer.propose(real_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[real_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(half_key)
        proposed_value = proposer.propose(half_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[half_key],
            SingleSiteHalfSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(simplex_key)
        proposed_value = proposer.propose(simplex_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[simplex_key],
            SingleSiteSimplexNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.zeros(2).shape)

        proposer = nw.find_best_single_site_proposer(interval_key)
        proposed_value = proposer.propose(interval_key, nw.world_)[0]
        self.assertNotIn(interval_key, proposer.proposers_)
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(beta_key)
        proposed_value = proposer.propose(beta_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[beta_key], SingleSiteSimplexNewtonianMonteCarloProposer
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

    def test_single_site_newtonian_monte_carlo_transform_shape(self):
        model = self.SampleShapeModel()
        nw = SingleSiteNewtonianMonteCarlo(transform_type=TransformType.DEFAULT)

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()

        nw.queries_ = [
            model.realspace(),
            model.halfspace(),
            model.simplex(),
            model.interval(),
            model.beta(),
        ]
        nw.observations_ = {}
        nw.initialize_world()

        # test that resulting shapes of proposed values are correct
        proposer = nw.find_best_single_site_proposer(real_key)
        proposed_value = proposer.propose(real_key, nw.world_)[0]
        self.assertEqual(proposed_value.shape, torch.Size([2, 4]))

        proposer = nw.find_best_single_site_proposer(half_key)
        proposed_value = proposer.propose(half_key, nw.world_)[0]
        self.assertEqual(proposed_value.shape, torch.Size([1, 2, 4]))

        proposer = nw.find_best_single_site_proposer(simplex_key)
        proposed_value = proposer.propose(simplex_key, nw.world_)[0]
        self.assertEqual(proposed_value.shape, torch.Size([2]))

        proposer = nw.find_best_single_site_proposer(interval_key)
        proposed_value = proposer.propose(interval_key, nw.world_)[0]
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw.find_best_single_site_proposer(beta_key)
        proposed_value = proposer.propose(beta_key, nw.world_)[0]
        self.assertEqual(proposed_value.shape, torch.Size([3]))
