# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
    SingleSiteNewtonianMonteCarloProposer,
    SingleSiteRealSpaceNewtonianMonteCarloProposer,
    SingleSiteSimplexNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.world import TransformType
from beanmachine.ppl.world.utils import BetaDimensionTransform, get_default_transforms
from torch import tensor


class SingleSiteNewtonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleTransformModel(object):
        @bm.random_variable
        def realspace(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def halfspace(self):
            return dist.Gamma(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def simplex(self):
            return dist.Dirichlet(tensor([0.1, 0.9]))

        @bm.random_variable
        def interval(self):
            return dist.Uniform(tensor(1.0), tensor(3.0))

        @bm.random_variable
        def beta(self):
            return dist.Beta(tensor(1.0), tensor(1.0))

    class SampleShapeModel(object):
        @bm.random_variable
        def realspace(self):
            return dist.Normal(torch.zeros(2, 4), tensor(1.0))

        @bm.random_variable
        def halfspace(self):
            return dist.Gamma(torch.zeros(1, 2, 4) + tensor(2.0), tensor(2.0))

        @bm.random_variable
        def simplex(self):
            return dist.Dirichlet(tensor([0.1, 0.9]))

        @bm.random_variable
        def interval(self):
            return dist.Uniform(tensor(1.0), tensor(3.0))

        @bm.random_variable
        def beta(self):
            return dist.Beta(tensor([1.0, 2.0, 3.0]), tensor([1.0, 2.0, 3.0]))

    class SampleIndependentShapeModel(object):
        @bm.random_variable
        def realspace(self):
            return dist.Independent(dist.Normal(torch.zeros(2, 4), tensor(1.0)), 1)

        @bm.random_variable
        def halfspace(self):
            return dist.Independent(
                dist.Gamma(torch.zeros(1, 2, 4) + tensor(2.0), tensor(2.0)), 1
            )

        @bm.random_variable
        def simplex(self):
            return dist.Independent(dist.Dirichlet(tensor([[0.1, 0.9], [0.1, 0.9]])), 1)

        @bm.random_variable
        def interval(self):
            return dist.Independent(
                dist.Uniform(tensor([1.0, 1.0]), tensor([3.0, 3.0])), 1
            )

        @bm.random_variable
        def beta(self):
            return dist.Independent(
                dist.Beta(tensor([1.0, 2.0, 3.0]), tensor([1.0, 2.0, 3.0])), 1
            )

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

        queries = [
            model.realspace(),
            model.halfspace(),
            model.simplex(),
            model.interval(),
            model.beta(),
        ]
        nw.queries_ = queries
        nw.observations_ = {}
        nw.initialize_world()
        var_dict = nw.world_.variables_.vars()

        # test that transforms in variable are correct
        for key in queries:
            self.assertIn(key, var_dict)
            self.assertEqual(
                var_dict[key].transform,
                get_default_transforms(var_dict[key].distribution),
            )

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

        identity_transform = dist.transforms.identity_transform

        self.assertTrue(real_key in var_dict)
        self.assertEqual(var_dict[real_key].transform, identity_transform)

        self.assertTrue(half_key in var_dict)
        self.assertEqual(var_dict[half_key].transform, identity_transform)

        self.assertTrue(simplex_key in var_dict)
        self.assertEqual(var_dict[simplex_key].transform, identity_transform)

        self.assertTrue(interval_key in var_dict)
        self.assertEqual(var_dict[interval_key].transform, identity_transform)

        self.assertTrue(beta_key in var_dict)
        self.assertEqual(
            var_dict[beta_key].transform, BetaDimensionTransform(),
        )

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

    def test_single_site_newtonian_monte_carlo_no_transform_independent_shape(self):
        model = self.SampleIndependentShapeModel()
        nw = SingleSiteNewtonianMonteCarlo(transform_type=TransformType.NONE)

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()

        nw.queries_ = [
            real_key,
            half_key,
            simplex_key,
            interval_key,
            beta_key,
        ]
        nw.observations_ = {}
        nw.initialize_world()

        # test that resulting shapes of proposed values are correct
        proposer = nw.find_best_single_site_proposer(real_key)
        proposed_value = proposer.propose(real_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[real_key],
            SingleSiteRealSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([2, 4]))

        proposer = nw.find_best_single_site_proposer(half_key)
        proposed_value = proposer.propose(half_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[half_key],
            SingleSiteHalfSpaceNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([1, 2, 4]))

        proposer = nw.find_best_single_site_proposer(simplex_key)
        proposed_value = proposer.propose(simplex_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[simplex_key],
            SingleSiteSimplexNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([2, 2]))

        proposer = nw.find_best_single_site_proposer(interval_key)
        proposed_value = proposer.propose(interval_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[interval_key],
            type(
                super(
                    SingleSiteNewtonianMonteCarloProposer,
                    SingleSiteNewtonianMonteCarloProposer,
                )
            ),
        )
        self.assertEqual(proposed_value.shape, torch.Size([2]))

        proposer = nw.find_best_single_site_proposer(beta_key)
        proposed_value = proposer.propose(beta_key, nw.world_)[0]
        self.assertIsInstance(
            proposer.proposers_[beta_key], SingleSiteSimplexNewtonianMonteCarloProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([3]))
