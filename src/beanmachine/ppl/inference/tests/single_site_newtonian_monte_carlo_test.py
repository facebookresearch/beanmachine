# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.proposer.nmc import (
    SingleSiteHalfSpaceNMCProposer,
    SingleSiteRealSpaceNMCProposer,
    SingleSiteSimplexSpaceNMCProposer,
)
from beanmachine.ppl.experimental.global_inference.single_site_nmc import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteRealSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.world import TransformType
from beanmachine.ppl.world.utils import BetaDimensionTransform, get_default_transforms
from torch import tensor


class SingleSiteNewtonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel:
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleTransformModel:
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

    class SampleShapeModel:
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

    class SampleIndependentShapeModel:
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

    class SampleStudentTModel:
        @bm.random_variable
        def x(self):
            return dist.StudentT(df=2.0)

    def test_single_site_newtonian_monte_carlo_student_t(self):
        model = self.SampleStudentTModel()
        samples = (
            SingleSiteNewtonianMonteCarlo()
            .infer(
                queries=[model.x()],
                observations={},
                num_samples=1_000,
                num_chains=1,
            )
            .get_chain(0)[model.x()]
        )

        self.assertTrue((samples.abs() > 2.0).any())

    def test_single_site_newtonian_monte_carlo_default_transform(self):
        model = self.SampleTransformModel()
        nw = bm.SingleSiteNewtonianMonteCarlo(transform_type=TransformType.DEFAULT)

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
        nw = SingleSiteNewtonianMonteCarlo()

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
        observations = {}
        world = nw._initialize_world(queries, observations)
        self.assertTrue(real_key in world)
        self.assertTrue(half_key in world)
        self.assertTrue(simplex_key in world)
        self.assertTrue(interval_key in world)
        self.assertTrue(beta_key in world)

        # trigger proposer initialization
        nw.get_proposers(world, world.latent_nodes, 0)

        # test that resulting shapes of proposed values are correct
        proposer = nw._proposers[real_key]
        proposed_value = proposer.propose(world)[0][real_key]
        self.assertIsInstance(
            proposer,
            SingleSiteRealSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw._proposers[half_key]
        proposed_value = proposer.propose(world)[0][half_key]
        self.assertIsInstance(
            proposer,
            SingleSiteHalfSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw._proposers[simplex_key]
        proposed_value = proposer.propose(world)[0][simplex_key]
        self.assertIsInstance(
            proposer,
            SingleSiteSimplexSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.zeros(2).shape)

        proposer = nw._proposers[interval_key]
        proposed_value = proposer.propose(world)[0][interval_key]
        self.assertEqual(proposed_value.shape, torch.Size([]))

        proposer = nw._proposers[beta_key]
        proposed_value = proposer.propose(world)[0][beta_key]
        self.assertIsInstance(proposer, SingleSiteSimplexSpaceNMCProposer)
        self.assertEqual(proposed_value.shape, torch.Size([]))
        self.assertEqual(
            proposer._transform,
            BetaDimensionTransform(),
        )

    def test_single_site_newtonian_monte_carlo_transform_shape(self):
        model = self.SampleShapeModel()
        nw = bm.SingleSiteNewtonianMonteCarlo(transform_type=TransformType.DEFAULT)

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
        nw = SingleSiteNewtonianMonteCarlo()

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()

        queries = [
            real_key,
            half_key,
            simplex_key,
            interval_key,
            beta_key,
        ]
        observations = {}
        world = nw._initialize_world(queries, observations)

        # trigger proposer initialization
        nw.get_proposers(world, world.latent_nodes, 0)

        # test that resulting shapes of proposed values are correct
        proposer = nw._proposers[real_key]
        proposed_value = proposer.propose(world)[0][real_key]
        self.assertIsInstance(
            proposer,
            SingleSiteRealSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([2, 4]))

        proposer = nw._proposers[half_key]
        proposed_value = proposer.propose(world)[0][half_key]
        self.assertIsInstance(
            proposer,
            SingleSiteHalfSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([1, 2, 4]))

        proposer = nw._proposers[simplex_key]
        proposed_value = proposer.propose(world)[0][simplex_key]
        self.assertIsInstance(
            proposer,
            SingleSiteSimplexSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([2, 2]))

        proposer = nw._proposers[interval_key]
        proposed_value = proposer.propose(world)[0][interval_key]
        self.assertEqual(proposed_value.shape, torch.Size([2]))

        proposer = nw._proposers[beta_key]
        proposed_value = proposer.propose(world)[0][beta_key]
        self.assertIsInstance(
            proposer,
            SingleSiteSimplexSpaceNMCProposer,
        )
        self.assertEqual(proposed_value.shape, torch.Size([3]))
