# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from unittest.mock import patch

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.inference.utils import Block, BlockType
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.model.utils import get_wrapper
from beanmachine.ppl.world.utils import BetaDimensionTransform
from beanmachine.ppl.world.variable import TransformType, Variable
from beanmachine.ppl.world.world import World


class CompositionalInferenceTest(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def foobar(self):
            return dist.Categorical(tensor([0.5, 0, 5]))

        @bm.random_variable
        def foobaz(self):
            return dist.Bernoulli(0.1)

        @bm.random_variable
        def bazbar(self):
            return dist.Poisson(tensor([4]))

    class SampleNormalModel(object):
        @bm.random_variable
        def foo(self, i):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self, i):
            return dist.Normal(tensor(10.0), tensor(1.0))

        @bm.random_variable
        def foobar(self, i):
            return dist.Normal(self.foo(i) + self.bar(i), tensor(1.0))

    class ChangingSupportSameShapeModel(object):
        # the support of `component` is changing, but (because we indexed alpha
        # by k) all random_variables have the same shape
        @bm.random_variable
        def K(self):
            return dist.Poisson(rate=2.0)

        @bm.random_variable
        def alpha(self, k):
            return dist.Dirichlet(torch.ones(k))

        @bm.random_variable
        def component(self, i):
            alpha = self.alpha(self.K().int().item() + 1)
            return dist.Categorical(alpha)

    class ChangingShapeModel(object):
        # here since we did not index alpha, its shape in each world is changing
        @bm.random_variable
        def K(self):
            return dist.Poisson(rate=2.0)

        @bm.random_variable
        def alpha(self):
            return dist.Dirichlet(torch.ones(self.K().int().item() + 1))

        @bm.random_variable
        def component(self, i):
            return dist.Categorical(self.alpha())

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

        @sample
        def discrete(self):
            return dist.Poisson(tensor(2.0))

    def test_single_site_compositionl_inference(self):
        model = self.SampleModel()
        c = bm.CompositionalInference()
        foo_key = model.foo()
        c.world_ = World()
        distribution = dist.Bernoulli(0.1)
        val = distribution.sample()
        world_vars = c.world_.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteUniformProposer
            ),
            True,
        )

        c.proposers_per_rv_ = {}
        distribution = dist.Normal(tensor(0.0), tensor(1.0))
        val = distribution.sample()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )

        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key),
                SingleSiteNewtonianMonteCarloProposer,
            ),
            True,
        )

        c.proposers_per_rv_ = {}
        distribution = dist.Categorical(tensor([0.5, 0, 5]))
        val = distribution.sample()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )

        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteUniformProposer
            ),
            True,
        )

        c.proposers_per_rv_ = {}
        distribution = dist.Poisson(tensor([4.0]))
        val = distribution.sample()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteAncestralProposer
            ),
            True,
        )

    def test_single_site_compositionl_inference_with_input(self):
        model = self.SampleModel()
        c = bm.CompositionalInference({model.foo: SingleSiteAncestralProposer()})
        foo_key = model.foo()
        c.world_ = World()
        distribution = dist.Normal(0.1, 1)
        val = distribution.sample()

        world_vars = c.world_.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteAncestralProposer
            ),
            True,
        )

    def test_proposer_for_block(self):
        model = self.SampleNormalModel()
        ci = bm.CompositionalInference()
        ci.add_sequential_proposer([model.foo, model.bar])
        ci.queries_ = [
            model.foo(0),
            model.foo(1),
            model.foo(2),
            model.bar(0),
            model.bar(1),
            model.bar(2),
        ]
        ci.observations_ = {
            model.foobar(0): tensor(0.0),
            model.foobar(1): tensor(0.1),
            model.foobar(2): tensor(0.11),
        }

        foo_0_key = model.foo(0)
        foo_1_key = model.foo(1)
        foo_2_key = model.foo(2)
        bar_0_key = model.bar(0)
        foobar_0_key = model.foobar(0)

        ci._infer(2)
        blocks = ci.process_blocks()
        self.assertEqual(len(blocks), 9)
        first_nodes = []
        for block in blocks:
            if block.type == BlockType.SEQUENTIAL:
                first_nodes.append(block.first_node)
                self.assertEqual(
                    block.block,
                    list(map(get_wrapper, [foo_0_key.function, bar_0_key.function])),
                )
            if block.type == BlockType.SINGLENODE:
                self.assertEqual(block.block, [])

        self.assertTrue(foo_0_key in first_nodes)
        self.assertTrue(foo_1_key in first_nodes)
        self.assertTrue(foo_2_key in first_nodes)

        nodes_log_updates, children_log_updates, _ = ci.block_propose_change(
            Block(
                first_node=foo_0_key,
                type=BlockType.SEQUENTIAL,
                block=list(map(get_wrapper, [foo_0_key.function, bar_0_key.function])),
            )
        )

        diff_level_1 = ci.world_.diff_stack_.diff_stack_[-2]
        diff_level_2 = ci.world_.diff_stack_.diff_stack_[-1]

        self.assertEqual(diff_level_1.contains_node(foo_0_key), True)
        self.assertEqual(diff_level_1.contains_node(foobar_0_key), True)
        self.assertEqual(diff_level_2.contains_node(bar_0_key), True)
        self.assertEqual(diff_level_2.contains_node(foobar_0_key), True)

        expected_node_log_updates = (
            ci.world_.diff_stack_.get_node(foo_0_key).log_prob
            - ci.world_.variables_.get_node(foo_0_key).log_prob
        )

        expected_node_log_updates += (
            ci.world_.diff_stack_.get_node(bar_0_key).log_prob
            - ci.world_.variables_.get_node(bar_0_key).log_prob
        )

        expected_children_log_updates = (
            ci.world_.diff_stack_.get_node(foobar_0_key).log_prob
            - ci.world_.variables_.get_node(foobar_0_key).log_prob
        )

        self.assertAlmostEqual(
            expected_node_log_updates.item(), nodes_log_updates.item(), delta=0.001
        )
        self.assertAlmostEqual(
            expected_children_log_updates.item(),
            children_log_updates.item(),
            delta=0.001,
        )

    def test_block_inference_on_functions(self):
        @bm.random_variable
        def foo():
            return dist.Normal(0, 1)

        ci = bm.CompositionalInference()
        ci.add_sequential_proposer([foo])

        with patch.object(
            ci, "block_propose_change", wraps=ci.block_propose_change
        ) as mock:
            ci.infer([foo()], {}, num_samples=2)
            mock.assert_called()

    def test_block_inference_changing_support(self):
        torch.manual_seed(41)
        model = self.ChangingSupportSameShapeModel()
        queries = [model.K()] + [model.component(j) for j in range(10)]
        mh = bm.CompositionalInference()
        mh.add_sequential_proposer([model.K, model.component])
        with patch.object(
            mh, "block_propose_change", wraps=mh.block_propose_change
        ) as block_propose_spy:
            mh.infer(queries, {}, num_samples=10, num_chains=1)
            block_propose_spy.assert_called()

    def test_block_inference_changing_shape(self):
        model = self.ChangingShapeModel()
        queries = [model.K()] + [model.component(j) for j in range(10)]
        mh = bm.CompositionalInference()

        # TODO: we should never raise RuntimeError, blocked by T67717820
        with self.assertRaises(RuntimeError):
            mh.infer(queries, {}, num_samples=10, num_chains=1)

    def test_single_site_compositional_inference_transform_default(self):
        model = self.SampleTransformModel()
        ci = bm.CompositionalInference(
            {
                model.realspace: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.DEFAULT
                ),
                model.halfspace: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.DEFAULT
                ),
                model.simplex: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.DEFAULT
                ),
                model.interval: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.DEFAULT
                ),
                model.beta: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.DEFAULT
                ),
            }
        )

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()

        ci.queries_ = [
            model.realspace(),
            model.halfspace(),
            model.simplex(),
            model.interval(),
            model.beta(),
        ]
        ci.observations_ = {}
        ci.initialize_world()
        var_dict = ci.world_.variables_.vars()

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

    def test_single_site_compositional_inference_transform_mixed(self):
        model = self.SampleTransformModel()
        ci = bm.CompositionalInference(
            {
                model.realspace: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.CUSTOM,
                    transforms=[dist.ExpTransform()],
                ),
                model.halfspace: SingleSiteHamiltonianMonteCarloProposer(
                    0.1, 10, transform_type=TransformType.DEFAULT
                ),
                model.simplex: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.NONE
                ),
                model.interval: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.CUSTOM,
                    transforms=[dist.AffineTransform(1.0, 2.0)],
                ),
                model.beta: SingleSiteNewtonianMonteCarloProposer(
                    transform_type=TransformType.NONE
                ),
                model.discrete: SingleSiteUniformProposer(
                    transform_type=TransformType.NONE
                ),
            }
        )

        real_key = model.realspace()
        half_key = model.halfspace()
        simplex_key = model.simplex()
        interval_key = model.interval()
        beta_key = model.beta()
        discrete_key = model.discrete()

        ci.queries_ = [
            model.realspace(),
            model.halfspace(),
            model.simplex(),
            model.interval(),
            model.beta(),
            model.discrete(),
        ]
        ci.observations_ = {}
        ci.initialize_world()
        var_dict = ci.world_.variables_.vars()

        self.assertTrue(real_key in var_dict)
        self.assertEqual(var_dict[real_key].transforms, [dist.ExpTransform()])

        self.assertTrue(half_key in var_dict)
        lower_bound_zero = dist.AffineTransform(0.0, 1.0)
        log_transform = dist.ExpTransform().inv
        expected_transforms = [lower_bound_zero, log_transform]
        self.assertEqual(var_dict[half_key].transforms, expected_transforms)

        self.assertTrue(simplex_key in var_dict)
        self.assertEqual(var_dict[simplex_key].transforms, [])

        self.assertTrue(interval_key in var_dict)
        self.assertEqual(
            var_dict[interval_key].transforms, [dist.AffineTransform(1.0, 2.0)]
        )

        self.assertTrue(beta_key in var_dict)
        self.assertEqual(var_dict[beta_key].transforms, [BetaDimensionTransform()])

        self.assertTrue(discrete_key in var_dict)
        self.assertEqual(var_dict[discrete_key].transforms, [])

    def test_single_site_compositional_inference_ancestral_beta(self):
        model = self.SampleTransformModel()
        ci = bm.CompositionalInference(
            {model.beta: SingleSiteAncestralProposer(transform_type=TransformType.NONE)}
        )

        beta_key = model.beta()

        ci.queries_ = [model.beta()]
        ci.observations_ = {}
        ci.initialize_world()
        var_dict = ci.world_.variables_.vars()

        self.assertTrue(beta_key in var_dict)
        self.assertEqual(var_dict[beta_key].transforms, [])
