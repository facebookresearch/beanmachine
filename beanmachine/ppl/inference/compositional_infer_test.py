# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.compositional_infer import CompositionalInference
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.inference.utils import Block, BlockType
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


class CompositionalInferenceTest(unittest.TestCase):
    class SampleModel(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def foobar(self):
            return dist.Categorical(tensor([0.5, 0, 5]))

        @sample
        def foobaz(self):
            return dist.Bernoulli(0.1)

        @sample
        def bazbar(self):
            return dist.Poisson(tensor([4]))

    class SampleNormalModel(object):
        @sample
        def foo(self, i):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @sample
        def bar(self, i):
            return dist.Normal(tensor(10.0), tensor(1.0))

        @sample
        def foobar(self, i):
            return dist.Normal(self.foo(i) + self.bar(i), tensor(1.0))

    def test_single_site_compositionl_inference(self):
        model = self.SampleModel()
        c = CompositionalInference()
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
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteUniformProposer
            ),
            True,
        )

        distribution = dist.Normal(tensor(0.0), tensor(1.0))
        val = distribution.sample()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )

        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key),
                SingleSiteNewtonianMonteCarloProposer,
            ),
            True,
        )

        distribution = dist.Categorical(tensor([0.5, 0, 5]))
        val = distribution.sample()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )

        self.assertEqual(
            isinstance(
                c.find_best_single_site_proposer(foo_key), SingleSiteUniformProposer
            ),
            True,
        )

        distribution = dist.Poisson(tensor([4.0]))
        val = distribution.sample()
        world_vars[foo_key] = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
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
        c = CompositionalInference({model.foo: SingleSiteAncestralProposer()})
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
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
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
        ci = CompositionalInference()
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
                    [foo_0_key.function._wrapper, bar_0_key.function._wrapper],
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
                block=[foo_0_key.function._wrapper, bar_0_key.function._wrapper],
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
