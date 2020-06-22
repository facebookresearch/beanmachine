# Copyright (c) Facebook, Inc. and its affiliates
import unittest
from typing import Dict, Tuple

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.abstract_single_site_single_step_proposer import (
    AbstractSingleSiteSingleStepProposer,
)
from beanmachine.ppl.inference.proposer.single_site_half_space_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World


class SingleSiteCustomProposerTest(unittest.TestCase):
    class CustomProposer(AbstractSingleSiteSingleStepProposer):
        def get_proposal_distribution(
            self,
            node: RVIdentifier,
            node_var: Variable,
            world: World,
            additional_args: Dict,
        ) -> Tuple[ProposalDistribution, Dict]:

            if node.function.__name__ == "foo":
                bar_node = list(node_var.children)[0]
                bar_node_var = world.get_node_in_world_raise_error(bar_node, False)
                return (
                    ProposalDistribution(
                        proposal_distribution=dist.Gamma(
                            bar_node_var.value, tensor(1.0)
                        ),
                        requires_transform=False,
                        requires_reshape=False,
                        arguments={},
                    ),
                    {},
                )
            else:
                return (
                    ProposalDistribution(
                        proposal_distribution=node_var.distribution,
                        requires_transform=False,
                        requires_reshape=False,
                        arguments={},
                    ),
                    {},
                )

    class SampleGammaModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Gamma(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Gamma(self.foo(), torch.tensor(1.0))

    def test_custom_proposer(self):
        model = self.SampleGammaModel()
        ci = bm.CompositionalInference({model.foo: self.CustomProposer()})
        ci.queries_ = [model.foo()]
        ci.observations_ = {model.bar(): tensor(2.0)}
        ci._infer(2)
        world_vars = ci.world_.variables_.vars()
        for key in world_vars:
            if key.function.__name__ == "foo":
                proposer = world_vars[key].proposal_distribution.proposal_distribution
                requires_transform = world_vars[
                    key
                ].proposal_distribution.requires_transform
                requires_reshape = world_vars[
                    key
                ].proposal_distribution.requires_reshape
                self.assertEqual(proposer.concentration.item(), 2.0)
                self.assertEqual(proposer.rate.item(), 1.0)
                self.assertEqual(requires_transform, False)
                self.assertEqual(requires_reshape, False)

    def test_fallback_to_ancestral(self):
        model = self.SampleGammaModel()
        # force invalid gradient during halfspace proposal
        proposer = SingleSiteHalfSpaceNewtonianMonteCarloProposer()
        ci = bm.CompositionalInference({model.foo: proposer})
        foo_key = model.foo()
        ci.queries_ = [model.foo()]
        ci.observations_ = {model.bar(): tensor(-1.0)}
        ci.initialize_world()
        node_var = ci.world_.get_node_in_world_raise_error(foo_key, False)
        world = ci.world_
        (
            proposal_distribution_struct,
            auxiliary_variables,
        ) = proposer.get_proposal_distribution(foo_key, node_var, world, {})

        # test that there is no transform after fallback to ancestral MH
        self.assertEqual(proposal_distribution_struct.requires_transform, False)

        # valid gradient during halfspace proposal
        proposer = SingleSiteHalfSpaceNewtonianMonteCarloProposer()
        ci = bm.CompositionalInference({model.foo: proposer})
        foo_key = model.foo()
        ci.queries_ = [model.foo()]
        ci.observations_ = {model.bar(): tensor(2.0)}
        ci.initialize_world()
        world = ci.world_
        (
            proposal_distribution_struct,
            auxiliary_variables,
        ) = proposer.get_proposal_distribution(foo_key, node_var, world, {})
        # test that there is transform without fallback to ancestral MH
        self.assertEqual(proposal_distribution_struct.requires_transform, True)
