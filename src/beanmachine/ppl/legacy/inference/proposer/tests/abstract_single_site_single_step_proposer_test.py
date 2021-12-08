# Copyright (c) Facebook, Inc. and its affiliates
import unittest
from typing import Dict, List, Optional, Tuple

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.legacy.inference import CompositionalInference
from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.abstract_single_site_single_step_proposer import (
    AbstractSingleSiteSingleStepProposer,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_half_space_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.world import (
    ProposalDistribution,
    Variable,
    World,
    TransformType,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import tensor


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
            # validation turned off to test fallback behavior
            return dist.Gamma(self.foo(), torch.tensor(1.0), validate_args=False)

    def test_custom_proposer(self):
        model = self.SampleGammaModel()
        ci = CompositionalInference({model.foo: self.CustomProposer()})
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
        ci = CompositionalInference({model.foo: proposer})
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
        ci = CompositionalInference({model.foo: proposer})
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

    class CustomTransformProposer(AbstractSingleSiteSingleStepProposer):
        def get_proposal_distribution(
            self,
            node: RVIdentifier,
            node_var: Variable,
            world: World,
            auxiliary_variables: Dict,
        ) -> Tuple[ProposalDistribution, Dict]:
            return (
                ProposalDistribution(
                    proposal_distribution=dist.TransformedDistribution(
                        node_var.distribution, dist.ExpTransform().inv
                    ),
                    requires_transform=True,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )

    class CustomInference(AbstractMHInference):
        def __init__(
            self,
            proposer,
            transform_type: TransformType = TransformType.DEFAULT,
            transforms: Optional[List] = None,
        ):
            super().__init__(
                proposer, transform_type=transform_type, transforms=transforms
            )
            self.proposer_ = proposer

        def find_best_single_site_proposer(self, node: RVIdentifier):
            return self.proposer_

    def test_transform_for_single_site_single_step_proposer(self):
        model = self.SampleGammaModel()
        infer = self.CustomInference(self.CustomTransformProposer)
        foo_key = model.foo()
        infer.queries_ = [model.foo()]
        infer.initialize_world(initialize_from_prior=True)
        is_accepted, acceptance_probability = infer.single_inference_run(
            foo_key, self.CustomTransformProposer()
        )
        self.assertEqual(is_accepted, True)
        self.assertAlmostEqual(acceptance_probability, 1.0, delta=1e-5)
