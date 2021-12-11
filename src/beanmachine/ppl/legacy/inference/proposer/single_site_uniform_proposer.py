# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.world import ProposalDistribution, Variable, World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.utils import is_constraint_eq


class SingleSiteUniformProposer(SingleSiteAncestralProposer):
    """
    Single-Site Uniform Metropolis Hastings Implementations

    For random variables with Bernoulli and Categorical distributions, returns a
    sample from their distribution with equal probability across all values. For
    the rest of the random variables, it returns ancestral metropolis hastings
    proposal.
    """

    def get_proposal_distribution(
        self,
        node: RVIdentifier,
        node_var: Variable,
        world: World,
        auxiliary_variables: Dict,
    ) -> Tuple[ProposalDistribution, Dict]:
        """
        Returns the proposal distribution of the node.

        :param node: the node for which we're proposing a new value for
        :param node_var: the Variable of the node
        :param world: the world in which we're proposing a new value for node
        :param auxiliary_variables: additional auxiliary variables that may be
        required to find a proposal distribution
        :returns: the tuple of proposal distribution of the node and arguments
        that was used or needs to be used to find the proposal distribution
        """
        node_distribution = node_var.distribution
        if (
            is_constraint_eq(
                # pyre-fixme
                node_distribution.support,
                dist.constraints.boolean,
            )
            and isinstance(node_distribution, dist.Bernoulli)
        ):
            return (
                ProposalDistribution(
                    proposal_distribution=dist.Bernoulli(
                        torch.ones(node_distribution.param_shape) / 2.0
                    ),
                    requires_transform=False,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )
        if is_constraint_eq(
            node_distribution.support, dist.constraints.integer_interval
        ) and isinstance(node_distribution, dist.Categorical):
            probs = torch.ones(node_distribution.param_shape)
            # In Categorical distrbution, the samples are integers from 0-k
            # where K is probs.size(-1).
            probs /= float(node_distribution.param_shape[-1])
            distribution = dist.Categorical(probs)
            return (
                ProposalDistribution(
                    proposal_distribution=distribution,
                    requires_transform=False,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )
        return super().get_proposal_distribution(
            node, node_var, world, auxiliary_variables
        )
