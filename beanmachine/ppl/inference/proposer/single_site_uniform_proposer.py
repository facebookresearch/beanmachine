# Copyright (c) Facebook, Inc. and its affiliates
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World


class SingleSiteUniformProposer(SingleSiteAncestralProposer):
    """
    Single-Site Uniform Metropolis Hastings Implementations

    For random variables with Bernoulli and Categorical distributions, returns a
    sample from their distribution with equal probability across all values. For
    the rest of the random variables, it returns ancestral metropolis hastings
    proposal.
    """

    def get_proposal_distribution(
        self, node: RVIdentifier, node_var: Variable, world: World
    ) -> ProposalDistribution:
        """
        Returns the proposal distribution of the node.

        :param node: the node for which we're proposing a new value for
        :param node_var: the Variable of the node
        :param world: the world in which we're proposing a new value for node
        :returns: the proposal distribution of the node
        """
        node_distribution = node_var.distribution
        if isinstance(
            # pyre-fixme
            node_distribution.support,
            dist.constraints._Boolean,
        ) and isinstance(node_distribution, dist.Bernoulli):
            return ProposalDistribution(
                proposal_distribution=dist.Bernoulli(
                    torch.ones(node_distribution.param_shape) / 2.0
                ),
                requires_transform=False,
                requires_reshape=False,
            )
        if isinstance(
            node_distribution.support, dist.constraints._IntegerInterval
        ) and isinstance(node_distribution, dist.Categorical):
            probs = torch.ones(node_distribution.param_shape)
            # In Categorical distrbution, the samples are integers from 0-k
            # where K is probs.size(-1).
            probs /= float(node_distribution.param_shape[-1])
            distribution = dist.Categorical(probs)
            return ProposalDistribution(
                proposal_distribution=distribution,
                requires_transform=False,
                requires_reshape=False,
            )
        return super().get_proposal_distribution(node, node_var, world)
