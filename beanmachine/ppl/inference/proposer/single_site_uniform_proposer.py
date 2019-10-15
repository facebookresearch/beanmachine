# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RandomVariable
from torch import Tensor


class SingleSiteUniformProposer(SingleSiteAncestralProposer):
    """
    Single-Site Uniform Metropolis Hastings Implementations

    For random variables with Bernoulli and Categorical distributions, returns a
    sample from their distribution with equal probability across all values. For
    the rest of the random variables, it returns ancestral metropolis hastings
    proposal.
    """

    def __init__(self, world):
        super().__init__(world)

    def propose(self, node: RandomVariable) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node_var = self.world_.get_node_in_world(node, False)
        node_distribution = node_var.distribution
        if isinstance(
            node_distribution.support, dist.constraints._Boolean
        ) and isinstance(node_distribution, dist.Bernoulli):
            distribution = dist.Bernoulli(
                torch.ones(node_distribution.param_shape) / 2.0
            )
            new_value = distribution.sample()
            return (new_value, -1 * distribution.log_prob(new_value))
        elif isinstance(
            node_distribution.support, dist.constraints._IntegerInterval
        ) and isinstance(node_distribution, dist.Categorical):
            probs = torch.ones(node_distribution.param_shape)
            # In Categorical distrbution, the samples are integers from 0-k
            # where K is probs.size(-1).
            probs /= float(node_distribution.param_shape[-1])
            distribution = dist.Categorical(probs)
            new_value = distribution.sample()
            return (new_value, -1 * distribution.log_prob(new_value))
        else:
            return super().propose(node)

    def post_process(self, node: RandomVariable) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :returns: the log probability of proposing the old value from this new world.
        """
        node_var = self.world_.get_node_in_world(node, False)
        node_distribution = node_var.distribution
        if isinstance(
            node_distribution.support, dist.constraints._Boolean
        ) and isinstance(node_distribution, dist.Bernoulli):
            distribution = dist.Bernoulli(
                torch.ones(node_distribution.param_shape) / 2.0
            )
            return distribution.log_prob(self.world_.variables_[node].value)
        if isinstance(
            node_distribution.support, dist.constraints._IntegerInterval
        ) and isinstance(node_distribution, dist.Categorical):
            probs = torch.ones(node_distribution.param_shape)
            # In Categorical distrbution, the samples are integers from 0-k
            # where K is probs.size(-1).
            probs /= float(node_distribution.param_shape[-1])
            distribution = dist.Categorical(probs)
            return distribution.log_prob(self.world_.variables_[node].value)
        else:
            return super().post_process(node)
