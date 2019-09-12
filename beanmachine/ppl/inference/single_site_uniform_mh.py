# Copyright (c) Facebook, Inc. and its affiliates
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)


class SingleSiteUniformMetropolisHastings(SingleSiteAncestralMetropolisHastings):
    """
    Single-Site Uniform Metropolis Hastings Implementations

    For random variables with Bernoulli and Categorical distributions, returns a
    sample from their distribution with equal probability across all values. For
    the rest of the random variables, it returns ancestral metropolis hastings
    proposal.
    """

    def __init__(self):
        super().__init__()

    def propose(self, node):
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the log of the proposal
        ratio
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
            return (new_value, tensor(0.0))
        elif isinstance(
            node_distribution.support, dist.constraints._IntegerInterval
        ) and isinstance(node_distribution, dist.Categorical):
            probs = torch.ones(node_distribution.param_shape)
            # In Categorical distrbution, the samples are integers from 0-k
            # where K is probs.size(-1).
            probs /= float(node_distribution.param_shape[-1])
            distribution = dist.Categorical(probs)
            new_value = distribution.sample()
            return (new_value, tensor(0.0))
        else:
            return super().propose(node)
