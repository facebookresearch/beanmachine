# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor


class SingleSiteRandomWalkProposer(SingleSiteAncestralProposer):
    """
    Single-Site Random Walk Metropolis Hastings Implementations

    For random variables with continuous support distributions, returns a
    sample with a perturbation added to the current variable.
    For the rest of the random variables, it returns ancestral metropolis hastings
    proposal.
    """

    def __init__(self, world, step_size: float = 1.0):
        """
        Initialize object.

        :param step_size: Standard Deviation of the noise used for Diff proposals.
        """
        self.step_size = step_size
        super().__init__(world)

    def propose(self, node: RVIdentifier) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the log of the proposal
        ratio
        """
        node_var = self.world_.get_node_in_world(node, False)
        node_distribution = node_var.distribution

        # Full real number line
        if isinstance(node_distribution.support, dist.constraints._Real):
            distribution = dist.Normal(
                node_var.value, torch.ones(node_var.value.shape) * self.step_size
            )
            new_value = distribution.sample()

            negative_proposal_log_update = -1 * distribution.log_prob(new_value).sum()
            return (new_value, negative_proposal_log_update)
        # Half number line
        # Does not yet support PositiveDefinite or LessThan (TODO?)
        if isinstance(
            node_distribution.support, dist.constraints._GreaterThan
        ) or isinstance(node_distribution.support, dist.constraints._GreaterThanEq):
            lower_bound = node_distribution.support.lower_bound
            proposal_distribution = self.gamma_distbn_from_moments(
                node_var.value - lower_bound, self.step_size ** 2
            )
            node_var.proposal_distribution = proposal_distribution

            new_sample = proposal_distribution.sample()
            negative_proposal_log_update = (
                -1 * proposal_distribution.log_prob(new_sample).sum()
            )

            new_value = new_sample + lower_bound
            return (new_value, negative_proposal_log_update)
        return super().propose(node)

    def post_process(self, node: RVIdentifier) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        node_var = self.world_.get_node_in_world(node, False)
        old_node_var = self.world_.variables_[node]
        node_distribution = node_var.distribution

        if isinstance(node_distribution.support, dist.constraints._Real):
            distribution = dist.Normal(
                node_var.value, torch.ones(node_var.value.shape) * self.step_size
            )
            positive_proposal_log_update = distribution.log_prob(
                old_node_var.value
            ).sum()
            return positive_proposal_log_update

        if isinstance(
            node_distribution.support, dist.constraints._GreaterThan
        ) or isinstance(node_distribution.support, dist.constraints._GreaterThanEq):
            lower_bound = node_distribution.support.lower_bound
            node_var.proposal_distribution = self.gamma_distbn_from_moments(
                node_var.value - lower_bound, self.step_size ** 2
            )

            positive_proposal_log_update = node_var.proposal_distribution.log_prob(
                old_node_var.value - lower_bound
            ).sum()
            return positive_proposal_log_update

        return super().post_process(node)

    def gamma_distbn_from_moments(self, expectation, variance):
        beta = expectation / variance
        alpha = expectation * beta
        distribution = dist.Gamma(concentration=alpha, rate=beta)
        return distribution
