# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

from beanmachine.ppl.inference.proposer.abstract_proposer import AbstractProposer
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor


class SingleSiteAncestralProposer(AbstractProposer):
    """
    Single-Site Ancestral Metropolis Hastings Implementations
    """

    def __init__(self, world):
        super().__init__(world)

    def propose(self, node: RVIdentifier) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node. In Single-Site Ancestral Metropolis
        Hastings, we just need to draw a new sample from the node's distribution.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        # the node Variable from the world (no diff is created here yet)
        node_var = self.world_.get_node_in_world(node, False)
        new_value = node_var.distribution.sample()
        negative_proposal_log_update = (
            -1 * node_var.distribution.log_prob(new_value).sum()
        )
        node_var.proposal_distribution = node_var.distribution

        return (new_value, negative_proposal_log_update)

    def post_process(self, node: RVIdentifier) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        # the probability of proposing the old value in the new world
        return self.world_.variables_[node].log_prob
