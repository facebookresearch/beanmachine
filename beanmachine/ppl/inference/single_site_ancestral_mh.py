# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.model.utils import RandomVariable
from torch import Tensor


class SingleSiteAncestralMetropolisHastings(AbstractSingleSiteMHInference):
    """
    Single-Site Ancestral Metropolis Hastings Implementations
    """

    def __init__(self):
        super().__init__()

    def propose(self, node: RandomVariable) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node. In Single-Site Ancestral Metropolis
        Hastings, we just need to draw a new sample from the node's distribution.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node = self.world_.get_node_in_world(node, False)
        new_value = node.distribution.sample()
        negative_proposal_log_update = -1 * node.distribution.log_prob(new_value).sum()
        return (new_value, negative_proposal_log_update)

    def post_process(self, node: RandomVariable) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        return self.world_.variables_[node].log_prob
