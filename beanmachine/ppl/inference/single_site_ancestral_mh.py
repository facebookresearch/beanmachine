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
        :returns: a new proposed value for the node and the difference in log_prob
        of the old and newly proposed value.
        """
        node = self.world_.get_node_in_world(node, False)
        new_value = node.distribution.sample()
        proposal_log_update = (
            node.log_prob - node.distribution.log_prob(new_value).sum()
        )
        return (new_value, proposal_log_update)
