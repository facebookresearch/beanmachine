# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

from beanmachine.ppl.inference.proposer.abstract_single_site_single_step_proposer import (
    AbstractSingleSiteSingleStepProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World


class SingleSiteAncestralProposer(AbstractSingleSiteSingleStepProposer):
    """
    Single-Site Ancestral Metropolis Hastings Implementations
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
        return (
            ProposalDistribution(
                proposal_distribution=node_var.distribution,
                requires_transform=False,
                requires_reshape=False,
                arguments={},
            ),
            {},
        )
