# Copyright (c) Facebook, Inc. and its affiliates
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
        self, node: RVIdentifier, node_var: Variable, world: World
    ) -> ProposalDistribution:
        """
        Returns the proposal distribution of the node.

        :param node: the node for which we're proposing a new value for
        :param node_var: the Variable of the node
        :param world: the world in which we're proposing a new value for node
        :returns: the proposal distribution of the node
        """
        return ProposalDistribution(
            proposal_distribution=node_var.distribution,
            requires_transform=False,
            requires_reshape=False,
        )
