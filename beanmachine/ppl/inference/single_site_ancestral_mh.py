# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteAncestralMetropolisHastings(AbstractMHInference):
    """
    Implementation for SingleSiteAncestralMetropolisHastings
    """

    def __init__(self):
        self.proposer_ = SingleSiteAncestralProposer()
        super().__init__(proposer=self.proposer_)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteAncestralMetropolisHastingsProposer for
        SingleSiteAncestralMetropolisHastings

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
