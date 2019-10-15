# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RandomVariable


class SingleSiteAncestralMetropolisHastings(AbstractSingleSiteMHInference):
    """
    Implementation for SingleSiteAncestralMetropolisHastings
    """

    def __init__(self):
        super().__init__()
        self.proposer_ = SingleSiteAncestralProposer(self.world_)

    def find_best_single_site_proposer(self, node: RandomVariable):
        """
        Finds the best proposer for a node which is
        SingleSiteAncestralMetropolisHastingsProposer for
        SingleSiteAncestralMetropolisHastings

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
