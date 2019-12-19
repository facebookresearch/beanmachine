# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.inference.proposer.single_site_random_walk_proposer import (
    SingleSiteRandomWalkProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteRandomWalk(AbstractSingleSiteMHInference):
    """
    Implementation for SingleSiteNewtonianMonteCarlo
    """

    def __init__(self, step_size: float = 1.0):
        super().__init__()
        self.proposer_ = SingleSiteRandomWalkProposer(step_size)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteRandomWalkProposer for
        SingleSiteRandomWalk

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
