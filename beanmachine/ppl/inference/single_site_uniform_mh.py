# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteUniformMetropolisHastings(AbstractSingleSiteMHInference):
    """
    Implementation for SingleSiteNewtonianMonteCarlo
    """

    def __init__(self):
        super().__init__()
        self.proposer_ = SingleSiteUniformProposer()

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteUniformMetropolisHastingsProposer for
        SingleSiteUniformMetropolisHastings

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
