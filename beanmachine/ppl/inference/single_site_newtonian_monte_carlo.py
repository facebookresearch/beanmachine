# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteNewtonianMonteCarlo(AbstractSingleSiteMHInference):
    """
    Implementation for SingleSiteNewtonianMonteCarlo
    """

    def __init__(
        self,
        use_transform_: bool = False,
        real_space_alpha: float = 10.0,
        real_space_beta: float = 1.0,
    ):
        super().__init__()
        self.world_.set_all_nodes_transform(use_transform_)
        self.proposer_ = SingleSiteNewtonianMonteCarloProposer(
            real_space_alpha, real_space_beta
        )

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteNewtonianMonteCarloProposer for SingleSiteNewtonianMonteCarlo

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
