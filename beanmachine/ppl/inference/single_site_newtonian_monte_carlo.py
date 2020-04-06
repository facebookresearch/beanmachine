# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteNewtonianMonteCarlo(AbstractMHInference):
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
        self.proposer_ = {}
        self.real_space_alpha_ = real_space_alpha
        self.real_space_beta_ = real_space_beta

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteNewtonianMonteCarloProposer for SingleSiteNewtonianMonteCarlo

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        if node not in self.proposer_:
            self.proposer_[node] = SingleSiteNewtonianMonteCarloProposer(
                self.real_space_alpha_, self.real_space_beta_
            )
        return self.proposer_[node]
