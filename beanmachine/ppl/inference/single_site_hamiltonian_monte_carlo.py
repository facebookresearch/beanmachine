# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteHamiltonianMonteCarlo(AbstractMHInference):
    """
    Implementation for SingleSiteHamiltonianMonteCarlo
    """

    def __init__(self, path_length: float, step_size: float = 0.1):
        super().__init__()
        self.world_.set_all_nodes_transform(True)
        self.proposer_ = {}
        self.path_length_ = path_length
        self.step_size_ = step_size

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteHamiltonianMonteCarloProposer for SingleSiteHamiltonianMonteCarlo

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        if node not in self.proposer_:
            self.proposer_[node] = SingleSiteHamiltonianMonteCarloProposer(
                self.path_length_, self.step_size_
            )
        return self.proposer_[node]
