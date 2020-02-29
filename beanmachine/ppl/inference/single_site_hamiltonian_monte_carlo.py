# Copyright (c) Facebook, Inc. and its affiliates.
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteHamiltonianMonteCarlo(AbstractSingleSiteMHInference):
    """
    Implementation for SingleSiteHamiltonianMonteCarlo
    """

    def __init__(self, step_size: float, num_steps: int, use_transform_: bool = True):
        super().__init__()
        self.world_.set_all_nodes_transform(use_transform_)
        self.proposer_ = SingleSiteHamiltonianMonteCarloProposer(step_size, num_steps)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteHamiltonianMonteCarloProposer for SingleSiteHamiltonianMonteCarlo

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
