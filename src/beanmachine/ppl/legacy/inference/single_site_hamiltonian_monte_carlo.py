# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.world import TransformType
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteHamiltonianMonteCarlo(AbstractMHInference):
    """
    Implementation for SingleSiteHamiltonianMonteCarlo
    """

    def __init__(
        self,
        path_length: float,
        step_size: float = 0.1,
        transform_type: TransformType = TransformType.DEFAULT,
        transforms: Optional[List] = None,
    ):
        super().__init__(
            proposer=SingleSiteHamiltonianMonteCarloProposer(path_length, step_size),
            transform_type=transform_type,
            transforms=transforms,
        )
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
