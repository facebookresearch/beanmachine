# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.world import TransformType
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteNewtonianMonteCarlo(AbstractMHInference):
    """
    Implementation for SingleSiteNewtonianMonteCarlo
    """

    def __init__(
        self,
        transform_type: TransformType = TransformType.NONE,
        transforms: Optional[List] = None,
        real_space_alpha: float = 10.0,
        real_space_beta: float = 1.0,
    ):
        self.proposer_ = {}
        self.real_space_alpha_ = real_space_alpha
        self.real_space_beta_ = real_space_beta
        super().__init__(
            SingleSiteNewtonianMonteCarloProposer(), transform_type, transforms
        )

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
