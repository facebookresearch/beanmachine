# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.legacy.world import TransformType
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteUniformMetropolisHastings(AbstractMHInference):
    """
    Implementation for SingleSiteNewtonianMonteCarlo
    """

    def __init__(
        self,
        transform_type: TransformType = TransformType.NONE,
        transforms: Optional[List] = None,
    ):
        self.proposer_ = SingleSiteUniformProposer(transform_type, transforms)
        super().__init__(self.proposer_, transform_type, transforms)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteUniformMetropolisHastingsProposer for
        SingleSiteUniformMetropolisHastings

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
