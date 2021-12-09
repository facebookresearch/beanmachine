# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.single_site_random_walk_proposer import (
    SingleSiteRandomWalkProposer,
)
from beanmachine.ppl.legacy.world import TransformType
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteRandomWalk(AbstractMHInference):
    """
    Implementation for SingleSiteRandomWalk
    """

    def __init__(
        self,
        step_size: float = 1.0,
        transform_type: TransformType = TransformType.NONE,
        transforms: Optional[List] = None,
    ):
        self.proposer_ = SingleSiteRandomWalkProposer(step_size)
        super().__init__(self.proposer_, transform_type, transforms)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteRandomWalkProposer for
        SingleSiteRandomWalk

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        return self.proposer_
