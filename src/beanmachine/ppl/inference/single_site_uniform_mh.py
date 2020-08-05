# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional

from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world.world import TransformType


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
