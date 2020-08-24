# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional

from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world.world import TransformType

from .proposer.single_site_no_u_turn_sampler_proposer import (
    SingleSiteNoUTurnSamplerProposer,
)


class SingleSiteNoUTurnSampler(AbstractMHInference):
    """
    Implementation for SingleSiteNoUTurnSampler
    """

    def __init__(
        self,
        transform_type: TransformType = TransformType.DEFAULT,
        transforms: Optional[List] = None,
    ):
        super().__init__(
            proposer=SingleSiteNoUTurnSamplerProposer(),
            transform_type=transform_type,
            transforms=transforms,
        )
        self.proposer_ = {}

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node which is
        SingleSiteHamiltonianMonteCarloProposer for SingleSiteHamiltonianMonteCarlo

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        if node not in self.proposer_:
            self.proposer_[node] = SingleSiteNoUTurnSamplerProposer()
        return self.proposer_[node]
