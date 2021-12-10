# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Set

from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.inference.proposer.single_site_random_walk_proposer import (
    SingleSiteRandomWalkProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class SingleSiteRandomWalk(BaseInference):
    """
    Single Site random walk Metropolis-Hastings. This single site algorithm uses a Normal distribution
    proposer.

    Args:
        step_size: Step size, defaults to 1.0
    """

    def __init__(self, step_size: float = 1.0):
        self.step_size = step_size
        self._proposers = {}

    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        proposers = []
        for node in target_rvs:
            if node not in self._proposers:
                self._proposers[node] = SingleSiteRandomWalkProposer(
                    node, self.step_size
                )
            proposers.append(self._proposers[node])
        return proposers
