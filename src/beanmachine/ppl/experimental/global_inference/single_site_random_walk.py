from typing import List, Set

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.single_site_random_walk_proposer import (
    SingleSiteRandomWalkProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    SimpleWorld,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteRandomWalk(BaseInference):
    def __init__(self, step_size: float = 1.0):
        self.step_size = step_size
        self._proposers = {}

    def get_proposers(
        self,
        world: SimpleWorld,
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
