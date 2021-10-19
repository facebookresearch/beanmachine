from typing import List

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    SimpleWorld,
)


class SingleSiteAncestralMetropolisHastings(BaseInference):
    def __init__(self):
        self._proposers = {}

    def get_proposers(
        self, world: SimpleWorld, num_adaptive_sample: int
    ) -> List[BaseProposer]:
        proposers = []
        for node in world.latent_nodes:
            if node not in self._proposers:
                self._proposers[node] = SingleSiteAncestralProposer(node)
            proposers.append(self._proposers[node])
        return proposers
