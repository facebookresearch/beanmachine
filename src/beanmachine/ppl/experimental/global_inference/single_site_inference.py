from typing import List, Set, Type

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.base_single_site_proposer import (
    BaseSingleSiteMHProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    SimpleWorld,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteInference(BaseInference):
    def __init__(self, proposer_class: Type[BaseSingleSiteMHProposer], *args, **kwargs):
        self.proposer_class = proposer_class
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
                self._proposers[node] = self.proposer_class(node)  # pyre-ignore [45]
            proposers.append(self._proposers[node])
        return proposers
