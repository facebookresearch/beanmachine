from typing import List, Set

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    SimpleWorld,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class SingleSiteUniformMetropolisHastings(BaseInference):
    def __init__(self):
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
                self._proposers[node] = SingleSiteUniformProposer(node)
            proposers.append(self._proposers[node])
        return proposers
