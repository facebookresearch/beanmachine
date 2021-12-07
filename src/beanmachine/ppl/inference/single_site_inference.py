import random
from typing import List, Set, Type

from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.inference.proposer.base_single_site_proposer import (
    BaseSingleSiteMHProposer,
)
from beanmachine.ppl.inference.proposer.sequential_proposer import (
    SequentialProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class SingleSiteInference(BaseInference):
    def __init__(self, proposer_class: Type[BaseSingleSiteMHProposer], **kwargs):
        self.proposer_class = proposer_class
        self.inference_args = kwargs
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
                self._proposers[node] = self.proposer_class(  # pyre-ignore [45]
                    node, **self.inference_args
                )
            proposers.append(self._proposers[node])
        return proposers


class JointSingleSiteInference(SingleSiteInference):
    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        proposers = super().get_proposers(world, target_rvs, num_adaptive_sample)
        random.shuffle(proposers)
        return [SequentialProposer(proposers)]
