from typing import List, Tuple, cast

import torch
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.world import World


class SequentialProposer(BaseProposer):
    def __init__(self, proposers: List[BaseProposer]):
        self.proposers = proposers

    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        accept_log_prob = 0.0
        for proposer in self.proposers:
            world, log_prob = proposer.propose(world)
            accept_log_prob += log_prob
        return world, cast(torch.Tensor, accept_log_prob)

    def do_adaptation(self, *args, **kwargs) -> None:
        for proposer in self.proposers:
            proposer.do_adaptation(*args, **kwargs)

    def finish_adaptation(self) -> None:
        for proposer in self.proposers:
            proposer.finish_adaptation()
