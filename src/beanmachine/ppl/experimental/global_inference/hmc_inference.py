import dataclasses

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.hmc_proposer import (
    HMCProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.nuts_proposer import (
    NUTSProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


@dataclasses.dataclass
class GlobalHamiltonianMonteCarlo(BaseInference):
    trajectory_length: float
    initial_step_size: float = 1.0
    adapt_step_size: bool = True

    def get_proposer(self, world: SimpleWorld) -> HMCProposer:
        return HMCProposer(world, **dataclasses.asdict(self))


@dataclasses.dataclass
class GlobalNoUTurnSampler(BaseInference):
    max_tree_depth: int = 10
    max_delta_energy: float = 1000.0
    initial_step_size: float = 1.0
    adapt_step_size: bool = True

    def get_proposer(self, world: SimpleWorld) -> NUTSProposer:
        return NUTSProposer(world, **dataclasses.asdict(self))
