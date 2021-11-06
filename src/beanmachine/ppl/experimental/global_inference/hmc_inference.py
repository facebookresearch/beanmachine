from typing import List, Set

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.hmc_proposer import (
    HMCProposer,
)
from beanmachine.ppl.experimental.global_inference.proposer.nuts_proposer import (
    NUTSProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class GlobalHamiltonianMonteCarlo(BaseInference):
    def __init__(
        self,
        trajectory_length: float,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
        nnc_compile: bool = False,
    ):
        self.trajectory_length = trajectory_length
        self.initial_step_size = initial_step_size
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.target_accept_prob = target_accept_prob
        self.nnc_compile = nnc_compile
        self._proposer = None

    def get_proposers(
        self,
        world: SimpleWorld,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        if self._proposer is None:
            self._proposer = HMCProposer(
                world,
                target_rvs,
                num_adaptive_sample,
                self.trajectory_length,
                self.initial_step_size,
                self.adapt_step_size,
                self.adapt_mass_matrix,
                self.target_accept_prob,
                self.nnc_compile,
            )
        return [self._proposer]


class GlobalNoUTurnSampler(BaseInference):
    def __init__(
        self,
        max_tree_depth: int = 10,
        max_delta_energy: float = 1000.0,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        multinomial_sampling: bool = True,
        target_accept_prob: float = 0.8,
        nnc_compile: bool = False,
    ):
        self.max_tree_depth = max_tree_depth
        self.max_delta_energy = max_delta_energy
        self.initial_step_size = initial_step_size
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.multinomial_sampling = multinomial_sampling
        self.target_accept_prob = target_accept_prob
        self.nnc_compile = nnc_compile
        self._proposer = None

    def get_proposers(
        self,
        world: SimpleWorld,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        if self._proposer is None:
            self._proposer = NUTSProposer(
                world,
                target_rvs,
                num_adaptive_sample,
                self.max_tree_depth,
                self.max_delta_energy,
                self.initial_step_size,
                self.adapt_step_size,
                self.adapt_mass_matrix,
                self.multinomial_sampling,
                self.target_accept_prob,
                self.nnc_compile,
            )
        return [self._proposer]
