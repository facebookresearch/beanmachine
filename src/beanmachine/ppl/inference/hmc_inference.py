# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Set

from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.inference.proposer.hmc_proposer import (
    HMCProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class GlobalHamiltonianMonteCarlo(BaseInference):
    """
    Global (multi-site) Hamiltonian Monte Carlo [1] sampler. This global sampler blocks
    all of the target random_variables in the World together and proposes them jointly.

    [1] Neal, Radford. `MCMC Using Hamiltonian Dynamics`.

    Args:
        trajectory_length (float): Length of single trajectory.
        initial_step_size (float): Defaults to 1.0.
        adapt_step_size (bool): Whether to adapt the step size, Defaults to True,
        adapt_mass_matrix (bool): Whether to adapt the mass matrix. Defaults to True,
        target_accept_prob (float): Target accept prob. Increasing this value would lead
            to smaller step size. Defaults to 0.8.
    """

    def __init__(
        self,
        trajectory_length: float,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
    ):
        self.trajectory_length = trajectory_length
        self.initial_step_size = initial_step_size
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.target_accept_prob = target_accept_prob
        self._proposer = None

    def get_proposers(
        self,
        world: World,
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
            )
        return [self._proposer]


class SingleSiteHamiltonianMonteCarlo(BaseInference):
    """
    Single site Hamiltonian Monte Carlo [1] sampler. During inference, each random
    variable is proposed through its own leapfrog trajectory while fixing the rest of
    World as constant.

    [1] Neal, Radford. `MCMC Using Hamiltonian Dynamics`.

    Args:
        trajectory_length (float): Length of single trajectory.
        initial_step_size (float): Defaults to 1.0.
        adapt_step_size (bool): Whether to adapt the step size, Defaults to True,
        adapt_mass_matrix (bool): Whether to adapt the mass matrix. Defaults to True,
        target_accept_prob (float): Target accept prob. Increasing this value would lead
            to smaller step size. Defaults to 0.8.
    """

    def __init__(
        self,
        trajectory_length: float,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
    ):
        self.trajectory_length = trajectory_length
        self.initial_step_size = initial_step_size
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.target_accept_prob = target_accept_prob
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
                self._proposers[node] = HMCProposer(
                    world,
                    {node},
                    num_adaptive_sample,
                    self.trajectory_length,
                    self.initial_step_size,
                    self.adapt_step_size,
                    self.adapt_mass_matrix,
                    self.target_accept_prob,
                )
            proposers.append(self._proposers[node])
        return proposers
