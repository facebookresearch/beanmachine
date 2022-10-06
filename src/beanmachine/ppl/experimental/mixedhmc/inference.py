# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Set, Tuple

import torch
import torch.distributions as dist

from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import BaseProposer

from beanmachine.ppl.inference.proposer.hmc_proposer import HMCProposer
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)

from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class MixedHMC(BaseInference):
    """
    MixedHMC with Laplace momentum [1]. This sampler alternatives between taking
    leapfrog steps for continuous random variables and steps from a discrete
    proposal distributions.

    Reference:
        [1] Guangyao Zhou. "Mixed Hamiltonian Monte Carlo for Mixed Discrete
            and Continuous Variables" (2019). https://arxiv.org/abs/1909.04852v6

    Args:
        max_step_size (float): Maximum step size
        num_discrete_update (int): Number of times to update the discrete variables
        trajectory_length (int): Length of single trajectory.
    """

    def __init__(
        self,
        max_step_size,
        num_discrete_updates,
        trajectory_length,
    ):
        self.max_step_size = max_step_size
        self.num_discrete_updates = num_discrete_updates
        self.trajectory_length = trajectory_length
        self._proposer = None

    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        if self._proposer is None:
            self._proposer = MixedHMCProposer(
                world,
                target_rvs,
                self.max_step_size,
                num_adaptive_sample,
                self.num_discrete_updates,
                self.trajectory_length,
            )
        return [self._proposer]


class MixedHMCProposer(BaseProposer):
    """
    The Mixed HMC algorithm as described in [1]. The current implementation uses
    the simplified algorithm with Laplace momentum.

    Reference:
        [1] Guangyao Zhou. "Mixed Hamiltonian Monte Carlo for Mixed Discrete
            and Continuous Variables" (2019). https://arxiv.org/abs/1909.04852v6

    Args:
        initial_world: Initial world to propose from.
        target_rvs: Set of RVIdentifiers to indicate which variables to propose.
        max_step_size (float): Maximum step size
        num_discrete_update (int): Number of times to update the discrete variables
        trajectory_length: Length of single trajectory.
    """

    def __init__(
        self,
        initial_world: World,
        target_rvs: Set[RVIdentifier],
        max_step_size: float,
        num_adaptive_sample: int,
        num_discrete_updates: int,
        trajectory_length: int,
    ):
        self.world = initial_world
        self._cont_target_rvs = set()
        self._disc_target_rvs = []
        self.num_discrete_updates = num_discrete_updates
        self.step_size = torch.as_tensor(max_step_size)

        for node in target_rvs:
            support = initial_world.get_variable(node).distribution.support
            if not support.is_discrete:
                self._cont_target_rvs.add(node)
            else:
                self._disc_target_rvs.append(node)

        self.num_discretes = len(self._disc_target_rvs)

        self.hmc_kernel = HMCProposer(
            initial_world,
            self._cont_target_rvs,
            num_adaptive_sample,
            trajectory_length,
            max_step_size,
        )
        # initialize parameters
        self.trajectory_length = trajectory_length

    def continuous_update(self, positions, momentums, trajectory_length, pe_grad):
        step_size = self.hmc_kernel.step_size
        mass_inv = self.hmc_kernel._mass_inv
        positions, momentums, pe, pe_grad = self.hmc_kernel._leapfrog_updates(
            positions, momentums, trajectory_length, step_size, mass_inv, pe_grad
        )
        return positions, momentums, pe, pe_grad

    def discrete_update(
        self, world: World, idx: torch.Tensor, ke_discrete, delta_potential_energy
    ) -> Tuple[World, torch.Tensor, torch.Tensor]:
        n = self._disc_target_rvs[idx]
        discrete_kernel = SingleSiteUniformProposer(n)
        proposed_world, new_lp = discrete_kernel.propose(self.world)

        delta_discrete_energy = proposed_world.log_prob() - world.log_prob()
        accept = ke_discrete > delta_discrete_energy
        if accept:
            delta_potential_energy.add_(delta_discrete_energy)
            ke_discrete.subtract_(delta_discrete_energy)
            world = proposed_world

        return world, ke_discrete, delta_potential_energy

    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        if world is not self.world:
            # re-compute cached values since world was modified by other sources
            self.world = world

        positions = self.hmc_kernel._positions
        momentums = self.hmc_kernel._initialize_momentums(positions)
        hmc_pe, hmc_pe_grad = self.hmc_kernel._pe, self.hmc_kernel._pe_grad
        current_energy = self.hmc_kernel._hamiltonian(
            positions, momentums, self.hmc_kernel._mass_inv, hmc_pe
        )
        self.hmc_kernel.step_size = self.hmc_kernel._find_reasonable_step_size(
            torch.as_tensor(self.step_size),
            positions,
            hmc_pe,
            hmc_pe_grad,
        )

        delta_potential_energy = torch.as_tensor(0.0)

        ke_discrete = dist.Exponential(1).expand((self.num_discretes,)).sample()
        arrival_times = dist.Uniform(0, 1).expand((self.num_discretes,)).sample()
        arrival_times, _ = arrival_times.sort()
        idx = arrival_times.argmin()
        total_time = (
            self.num_discrete_updates - 1
        ) // self.num_discretes + arrival_times
        total_time = total_time[(self.num_discrete_updates - 1) % self.num_discretes]
        trajectory_length = self.trajectory_length / total_time

        for _ in range(self.num_discrete_updates):
            positions, momentums, hmc_pe, hmc_pe_grad = self.continuous_update(
                positions, momentums, trajectory_length, hmc_pe_grad
            )

            positions_dict = self.hmc_kernel._dict2vec.to_dict(positions)
            world = world.replace(self.hmc_kernel._to_unconstrained.inv(positions_dict))
            world, k, delta_potential_energy = self.discrete_update(
                world, idx, ke_discrete[idx], delta_potential_energy
            )
            ke_discrete[idx] = k
            arrival_times[idx] = 1
            idx = arrival_times.argmin()

        new_energy = self.hmc_kernel._hamiltonian(
            positions, momentums, self.hmc_kernel._mass_inv, hmc_pe
        )
        delta_energy = torch.nan_to_num(
            new_energy - current_energy - delta_potential_energy,
            float("inf"),
        )
        alpha = torch.clamp(torch.exp(-delta_energy), max=1.0)
        if torch.bernoulli(alpha):
            self.world = world.replace(
                self.hmc_kernel._to_unconstrained.inv(positions_dict)
            )
            (
                self.hmc_kernel._positions,
                self.hmc_kernel._pe,
                self.hmc_kernel._pe_grad,
            ) = (positions, hmc_pe, hmc_pe_grad)

        return self.world, torch.as_tensor(0.0)
