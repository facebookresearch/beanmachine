# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Tuple

import torch
from beanmachine.ppl.inference.proposer.hmc_proposer import HMCProposer
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class MALAProposer(HMCProposer):
    """
        The Metropolis Adapted Langevin Algorithm sampler [1] implemented as a special
        case of Hamiltonian Monte Carlo (HMC) algorithm with a single step as described in [2]

        Reference:
            [1] Gareth O. Roberts and Richard L. Tweedie, "Exponential Convergence of Langevin Distributions and Their Discrete
    Approximations" (1996). https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Langevin/RobertsTweedieBernoulli1996.pdf
            [2] Radford Neal. "MCMC Using Hamiltonian Dynamics" (2011).
                https://arxiv.org/abs/1206.1901

        Args:
            initial_world: Initial world to propose from.
            target_rvs: Set of RVIdentifiers to indicate which variables to propose.
            num_adaptive_samples: Number of adaptive samples to run.
            initial_step_size: Initial step size.
            adapt_step_size: Flag whether to adapt step size, defaults to True.
            adapt_mass_matrix: Flat whether to adapt mass matrix, defaults to True.
            target_accept_prob: Target accept prob, defaults to 0.8.
            nnc_compile: (Experimental) If True, NNC compiler will be used to accelerate the
                inference (defaults to False).
    """

    def __init__(
        self,
        initial_world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_samples: int,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
        nnc_compile: bool = False,
    ):
        arbitrary_trajectory_value = 0.0
        super().__init__(
            initial_world,
            target_rvs,
            num_adaptive_samples,
            trajectory_length=arbitrary_trajectory_value,
            initial_step_size=initial_step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            target_accept_prob=target_accept_prob,
            nnc_compile=nnc_compile,
        )

    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        if world is not self.world:
            # re-compute cached values since world was modified by other sources
            self.world = world
            self._positions = self._to_unconstrained(
                {node: world[node] for node in self._target_rvs}
            )
            self._pe, self._pe_grad = self._potential_grads(self._positions)
        momentums = self._initialize_momentums(self._positions)
        current_energy = self._hamiltonian(
            self._positions, momentums, self._mass_inv, self._pe
        )
        # only one leapfrog step is used.
        positions, momentums, pe, pe_grad = self._leapfrog_step(
            self._positions, momentums, self.step_size, self._mass_inv, self._pe_grad
        )
        new_energy = torch.nan_to_num(
            self._hamiltonian(positions, momentums, self._mass_inv, pe),
            float("inf"),
        )
        delta_energy = new_energy - current_energy
        self._alpha = torch.clamp(torch.exp(-delta_energy), max=1.0)
        # accept/reject new world
        if torch.bernoulli(self._alpha):
            self.world = self.world.replace(self._to_unconstrained.inv(positions))
            # update cache
            self._positions, self._pe, self._pe_grad = positions, pe, pe_grad
        return self.world, torch.zeros_like(self._alpha)
