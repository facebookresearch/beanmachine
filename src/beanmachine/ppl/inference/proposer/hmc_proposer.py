# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import Callable, Optional, Set, Tuple

import torch
from beanmachine.ppl.experimental.torch_jit_backend import jit_compile, TorchJITBackend
from beanmachine.ppl.inference.proposer.base_proposer import BaseProposer
from beanmachine.ppl.inference.proposer.hmc_utils import (
    DualAverageAdapter,
    MassMatrixAdapter,
    RealSpaceTransform,
    WindowScheme,
)
from beanmachine.ppl.inference.proposer.utils import DictToVecConverter
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class HMCProposer(BaseProposer):
    """
    The basic Hamiltonian Monte Carlo (HMC) algorithm as described in [1] plus a
    dual-averaging mechanism for dynamically adjusting the step size [2].

    Reference:
        [1] Radford Neal. "MCMC Using Hamiltonian Dynamics" (2011).
            https://arxiv.org/abs/1206.1901

        [2] Matthew Hoffman and Andrew Gelman. "The No-U-Turn Sampler: Adaptively
            Setting Path Lengths in Hamiltonian Monte Carlo" (2014).
            https://arxiv.org/abs/1111.4246

    The current implementation does not use nor adapt a mass matrix -- which is
    equivalent to setting the matrix M to I.

    Args:
        initial_world: Initial world to propose from.
        target_rvs: Set of RVIdentifiers to indicate which variables to propose.
        num_adaptive_samples: Number of adaptive samples to run.
        trajectory_length: Length of single trajectory.
        initial_step_size: Initial step size.
        adapt_step_size: Flag whether to adapt step size, defaults to True.
        adapt_mass_matrix: Flat whether to adapt mass matrix, defaults to True.
        target_accept_prob: Target accept prob, defaults to 0.8.
        nnc_compile: If True, NNC compiler will be used to accelerate the
            inference.
    """

    def __init__(
        self,
        initial_world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_samples: int,
        trajectory_length: float,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        full_mass_matrix: bool = False,
        target_accept_prob: float = 0.8,
        jit_backend: TorchJITBackend = TorchJITBackend.NNC,
    ):
        self.world = initial_world
        self._target_rvs = target_rvs
        self._to_unconstrained = RealSpaceTransform(initial_world, target_rvs)
        # concatenate and flatten the positions into a single tensor
        positions_dict = self._to_unconstrained(
            {node: initial_world[node] for node in self._target_rvs}
        )
        self._dict2vec = DictToVecConverter(positions_dict)
        self._positions = self._dict2vec.to_vec(positions_dict)
        # cache pe and pe_grad to prevent re-computation
        self._pe, self._pe_grad = self._potential_grads(self._positions)
        # initialize parameters
        self.trajectory_length = trajectory_length
        # initialize adapters
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        # we need mass matrix adapter to sample momentums
        self._mass_matrix_adapter = MassMatrixAdapter(
            len(self._positions), full_mass_matrix
        )
        if self.adapt_step_size:
            self.step_size = self._find_reasonable_step_size(
                torch.as_tensor(initial_step_size),
                self._positions,
                self._pe,
                self._pe_grad,
            )
            self._step_size_adapter = DualAverageAdapter(
                self.step_size, target_accept_prob
            )
        else:
            self.step_size = torch.as_tensor(initial_step_size)
        if self.adapt_mass_matrix:
            self._window_scheme = WindowScheme(num_adaptive_samples)
        else:
            self._window_scheme = None
        # alpha will store the accept prob and will be used to adapt step size
        self._alpha = None

        # pyre-ignore[8]
        self._leapfrog_step = jit_compile(self._leapfrog_step, jit_backend)

    @property
    def _initialize_momentums(self) -> Callable:
        return self._mass_matrix_adapter.initialize_momentums

    @property
    def _mass_inv(self) -> torch.Tensor:
        return self._mass_matrix_adapter.mass_inv

    def _scale_r(self, momentums: torch.Tensor, mass_inv: torch.Tensor) -> torch.Tensor:
        """Return the momentums (r) scaled by M^{-1} @ r"""
        if self._mass_matrix_adapter.diagonal:
            return mass_inv * momentums
        else:
            return mass_inv @ momentums

    def _kinetic_energy(
        self, momentums: torch.Tensor, mass_inv: torch.Tensor
    ) -> torch.Tensor:
        """Returns the kinetic energy KE = 1/2 * p^T @ M^{-1} @ p (equation 2.6 in [1])"""
        r_scale = self._scale_r(momentums, mass_inv)
        return torch.dot(momentums, r_scale) / 2

    def _kinetic_grads(
        self, momentums: torch.Tensor, mass_inv: torch.Tensor
    ) -> torch.Tensor:
        """Returns a dictionary of gradients of kinetic energy function with respect to
        the momentum at each site, computed as M^{-1} @ p"""
        return self._scale_r(momentums, mass_inv)

    def _potential_energy(self, positions: torch.Tensor) -> torch.Tensor:
        """Returns the potential energy PE = - L(world) (the joint log likelihood of the
        current values)"""
        positions_dict = self._dict2vec.to_dict(positions)
        constrained_vals = self._to_unconstrained.inv(positions_dict)
        log_joint = self.world.replace(constrained_vals).log_prob()
        log_joint = log_joint - self._to_unconstrained.log_abs_det_jacobian(
            constrained_vals, positions_dict
        )
        return -log_joint

    def _potential_grads(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns potential energy as well as a dictionary of its gradient with
        respect to the value at each site."""
        positions.requires_grad = True

        try:
            pe = self._potential_energy(positions)
            grads = torch.autograd.grad(pe, positions)[0]
        # We return NaN on Cholesky factorization errors which can be gracefully
        # handled by NUTS/HMC.
        except RuntimeError as e:
            err_msg = str(e)
            if "singular U" in err_msg or "input is not positive-definite" in err_msg:
                warnings.warn(
                    "Numerical error in potential energy computation."
                    " If automatic recovery does not happen, plese file an issue"
                    " at https://github.com/facebookresearch/beanmachine/issues/."
                )
                grads = torch.full_like(positions, float("nan"))
                pe = torch.tensor(
                    float("nan"), device=grads[0].device, dtype=grads[0].dtype
                )
            else:
                raise e

        positions.requires_grad = False
        return pe.detach(), grads

    def _hamiltonian(
        self,
        positions: torch.Tensor,
        momentums: torch.Tensor,
        mass_inv: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the value of Hamiltonian equation (equatino 2.5 in [1]). This function
        will be more efficient if pe is provided as it only needs to compute the
        kinetic energy"""
        ke = self._kinetic_energy(momentums, mass_inv)
        if pe is None:
            pe = self._potential_energy(positions)
        return pe + ke

    def _leapfrog_step(
        self,
        positions: torch.Tensor,
        momentums: torch.Tensor,
        step_size: torch.Tensor,
        mass_inv: torch.Tensor,
        pe_grad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single leapfrog integration (alson known as the velocity Verlet
        method) as described in equation 2.28-2.30 in [1]. If the values of potential
        grads of the current world is provided, then we only needs to compute the
        gradient once per step."""
        if pe_grad is None:
            _, pe_grad = self._potential_grads(positions)

        new_momentums = momentums - step_size * pe_grad / 2
        ke_grad = self._kinetic_grads(new_momentums, mass_inv)

        new_positions = positions + step_size * ke_grad

        pe, pe_grad = self._potential_grads(new_positions)
        new_momentums = new_momentums - step_size * pe_grad / 2

        return new_positions, new_momentums, pe, pe_grad

    def _leapfrog_updates(
        self,
        positions: torch.Tensor,
        momentums: torch.Tensor,
        trajectory_length: float,
        step_size: torch.Tensor,
        mass_inv: torch.Tensor,
        pe_grad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run multiple iterations of leapfrog integration until the length of the
        trajectory is greater than the specified trajectory_length."""
        # we should run at least 1 step
        num_steps = max(math.ceil(trajectory_length / step_size.item()), 1)
        for _ in range(num_steps):
            positions, momentums, pe, pe_grad = self._leapfrog_step(
                positions, momentums, step_size, mass_inv, pe_grad
            )
        # pyre-ignore[61]: `pe` may not be initialized here.
        return positions, momentums, pe, pe_grad

    def _find_reasonable_step_size(
        self,
        initial_step_size: torch.Tensor,
        positions: torch.Tensor,
        pe: torch.Tensor,
        pe_grad: torch.Tensor,
    ) -> torch.Tensor:
        """A heuristic of finding a reasonable initial step size (epsilon) as introduced
        in Algorithm 4 of [2]."""
        step_size = initial_step_size
        # the target is log(0.5) in the paper but is log(0.8) on Stan:
        # https://github.com/stan-dev/stan/pull/356
        target = math.log(0.8)
        momentums = self._initialize_momentums(positions)
        energy = self._hamiltonian(
            positions, momentums, self._mass_inv, pe
        )  # -log p(positions, momentums)
        new_positions, new_momentums, new_pe, _ = self._leapfrog_step(
            positions, momentums, step_size, self._mass_inv, pe_grad
        )
        new_energy = self._hamiltonian(
            new_positions, new_momentums, self._mass_inv, new_pe
        )
        # NaN will evaluate to False and set direction to -1
        new_direction = direction = 1 if energy - new_energy > target else -1
        step_size_scale = 2**direction
        while new_direction == direction:
            step_size *= step_size_scale
            if step_size == 0:
                raise ValueError(
                    f"Current step size is {step_size}. No acceptably small step size could be found."
                    "Perhaps the posterior is not continuous?"
                )
            if step_size > 1e7:
                raise ValueError(
                    f"Current step size is {step_size}. Posterior is improper. Please check your model"
                )
            # not covered in the paper, but both Stan and Pyro re-sample the momentum
            # after each update
            momentums = self._initialize_momentums(positions)
            energy = self._hamiltonian(positions, momentums, self._mass_inv, pe)
            new_positions, new_momentums, new_pe, _ = self._leapfrog_step(
                positions, momentums, step_size, self._mass_inv, pe_grad
            )
            new_energy = self._hamiltonian(
                new_positions, new_momentums, self._mass_inv, new_pe
            )
            new_direction = 1 if energy - new_energy > target else -1
        return step_size

    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        if world is not self.world:
            # re-compute cached values since world was modified by other sources
            self.world = world
            self._positions = self._dict2vec.to_vec(
                self._to_unconstrained({node: world[node] for node in self._target_rvs})
            )
            self._pe, self._pe_grad = self._potential_grads(self._positions)
        momentums = self._initialize_momentums(self._positions)
        current_energy = self._hamiltonian(
            self._positions, momentums, self._mass_inv, self._pe
        )
        positions, momentums, pe, pe_grad = self._leapfrog_updates(
            self._positions,
            momentums,
            self.trajectory_length,
            self.step_size,
            self._mass_inv,
            self._pe_grad,
        )
        new_energy = torch.nan_to_num(
            self._hamiltonian(positions, momentums, self._mass_inv, pe),
            float("inf"),
        )
        delta_energy = new_energy - current_energy
        self._alpha = torch.clamp(torch.exp(-delta_energy), max=1.0)
        # accept/reject new world
        if torch.bernoulli(self._alpha):
            positions_dict = self._dict2vec.to_dict(positions)
            self.world = self.world.replace(self._to_unconstrained.inv(positions_dict))
            # update cache
            self._positions, self._pe, self._pe_grad = positions, pe, pe_grad
        return self.world, torch.zeros_like(self._alpha)

    def do_adaptation(self, *args, **kwargs) -> None:
        if self._alpha is None:
            return

        if self.adapt_step_size:
            self.step_size = self._step_size_adapter.step(self._alpha)

        if self.adapt_mass_matrix:
            window_scheme = self._window_scheme
            assert window_scheme is not None
            if window_scheme.is_in_window:
                self._mass_matrix_adapter.step(self._positions)
                if window_scheme.is_end_window:
                    # update mass matrix at the end of a window
                    self._mass_matrix_adapter.finalize()

                    if self.adapt_step_size:
                        self.step_size = self._step_size_adapter.finalize()
                        self.step_size = self._find_reasonable_step_size(
                            self.step_size,
                            self._positions,
                            self._pe,
                            self._pe_grad,
                        )
                        self._step_size_adapter = DualAverageAdapter(self.step_size)
            window_scheme.step()
        self._alpha = None

    def finish_adaptation(self) -> None:
        if self.adapt_step_size:
            self.step_size = self._step_size_adapter.finalize()
