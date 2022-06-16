# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Set

from beanmachine.ppl.experimental.mala.mala_proposer import MALAProposer

from beanmachine.ppl.inference.hmc_inference import (
    GlobalHamiltonianMonteCarlo,
    SingleSiteHamiltonianMonteCarlo,
)
from beanmachine.ppl.inference.proposer.base_proposer import BaseProposer
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World


class GlobalMetropolisAdapatedLangevinAlgorithm(GlobalHamiltonianMonteCarlo):
    """
    Global (multi-site) Metropolis Adapted Langevin Algorithm [1] sampler implemented as a special
    case of Hamiltonian Monte Carlo (HMC) algorithm with a single step as described in [2]. This global sampler blocks
    all of the target random_variables in the World together and proposes them jointly.

    [1] Gareth O. Roberts and Richard L. Tweedie, "Exponential Convergence of Langevin Distributions and Their Discrete
    Approximations" (1996).
    https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Langevin/RobertsTweedieBernoulli1996.pdf
    [2] Radford Neal. "MCMC Using Hamiltonian Dynamics" (2011). https://arxiv.org/abs/1206.1901

    Args:
        initial_step_size (float): Defaults to 1.0.
        adapt_step_size (bool): Whether to adapt the step size, Defaults to True,
        adapt_mass_matrix (bool): Whether to adapt the mass matrix. Defaults to True,
        target_accept_prob (float): Target accept prob. Increasing this value would lead
            to smaller step size. Defaults to 0.8.
        nnc_compile: (Experimental) If True, NNC compiler will be used to accelerate the
            inference (defaults to False).
    """

    def __init__(
        self,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
        nnc_compile: bool = False,
    ):
        arbitrary_trajectory_value = 0.0
        super().__init__(
            trajectory_length=arbitrary_trajectory_value,
            initial_step_size=initial_step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            target_accept_prob=target_accept_prob,
            nnc_compile=nnc_compile,
        )

    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        if self._proposer is None:
            self._proposer = MALAProposer(
                world,
                target_rvs,
                num_adaptive_sample,
                self.initial_step_size,
                self.adapt_step_size,
                self.adapt_mass_matrix,
                self.target_accept_prob,
                self.nnc_compile,
            )
        return [self._proposer]


class SingleSiteMetropolisAdapatedLangevinAlgorithm(SingleSiteHamiltonianMonteCarlo):
    """
    Single site Metropolis Adapted Langevin Algorithm [1] sampler implemented as a special
    case of Hamiltonian Monte Carlo (HMC) algorithm with a single step as described in [2]. During inference, each random
    variable is proposed through its own leapfrog trajectory while fixing the rest of
    World as constant.

    [1] Gareth O. Roberts and Richard L. Tweedie, "Exponential Convergence of Langevin Distributions and Their Discrete
    Approximations" (1996).
    https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Langevin/RobertsTweedieBernoulli1996.pdf.
    [2] Radford Neal. "MCMC Using Hamiltonian Dynamics" (2011). https://arxiv.org/abs/1206.1901

    Args:
        initial_step_size (float): Defaults to 1.0.
        adapt_step_size (bool): Whether to adapt the step size, Defaults to True,
        adapt_mass_matrix (bool): Whether to adapt the mass matrix. Defaults to True,
        target_accept_prob (float): Target accept prob. Increasing this value would lead
            to smaller step size. Defaults to 0.8.
    """

    def __init__(
        self,
        initial_step_size: float = 1.0,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        target_accept_prob: float = 0.8,
        nnc_compile: bool = False,
    ):
        arbitrary_trajectory_value = 0.0
        super().__init__(
            trajectory_length=arbitrary_trajectory_value,
            initial_step_size=initial_step_size,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            target_accept_prob=target_accept_prob,
            nnc_compile=nnc_compile,
        )

    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        proposers = []
        for node in target_rvs:
            if node not in self._proposers:
                self._proposers[node] = MALAProposer(
                    world,
                    {node},
                    num_adaptive_sample,
                    self.initial_step_size,
                    self.adapt_step_size,
                    self.adapt_mass_matrix,
                    self.target_accept_prob,
                    self.nnc_compile,
                )
            proposers.append(self._proposers[node])
        return proposers
