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

    def discrete_update(
        self, world: World, idx: torch.Tensor, ke_discrete
    ) -> Tuple[World, torch.Tensor]:
        if world is not self.world:
            self.world = world

        disc_logprob = torch.as_tensor(0.0)

        n = self._disc_target_rvs[idx]
        discrete_kernel = SingleSiteUniformProposer(n)
        world, lp = discrete_kernel.propose(self.world)
        disc_logprob.add_(lp)

        if ke_discrete + disc_logprob > 0:
            self.world = world

        return self.world, disc_logprob

    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        if world is not self.world:
            # re-compute cached values since world was modified by other sources
            self.world = world

        ke_discrete = dist.Exponential(1).expand((self.num_discretes,)).sample()
        arrival_times = dist.Uniform(0, 1).expand((self.num_discretes,)).sample()
        idx = arrival_times.argmin()

        self.world, cont_logprob = self.hmc_kernel.propose(self.world)
        self.world, disc_logprob = self.discrete_update(
            self.world, idx, ke_discrete[idx]
        )

        return self.world, torch.as_tensor(0.0)
