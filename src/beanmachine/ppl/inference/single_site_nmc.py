# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Set

import torch.distributions as dist
from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.inference.proposer.nmc import (
    SingleSiteRealSpaceNMCProposer,
    SingleSiteHalfSpaceNMCProposer,
    SingleSiteSimplexSpaceNMCProposer,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World
from beanmachine.ppl.world.utils import (
    BetaDimensionTransform,
)
from beanmachine.ppl.world.utils import is_constraint_eq

LOGGER = logging.getLogger("beanmachine")


class SingleSiteNewtonianMonteCarlo(BaseInference):
    """
    Single site Newtonian Monte Carlo [1]. This algorithm selects a proposer
    based on the support of the random variable. Valid supports include real, positive real, and simplex.
    Each site is proposed independently.

    [1] Arora, Nim, et al. `Newtonian Monte Carlo: single-site MCMC meets second-order gradient methods`

    Args:
        real_space_alpha: alpha value for real space as specified in [1], defaults to 10.0
        real_space_beta: beta value for real space  as specified in [1], defaults to 1.0
    """

    def __init__(
        self,
        real_space_alpha: float = 10.0,
        real_space_beta: float = 1.0,
    ):
        self._proposers = {}
        self.alpha = real_space_alpha
        self.beta = real_space_beta

    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        proposers = []
        for node in target_rvs:
            if node not in self._proposers:
                self._proposers[node] = self._init_nmc_proposer(node, world)
            proposers.append(self._proposers[node])
        return proposers

    def _init_nmc_proposer(self, node: RVIdentifier, world: World) -> BaseProposer:
        """
        A helper function that initialize a NMC proposer for the given node. The type
        of NMC proposer will be chosen based on a node's support.
        """
        distribution = world.get_variable(node).distribution
        support = distribution.support  # pyre-ignore
        if is_constraint_eq(support, dist.constraints.real):
            return SingleSiteRealSpaceNMCProposer(node, self.alpha, self.beta)
        elif is_constraint_eq(support, dist.constraints.greater_than):
            return SingleSiteHalfSpaceNMCProposer(node)
        elif is_constraint_eq(support, dist.constraints.simplex) or (
            isinstance(support, dist.constraints.independent)
            and (support.base_constraint == dist.constraints.unit_interval)
        ):
            return SingleSiteSimplexSpaceNMCProposer(node)
        elif isinstance(distribution, dist.Beta):
            return SingleSiteSimplexSpaceNMCProposer(
                node, transform=BetaDimensionTransform()
            )
        else:
            LOGGER.warning(
                f"Node {node} has unsupported constraints. "
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return SingleSiteAncestralProposer(node)
