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
                # look at support and triage
                distribution = world.get_variable(node).distribution
                support = distribution.support  # pyre-ignore
                if is_constraint_eq(support, dist.constraints.real):
                    self._proposers[node] = SingleSiteRealSpaceNMCProposer(
                        node, self.alpha, self.beta
                    )
                elif is_constraint_eq(support, dist.constraints.greater_than):
                    self._proposers[node] = SingleSiteHalfSpaceNMCProposer(node)
                elif is_constraint_eq(support, dist.constraints.simplex) or (
                    isinstance(support, dist.constraints.independent)
                    and (support.base_constraint == dist.constraints.unit_interval)
                ):
                    self._proposers[node] = SingleSiteSimplexSpaceNMCProposer(node)
                elif isinstance(distribution, dist.Beta):
                    self._proposers[node] = SingleSiteSimplexSpaceNMCProposer(
                        node, transform=BetaDimensionTransform()
                    )
                else:
                    LOGGER.warning(
                        f"Node {node} has unsupported constraints. "
                        + "Proposer falls back to SingleSiteAncestralProposer.\n"
                    )
                    self._proposers[node] = SingleSiteAncestralProposer(node)

            proposers.append(self._proposers[node])
        return proposers
