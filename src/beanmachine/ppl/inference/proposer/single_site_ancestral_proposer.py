import torch.distributions as dist
from beanmachine.ppl.inference.proposer.base_single_site_proposer import (
    BaseSingleSiteMHProposer,
)
from beanmachine.ppl.world import World


class SingleSiteAncestralProposer(BaseSingleSiteMHProposer):
    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        """Propose a new value for self.node using the prior distribution."""
        return world.get_variable(self.node).distribution
