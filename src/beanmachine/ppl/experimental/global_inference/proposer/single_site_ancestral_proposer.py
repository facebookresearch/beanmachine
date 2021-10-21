import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.proposer.base_single_site_proposer import (
    BaseSingleSiteMHProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


class SingleSiteAncestralProposer(BaseSingleSiteMHProposer):
    def get_proposal_distribution(self, world: SimpleWorld) -> dist.Distribution:
        """Propose a new value for self.node using the prior distribution."""
        return world.get_variable(self.node).distribution
