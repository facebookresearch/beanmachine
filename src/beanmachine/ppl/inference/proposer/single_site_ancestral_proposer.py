# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributions as dist
from beanmachine.ppl.inference.proposer.base_single_site_proposer import (
    BaseSingleSiteMHProposer,
)
from beanmachine.ppl.world import World


class SingleSiteAncestralProposer(BaseSingleSiteMHProposer):
    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        """Propose a new value for self.node using the prior distribution."""
        return world.get_variable(self.node).distribution
