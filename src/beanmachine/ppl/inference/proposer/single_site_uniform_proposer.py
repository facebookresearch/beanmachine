# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.base_single_site_proposer import (
    BaseSingleSiteMHProposer,
)
from beanmachine.ppl.world import World


class SingleSiteUniformProposer(BaseSingleSiteMHProposer):
    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        """Propose a new value for self.node using the prior distribution."""
        node_dist = world.get_variable(self.node).distribution
        if isinstance(node_dist, dist.Bernoulli):
            return dist.Bernoulli(torch.ones(node_dist.param_shape) / 2.0)
        elif isinstance(node_dist, dist.Categorical):
            return dist.Categorical(torch.ones(node_dist.param_shape))
        else:
            # default to ancestral sampling
            # TODO: we should sample from a transformed dist
            # that transforms uniform to the support of the node
            return node_dist
