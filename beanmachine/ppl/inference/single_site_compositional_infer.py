# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

import torch.distributions as dist
from beanmachine.ppl.inference.abstract_single_site_mh_infer import (
    AbstractSingleSiteMHInference,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier


class SingleSiteCompositionalInference(AbstractSingleSiteMHInference):
    """
    Compositional inference
    """

    def __init__(self, proposers: Dict = None, should_transform: bool = False):
        self.proposers_ = proposers
        self.should_transform_ = should_transform
        super().__init__()

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node given the proposer dicts passed in
        once instantiating the class.

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        if self.proposers_ is not None and node.function in self.proposers_:
            return self.proposers_[node]

        node_var = self.world_.get_node_in_world(node, False)
        distribution = node_var.distribution
        support = distribution.support
        if isinstance(support, dist.constraints._Real) or isinstance(
            support, dist.constraints._Simplex
        ):
            return SingleSiteNewtonianMonteCarloProposer(self.world_)
        if isinstance(support, dist.constraints._IntegerInterval) and isinstance(
            distribution, dist.Categorical
        ):
            return SingleSiteUniformProposer(self.world_)
        if isinstance(support, dist.constraints._Boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            return SingleSiteUniformProposer(self.world_)
        return SingleSiteAncestralProposer(self.world_)
