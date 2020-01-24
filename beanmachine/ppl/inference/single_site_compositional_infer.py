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

    # pyre-fixme[9]: proposers has type `Dict[typing.Any, typing.Any]`; used as `None`.
    def __init__(self, proposers: Dict = None, should_transform: bool = False):
        self.proposers_ = {}
        if proposers is not None:
            for key in proposers:
                self.proposers_[key.__name__] = proposers[key]
        self.should_transform_ = should_transform
        self.newtonian_monte_carlo_proposer_ = SingleSiteNewtonianMonteCarloProposer()
        self.uniform_proposer_ = SingleSiteUniformProposer()
        self.ancestral_proposer_ = SingleSiteAncestralProposer()
        super().__init__()

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node given the proposer dicts passed in
        once instantiating the class.

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        if node.function.__name__ in self.proposers_:
            return self.proposers_[node.function.__name__]

        node_var = self.world_.get_node_in_world(node, False)
        distribution = node_var.distribution
        support = distribution.support
        if isinstance(support, dist.constraints._Real) or isinstance(
            support, dist.constraints._Simplex
        ):
            return self.newtonian_monte_carlo_proposer_
        if isinstance(support, dist.constraints._IntegerInterval) and isinstance(
            distribution, dist.Categorical
        ):
            return self.uniform_proposer_
        if isinstance(support, dist.constraints._Boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            return self.uniform_proposer_
        return self.ancestral_proposer_
