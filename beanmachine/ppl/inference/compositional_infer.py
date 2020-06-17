# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from typing import Dict, List

import torch.distributions as dist
from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference
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


class CompositionalInference(AbstractMHInference):
    """
    Compositional inference
    """

    # pyre-fixme[9]: proposers has type `Dict[typing.Any, typing.Any]`; used as `None`.
    def __init__(self, proposers: Dict = None, should_transform: bool = False):
        self.proposers_per_family_ = {}
        self.proposers_per_rv_ = {}
        if proposers is not None:
            for key in proposers:
                if hasattr(key, "__func__"):
                    self.proposers_per_family_[key.__func__] = proposers[key]
                else:
                    self.proposers_per_family_[key] = proposers[key]

        self.should_transform_ = should_transform
        super().__init__()

    def add_sequential_proposer(self, block: List) -> None:
        """
        Adds a sequential block to list of blocks.

        :param block: list of random variables functions that are to be sampled
        together sequentially.
        """
        blocks = []
        for rv in block:
            if hasattr(rv, "__func__"):
                blocks.append(rv.__func__)
            else:
                blocks.append(rv)
        self.blocks_.append(blocks)

    def find_best_single_site_proposer(self, node: RVIdentifier):
        """
        Finds the best proposer for a node given the proposer dicts passed in
        once instantiating the class.

        :param node: the node for which to return a proposer
        :returns: a proposer for the node
        """
        if node in self.proposers_per_rv_:
            return self.proposers_per_rv_[node]

        if node.function._wrapper in self.proposers_per_family_:
            proposer_inst = self.proposers_per_family_[node.function._wrapper]
            self.proposers_per_rv_[node] = copy.deepcopy(proposer_inst)
            return self.proposers_per_rv_[node]

        node_var = self.world_.get_node_in_world(node, False)
        # pyre-fixme
        distribution = node_var.distribution
        support = distribution.support
        if (
            isinstance(support, dist.constraints._Real)
            or isinstance(support, dist.constraints._Simplex)
            or isinstance(support, dist.constraints._GreaterThan)
        ):
            self.proposers_per_rv_[node] = SingleSiteNewtonianMonteCarloProposer()
        elif isinstance(support, dist.constraints._IntegerInterval) and isinstance(
            distribution, dist.Categorical
        ):
            self.proposers_per_rv_[node] = SingleSiteUniformProposer()
        elif isinstance(support, dist.constraints._Boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            self.proposers_per_rv_[node] = SingleSiteUniformProposer()
        else:
            self.proposers_per_rv_[node] = SingleSiteAncestralProposer()
        return self.proposers_per_rv_[node]
