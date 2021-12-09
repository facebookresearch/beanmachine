# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Dict, List

import torch.distributions as dist
from beanmachine.ppl.legacy.inference.abstract_mh_infer import AbstractMHInference
from beanmachine.ppl.legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_newtonian_monte_carlo_proposer import (
    SingleSiteNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.utils import is_constraint_eq


LOGGER = logging.getLogger("beanmachine")


class CompositionalInference(AbstractMHInference):
    """
    Compositional inference
    """

    # pyre-fixme[9]: proposers has type `Dict[typing.Any, typing.Any]`; used as `None`.
    def __init__(self, proposers: Dict = None):
        self.proposers_per_family_ = {}
        self.proposers_per_rv_ = {}
        super().__init__()
        # for setting the transform properly during initialization in Variable.py
        # NMC requires an additional transform from Beta -> Reshaped beta
        # so all nodes default to having this behavior unless otherwise specified using CI
        # should be updated as initialization gets moved to the proposer
        self.world_.set_all_nodes_proposer(SingleSiteNewtonianMonteCarloProposer())
        if proposers is not None:
            for key in proposers:
                if hasattr(key, "__func__"):
                    func_wrapper = key.__func__
                    self.proposers_per_family_[key.__func__] = proposers[key]
                    self.world_.set_transforms(
                        func_wrapper,
                        proposers[key].transform_type,
                        proposers[key].transforms,
                    )
                    self.world_.set_proposer(func_wrapper, proposers[key])
                else:
                    self.proposers_per_family_[key] = proposers[key]

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

        wrapped_fn = node.wrapper
        if wrapped_fn in self.proposers_per_family_:
            proposer_inst = self.proposers_per_family_[wrapped_fn]
            self.proposers_per_rv_[node] = copy.deepcopy(proposer_inst)
            return self.proposers_per_rv_[node]

        node_var = self.world_.get_node_in_world(node, False)
        # pyre-fixme
        distribution = node_var.distribution
        support = distribution.support
        if any(
            is_constraint_eq(
                support,
                (
                    dist.constraints.real,
                    dist.constraints.simplex,
                    dist.constraints.greater_than,
                ),
            )
        ):
            self.proposers_per_rv_[node] = SingleSiteNewtonianMonteCarloProposer()
        elif is_constraint_eq(
            support, dist.constraints.integer_interval
        ) and isinstance(distribution, dist.Categorical):
            self.proposers_per_rv_[node] = SingleSiteUniformProposer()
        elif is_constraint_eq(support, dist.constraints.boolean) and isinstance(
            distribution, dist.Bernoulli
        ):
            self.proposers_per_rv_[node] = SingleSiteUniformProposer()
        else:
            LOGGER.warning(
                "Node {n} has unsupported constraints. ".format(n=node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            self.proposers_per_rv_[node] = SingleSiteAncestralProposer()
        return self.proposers_per_rv_[node]
