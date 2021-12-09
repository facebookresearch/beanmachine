# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections import defaultdict
from typing import Dict, Tuple, Callable, Union, List, Set, Optional, TYPE_CHECKING

import torch.distributions as dist
from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.inference.proposer.sequential_proposer import (
    SequentialProposer,
)
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.inference.single_site_nmc import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World
from beanmachine.ppl.world.utils import is_constraint_eq

if TYPE_CHECKING:
    from enum import Enum

    class EllipsisClass(Enum):
        Ellipsis = "..."

        def __iter__(self):
            pass

    Ellipsis = EllipsisClass.Ellipsis
else:
    EllipsisClass = type(Ellipsis)


class _DefaultInference(SingleSiteNewtonianMonteCarlo):
    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        proposers = []
        for node in target_rvs:
            if node not in self._proposers:
                # pyre-ignore[16]
                support = world.get_variable(node).distribution.support
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
                    self._proposers[node] = self._init_nmc_proposer(node, world)
                else:
                    self._proposers[node] = SingleSiteUniformProposer(node)
            proposers.append(self._proposers[node])
        return proposers


class CompositionalInference(BaseInference):
    def __init__(
        self,
        inference_dict: Optional[
            Dict[
                Union[Callable, Tuple[Callable, ...], EllipsisClass],
                Union[BaseInference, Tuple[BaseInference, ...]],
            ]
        ] = None,
    ):
        self.config = {}
        default_ = _DefaultInference()
        if inference_dict is not None:
            default_ = inference_dict.pop(Ellipsis, default_)
            for rv_families, inference in inference_dict.items():
                if isinstance(rv_families, Callable):
                    rv_families = (rv_families,)
                # For methods, we'll need to use the unbounded function instead of the
                # bounded method to determine which proposer to apply
                config_key = tuple(
                    family.__func__ if inspect.ismethod(family) else family
                    for family in rv_families
                )
                self.config[config_key] = inference

        self._default_inference = default_
        # create a set for the RV families that are being covered in the config; this is
        # useful in get_proposers to determine which RV needs to be handle by the
        # default inference method
        self._covered_rv_families = set().union(*self.config)

    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        # create a RV family to RVIdentifier lookup map
        rv_family_to_node = defaultdict(set)
        for node in target_rvs:
            rv_family_to_node[node.wrapper].add(node)

        def _get_proposers_for_inference(
            rv_families: Tuple[Callable, ...],
            inferences: Union[BaseInference, Tuple[BaseInference, ...]],
        ) -> List[BaseProposer]:
            """Given a tuple of random variable families, this helper function collect
            all nodes in world that belong to the families of random variables and
            invoke the inference to spawn proposers for them."""
            if isinstance(inferences, tuple):
                # each inference method is responsible for updating a corresponding
                # rv_family
                assert len(inferences) == len(rv_families)
                sub_proposers = []
                for rv_family, inference in zip(rv_families, inferences):
                    sub_proposers.extend(
                        _get_proposers_for_inference((rv_family,), inference)
                    )
                if len(sub_proposers) > 0:
                    return [SequentialProposer(sub_proposers)]
            else:
                # collect all nodes that belong to rv_families
                nodes = set().union(
                    *(rv_family_to_node.get(family, set()) for family in rv_families)
                )
                if len(nodes) > 0:
                    return inferences.get_proposers(world, nodes, num_adaptive_sample)
            return []

        proposers = []
        for target_families, inferences in self.config.items():
            proposers.extend(_get_proposers_for_inference(target_families, inferences))

        # apply default proposers on nodes whose family are not covered by any of the
        # proposers listed in the config
        remaining_families = rv_family_to_node.keys() - self._covered_rv_families
        proposers.extend(
            _get_proposers_for_inference(
                tuple(remaining_families), self._default_inference
            )
        )

        return proposers
