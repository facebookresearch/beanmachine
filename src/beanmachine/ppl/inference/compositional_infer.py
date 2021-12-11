# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections import defaultdict
from typing import (
    Dict,
    Tuple,
    Callable,
    Union,
    List,
    Set,
    Optional,
    TYPE_CHECKING,
    cast,
)

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


def _get_rv_family(rv_wrapper: Callable) -> Callable:
    """A helper function that return the unbounded function for a give random variable
    wrapper"""
    if inspect.ismethod(rv_wrapper):
        # For methods, we'll need to use the unbounded function instead of the
        # bounded method to determine which proposer to apply
        return cast(Callable, rv_wrapper.__func__)
    else:
        return rv_wrapper


def _get_nodes_for_rv_family(
    rv_families: Union[Callable, Tuple[Callable, ...]],
    rv_family_to_node: Dict[Callable, Set[RVIdentifier]],
) -> Set[RVIdentifier]:
    """A helper function that returns a list of nodes that belong to a particular RV
    family (or a particular tuple of RV families)"""
    # collect all nodes that belong to rv_families
    families = {rv_families} if isinstance(rv_families, Callable) else set(rv_families)
    nodes = set().union(*(rv_family_to_node.get(family, set()) for family in families))
    return nodes


class CompositionalInference(BaseInference):
    """
    The ``CompositionalInference`` class enables combining multiple inference algorithms
    and blocking random variables together. By default, it uses different proposer to
    update each site by (by look at the support of the distribution that that site).
    To override the default behavior, you can pass an ``inference_dict``. To learn more
    about Compositional Inference, please see the `Compositional Inference
    <https://beanmachine.org/docs/compositional_inference/>`_ page on our website.

    Example 0 (use different inference method for different random variable families)::

        CompositionalInference({
            model.foo: bm.SingleSiteAncestralMetropolisHastings(),
            model.bar: bm.SingleSiteNewtonianMonteCarlo(),
        })

    Example 1 (override default inference method)::

        CompositionalInference({...: bm.SingleSiteAncestralMetropolisHastings()})

    Example 2 (block inference (jointly propose) ``model.foo`` and ``model.bar``)::

        CompositionalInference({(model.foo, model.bar): bm.GlobalNoUTurnSampler()})

    Args:
        inference_dict: an optional inference configuration as shown above.
    """

    def __init__(
        self,
        inference_dict: Optional[
            Dict[
                Union[Callable, Tuple[Callable, ...], EllipsisClass],
                Union[BaseInference, Tuple[BaseInference, ...], EllipsisClass],
            ]
        ] = None,
    ):
        self.config: Dict[Union[Callable, Tuple[Callable, ...]], BaseInference] = {}
        # create a set for the RV families that are being covered in the config; this is
        # useful in get_proposers to determine which RV needs to be handle by the
        # default inference method
        self._covered_rv_families = set()

        default_inference = _DefaultInference()
        if inference_dict is not None:
            default_inference = inference_dict.pop(Ellipsis, default_inference)
            assert isinstance(default_inference, BaseInference)
            # preprocess inference dict
            for rv_families, inference in inference_dict.items():
                # parse key
                if isinstance(rv_families, Callable):
                    config_key = _get_rv_family(rv_families)
                    self._covered_rv_families.add(config_key)
                else:
                    # key is a tuple/block of families
                    config_key = tuple(map(_get_rv_family, rv_families))
                    self._covered_rv_families.update(config_key)

                # parse value
                if isinstance(inference, BaseInference):
                    config_val = inference
                elif inference == Ellipsis:
                    config_val = default_inference
                else:
                    # value is a tuple of inferences
                    assert isinstance(inference, tuple)
                    # there should be a one to one relationship between key and value
                    assert isinstance(config_key, tuple) and len(config_key) == len(
                        inference
                    )
                    # convert to an equivalent nested compositional inference
                    config_val = CompositionalInference(
                        {
                            rv_family: algorithm
                            for rv_family, algorithm in zip(config_key, inference)
                        }
                    )

                self.config[config_key] = config_val

        self._default_inference = default_inference

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

        all_proposers = []
        for target_families, inferences in self.config.items():
            nodes = _get_nodes_for_rv_family(target_families, rv_family_to_node)
            if len(nodes) > 0:
                proposers = inferences.get_proposers(world, nodes, num_adaptive_sample)
                if isinstance(target_families, tuple):
                    # tuple of RVs == block into a single accept/reject step
                    proposers = [SequentialProposer(proposers)]
                all_proposers.extend(proposers)

        # apply default proposers on nodes whose family are not covered by any of the
        # proposers listed in the config
        remaining_families = rv_family_to_node.keys() - self._covered_rv_families
        remaining_nodes = _get_nodes_for_rv_family(
            tuple(remaining_families), rv_family_to_node
        )
        if len(remaining_nodes) > 0:
            proposers = self._default_inference.get_proposers(
                world, remaining_nodes, num_adaptive_sample
            )
            all_proposers.extend(proposers)

        return all_proposers
