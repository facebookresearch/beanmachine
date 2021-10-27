import inspect
from collections import defaultdict
from typing import Dict, Tuple, Callable, Union, List, Set, Optional

from beanmachine.ppl.experimental.global_inference.base_inference import BaseInference
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import (
    SimpleWorld,
)
from beanmachine.ppl.experimental.global_inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class CompositionalInference(BaseInference):
    def __init__(
        self,
        inference_dict: Optional[
            Dict[Union[Callable, Tuple[Callable, ...]], BaseInference]
        ] = None,
    ):
        self.config = {}
        if inference_dict is not None:
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

        self._default_inference = SingleSiteAncestralMetropolisHastings()
        # create a set for the RV families that are being covered in the config; this is
        # useful in get_proposers to determine which RV needs to be handle by the
        # default inference method
        self._covered_rv_families = set().union(*self.config)

    def get_proposers(
        self,
        world: SimpleWorld,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        # create a RV family to RVIdentifier lookup map
        rv_family_to_node = defaultdict(set)
        for node in target_rvs:
            rv_family_to_node[node.wrapper].add(node)

        proposers = []
        for target_families, inference in self.config.items():
            nodes = set().union(
                *(rv_family_to_node.get(family, set()) for family in target_families)
            )
            if len(nodes) > 0:
                proposers.extend(
                    inference.get_proposers(world, nodes, num_adaptive_sample)
                )

        # apply default proposers on nodes whose family are not covered by any of the
        # proposers listed in the config
        remaining_families = rv_family_to_node.keys() - self._covered_rv_families
        remaining_nodes = set().union(
            *(rv_family_to_node[family] for family in remaining_families)
        )
        if len(remaining_nodes) > 0:
            proposers.extend(
                self._default_inference.get_proposers(
                    world, remaining_nodes, num_adaptive_sample
                )
            )
        return proposers
