# Copyright (c) Facebook, Inc. and its affiliates.

"""An inference engine which uses Bean Machine Graph to make
inferences on Bean Machine models."""

from typing import Dict, List

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from torch import Tensor


class BMGInference:
    def __init__(self):
        pass

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_samples: int,
    ) -> MonteCarloSamples:
        # TODO: Add num_chains
        # TODO: Add verbose level
        # TODO: Add logging
        bmg = BMGraphBuilder()
        bmg.accumulate_graph(queries, observations)
        return bmg.infer(num_samples)

    def to_dot(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        after_transform: bool = True,
        label_edges: bool = False,
    ) -> str:
        bmg = BMGraphBuilder()
        bmg.accumulate_graph(queries, observations)
        graph_types = False
        inf_types = False
        point_at_input = True
        edge_requirements = False
        return bmg.to_dot(
            graph_types,
            inf_types,
            edge_requirements,
            point_at_input,
            after_transform,
            label_edges,
        )
