# Copyright (c) Facebook, Inc. and its affiliates.

"""An inference engine which uses Bean Machine Graph to make
inferences on Bean Machine models."""

from typing import Dict, List

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.inference.abstract_infer import _verify_queries_and_observations
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class BMGInference:
    def __init__(self):
        pass

    def _accumulate_graph(
        self, queries: List[RVIdentifier], observations: Dict[RVIdentifier, Tensor]
    ) -> BMGraphBuilder:
        _verify_queries_and_observations(queries, observations, True)
        bmg = BMGraphBuilder()
        bmg.accumulate_graph(queries, observations)
        return bmg

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_samples: int,
    ) -> MonteCarloSamples:
        # TODO: Add num_chains
        # TODO: Add verbose level
        # TODO: Add logging
        return self._accumulate_graph(queries, observations).infer(num_samples)

    def to_dot(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        after_transform: bool = True,
        label_edges: bool = False,
    ) -> str:
        """Produce a string containing a program in the GraphViz DOT language
        representing the graph deduced from the model."""
        graph_types = False
        inf_types = False
        point_at_input = True
        edge_requirements = False
        return self._accumulate_graph(queries, observations).to_dot(
            graph_types,
            inf_types,
            edge_requirements,
            point_at_input,
            after_transform,
            label_edges,
        )

    def to_cpp(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
    ) -> str:
        """Produce a string containing a C++ program fragment which
        produces the graph deduced from the model."""
        return self._accumulate_graph(queries, observations).to_cpp()

    def to_python(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
    ) -> str:
        """Produce a string containing a Python program fragment which
        produces the graph deduced from the model."""
        return self._accumulate_graph(queries, observations).to_python()
