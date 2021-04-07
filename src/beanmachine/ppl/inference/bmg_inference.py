# Copyright (c) Facebook, Inc. and its affiliates.

"""An inference engine which uses Bean Machine Graph to make
inferences on Bean Machine models."""

from typing import Dict, List, Tuple

from beanmachine.graph import InferenceType
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.performance_report import PerformanceReport
from beanmachine.ppl.inference.abstract_infer import _verify_queries_and_observations
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this module.
# pyre-ignore-all-errors


class BMGInference:

    _fix_observe_true: bool = False

    def __init__(self):
        pass

    def _accumulate_graph(
        self, queries: List[RVIdentifier], observations: Dict[RVIdentifier, Tensor]
    ) -> BMGraphBuilder:
        _verify_queries_and_observations(queries, observations, True)
        bmg = BMGraphBuilder()
        bmg._fix_observe_true = self._fix_observe_true
        bmg.accumulate_graph(queries, observations)
        return bmg

    def _infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_samples: int,
        inference_type: InferenceType = InferenceType.NMC,
    ) -> Tuple[MonteCarloSamples, PerformanceReport]:
        bmg = self._accumulate_graph(queries, observations)
        return bmg._infer(num_samples, inference_type, True)

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_samples: int,
        inference_type: InferenceType = InferenceType.NMC,
    ) -> MonteCarloSamples:
        # TODO: Add num_chains
        # TODO: Add verbose level
        # TODO: Add logging
        return self._accumulate_graph(queries, observations).infer(
            num_samples, inference_type
        )

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
        edge_requirements = False
        bmg = self._accumulate_graph(queries, observations)
        return to_dot(
            bmg,
            graph_types,
            inf_types,
            edge_requirements,
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
