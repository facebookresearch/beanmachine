# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""An inference engine which uses Bean Machine to make
inferences on optimized Bean Machine models."""

from typing import Dict, List, Set

import graphviz
import torch

from beanmachine.ppl.compiler.fix_problems import default_skip_optimizations
from beanmachine.ppl.compiler.gen_bm_python import InferenceType, to_bm_python
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.gen_mini import to_mini
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.utils import _verify_queries_and_observations
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class BMInference:
    """
    Interface to Bean Machine Inference on optimized models.

    Please note that this is a highly experimental implementation under active
    development, and that the subset of Bean Machine model is limited. Limitations
    include that the runtime graph should be static (meaning, it does not change
    during inference), and that the types of primitive distributions supported
    is currently limited.
    """

    _fix_observe_true: bool = False
    _infer_config = {}

    def __init__(self):
        pass

    def _accumulate_graph(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> BMGRuntime:
        _verify_queries_and_observations(queries, observations, True)
        rt = BMGRuntime()
        bmg = rt.accumulate_graph(queries, observations)
        # TODO: Figure out a better way to pass this flag around
        bmg._fix_observe_true = self._fix_observe_true
        return rt

    def _build_mcsamples(
        self,
        queries,
        opt_rv_to_query_map,
        samples,
    ) -> MonteCarloSamples:
        assert len(samples) == len(queries)

        results: Dict[RVIdentifier, torch.Tensor] = {}
        for rv in samples.keys():
            query = opt_rv_to_query_map[rv.__str__()]
            results[query] = samples[rv]
        mcsamples = MonteCarloSamples(results)
        return mcsamples

    def _infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        num_chains: int = 1,
        num_adaptive_samples: int = 0,
        inference_type: InferenceType = InferenceType.GlobalNoUTurnSampler,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> MonteCarloSamples:

        rt = self._accumulate_graph(queries, observations)
        bmg = rt._bmg

        self._infer_config["num_samples"] = num_samples
        self._infer_config["num_chains"] = num_chains
        self._infer_config["num_adaptive_samples"] = num_adaptive_samples

        generated_graph = to_bmg_graph(bmg, skip_optimizations)
        optimized_python, opt_rv_to_query_map = to_bm_python(
            generated_graph.bmg, inference_type, self._infer_config
        )

        try:
            exec(optimized_python, globals())  # noqa
        except RuntimeError as e:
            raise RuntimeError("Error during BM inference\n") from e

        opt_samples = self._build_mcsamples(
            queries,
            opt_rv_to_query_map,
            # pyre-ignore
            samples,  # noqa
        )
        return opt_samples

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        num_chains: int = 4,
        num_adaptive_samples: int = 0,
        inference_type: InferenceType = InferenceType.GlobalNoUTurnSampler,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> MonteCarloSamples:
        """
        Perform inference by (runtime) compilation of Python source code associated
        with its parameters, constructing an optimized BM graph, and then calling the
        BM implementation of a particular inference method on this graph.

        Args:
            queries: queried random variables
            observations: observations dict
            num_samples: number of samples in each chain
            num_chains: number of chains generated
            num_adaptive_samples: number of burn in samples to discard
            inference_type: inference method
            skip_optimizations: list of optimization to disable in this call

        Returns:
            MonteCarloSamples: The requested samples
        """
        # TODO: Add verbose level
        # TODO: Add logging
        samples = self._infer(
            queries,
            observations,
            num_samples,
            num_chains,
            num_adaptive_samples,
            inference_type,
            skip_optimizations,
        )
        return samples

    def to_dot(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        after_transform: bool = True,
        label_edges: bool = False,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> str:
        """Produce a string containing a program in the GraphViz DOT language
        representing the graph deduced from the model."""
        node_types = False
        node_sizes = False
        edge_requirements = False
        bmg = self._accumulate_graph(queries, observations)._bmg
        return to_dot(
            bmg,
            node_types,
            node_sizes,
            edge_requirements,
            after_transform,
            label_edges,
            skip_optimizations,
        )

    def _to_mini(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        indent=None,
    ) -> str:
        """Internal test method for Neal's MiniBMG prototype."""
        bmg = self._accumulate_graph(queries, observations)._bmg
        return to_mini(bmg, indent)

    def to_graphviz(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        after_transform: bool = True,
        label_edges: bool = False,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> graphviz.Source:
        """Small wrapper to generate an actual graphviz object"""
        s = self.to_dot(
            queries, observations, after_transform, label_edges, skip_optimizations
        )
        return graphviz.Source(s)

    def to_python(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        num_chains: int = 4,
        num_adaptive_samples: int = 0,
        inference_type: InferenceType = InferenceType.GlobalNoUTurnSampler,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> str:
        """Produce a string containing a BM Python program from the graph."""
        bmg = self._accumulate_graph(queries, observations)._bmg
        self._infer_config["num_samples"] = num_samples
        self._infer_config["num_chains"] = num_chains
        self._infer_config["num_adaptive_samples"] = num_adaptive_samples
        opt_bm, _ = to_bm_python(bmg, inference_type, self._infer_config)
        return opt_bm
