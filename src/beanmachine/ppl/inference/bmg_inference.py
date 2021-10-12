# Copyright (c) Facebook, Inc. and its affiliates.

"""An inference engine which uses Bean Machine Graph to make
inferences on Bean Machine models."""

from typing import Dict, List, Optional, Set, Tuple

import beanmachine.ppl.compiler.performance_report as pr
import beanmachine.ppl.compiler.profiler as prof
import graphviz
import torch
from beanmachine.graph import Graph, InferenceType  # pyre-ignore
from beanmachine.ppl.compiler.fix_problems import default_skip_optimizations
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.performance_report import PerformanceReport
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.inference.abstract_infer import _verify_queries_and_observations
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class BMGInference:

    _fix_observe_true: bool = False
    _pd: Optional[prof.ProfilerData] = None

    def __init__(self):
        pass

    def _begin(self, s: str) -> None:
        pd = self._pd
        if pd is not None:
            pd.begin(s)

    def _finish(self, s: str) -> None:
        pd = self._pd
        if pd is not None:
            pd.finish(s)

    def _accumulate_graph(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> BMGRuntime:
        _verify_queries_and_observations(queries, observations, True)
        rt = BMGRuntime()
        rt._pd = self._pd
        bmg = rt.accumulate_graph(queries, observations)
        # TODO: Figure out a better way to pass this flag around
        bmg._fix_observe_true = self._fix_observe_true
        return rt

    def _transpose_samples(self, raw):
        self._begin(prof.transpose_samples)
        samples = []
        num_samples = len(raw)
        bmg_query_count = len(raw[0])

        # Suppose we have two queries and three samples;
        # the shape we get from BMG is:
        #
        # [
        #   [s00, s01],
        #   [s10, s11],
        #   [s20, s21]
        # ]
        #
        # That is, each entry in the list has values from both queries.
        # But what we need in the final dictionary is:
        #
        # {
        #   RV0: tensor([[s00, s10, s20]]),
        #   RV1: tensor([[s01, s11, s21]])
        # }

        transposed = [torch.tensor([x]) for x in zip(*raw)]
        assert len(transposed) == bmg_query_count
        assert len(transposed[0]) == 1
        assert len(transposed[0][0]) == num_samples

        # We now have
        #
        # [
        #   tensor([[s00, s10, s20]]),
        #   tensor([[s01, s11, s21]])
        # ]
        #
        # which looks like what we need. But we have an additional problem:
        # if the the sample is a matrix then it is in columns but we need it in rows.
        #
        # If an element of transposed is (1 x num_samples x rows x 1) then we
        # will just reshape it to (1 x num_samples x rows).
        #
        # If it is (1 x num_samples x rows x columns) for columns > 1 then
        # we transpose it to (1 x num_samples x columns x rows)
        #
        # If it is any other shape we leave it alone.

        for i in range(len(transposed)):
            t = transposed[i]
            if len(t.shape) == 4:
                if t.shape[3] == 1:
                    assert t.shape[0] == 1
                    assert t.shape[1] == num_samples
                    samples.append(t.reshape(1, num_samples, t.shape[2]))
                else:
                    samples.append(t.transpose(2, 3))
            else:
                samples.append(t)

        assert len(samples) == bmg_query_count
        assert len(samples[0]) == 1
        assert len(samples[0][0]) == num_samples

        self._finish(prof.transpose_samples)
        return samples

    def _build_mcsamples(
        self, rv_to_query, samples, query_to_query_id, num_samples: int
    ) -> MonteCarloSamples:
        self._begin(prof.build_mcsamples)

        result: Dict[RVIdentifier, torch.Tensor] = {}
        for (rv, query) in rv_to_query.items():
            query_id = query_to_query_id[query]
            result[rv] = samples[query_id]
        mcsamples = MonteCarloSamples(result)

        self._finish(prof.build_mcsamples)

        return mcsamples

    def _infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        inference_type: InferenceType = InferenceType.NMC,  # pyre-ignore
        produce_report: bool = True,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> Tuple[MonteCarloSamples, PerformanceReport]:
        if produce_report:
            self._pd = prof.ProfilerData()

        rt = self._accumulate_graph(queries, observations)
        bmg = rt._bmg
        report = pr.PerformanceReport()

        self._begin(prof.infer)

        generated_graph = to_bmg_graph(bmg, skip_optimizations)
        g = generated_graph.graph
        query_to_query_id = generated_graph.query_to_query_id

        samples = []

        # BMG requires that we have at least one query.
        if len(query_to_query_id) != 0:
            g.collect_performance_data(produce_report)
            self._begin(prof.graph_infer)
            raw = g.infer(num_samples, inference_type)
            self._finish(prof.graph_infer)
            if produce_report:
                self._begin(prof.deserialize_perf_report)
                js = g.performance_report()
                report = pr.json_to_perf_report(js)
                self._finish(prof.deserialize_perf_report)
            assert len(raw) == num_samples
            samples = self._transpose_samples(raw)

        # TODO: Make _rv_to_query public. Add it to BMGraphBuilder?
        mcsamples = self._build_mcsamples(
            rt._rv_to_query, samples, query_to_query_id, num_samples
        )

        self._finish(prof.infer)

        if produce_report:
            report.profiler_report = self._pd.to_report()  # pyre-ignore

        return mcsamples, report

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        inference_type: InferenceType = InferenceType.NMC,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> MonteCarloSamples:
        # TODO: Add num_chains
        # TODO: Add verbose level
        # TODO: Add logging
        samples, _ = self._infer(
            queries,
            observations,
            num_samples,
            inference_type,
            False,
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

    def to_graphviz(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        after_transform: bool = True,
        label_edges: bool = False,
        skip_optimizations: Set[str] = default_skip_optimizations,
    ) -> graphviz.files.Source:
        """Small wrapper to generate an actual graphviz object"""
        s = self.to_dot(
            queries, observations, after_transform, label_edges, skip_optimizations
        )
        return graphviz.Source(s)

    def to_cpp(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> str:
        """Produce a string containing a C++ program fragment which
        produces the graph deduced from the model."""
        bmg = self._accumulate_graph(queries, observations)._bmg
        return to_bmg_cpp(bmg).code

    def to_python(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> str:
        """Produce a string containing a Python program fragment which
        produces the graph deduced from the model."""
        bmg = self._accumulate_graph(queries, observations)._bmg
        return to_bmg_python(bmg).code

    def to_graph(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> Tuple[Graph, Dict[RVIdentifier, int]]:  # pyre-ignore
        """Produce a BMG graph and a map from queried RVIdentifiers to the corresponding
        indices of the inference results."""
        rt = self._accumulate_graph(queries, observations)
        bmg = rt._bmg
        generated_graph = to_bmg_graph(bmg)
        g = generated_graph.graph
        query_to_query_id = generated_graph.query_to_query_id
        rv_to_query = rt._rv_to_query
        rv_to_query_id = {rv: query_to_query_id[rv_to_query[rv]] for rv in queries}
        return g, rv_to_query_id
