# Copyright (c) Facebook, Inc. and its affiliates.

"""An inference engine which uses Bean Machine Graph to make
inferences on Bean Machine models."""

from typing import Dict, List, Tuple

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.performance_report as pr
import beanmachine.ppl.compiler.profiler as prof
import torch
from beanmachine.graph import InferenceType  # pyre-ignore
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.performance_report import PerformanceReport
from beanmachine.ppl.inference.abstract_infer import _verify_queries_and_observations
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class BMGInference:

    _fix_observe_true: bool = False

    def __init__(self):
        pass

    def _accumulate_graph(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> BMGraphBuilder:
        _verify_queries_and_observations(queries, observations, True)
        bmg = BMGraphBuilder()
        bmg._fix_observe_true = self._fix_observe_true
        bmg.accumulate_graph(queries, observations)
        return bmg

    def _transpose_samples(self, bmg, raw):
        bmg.pd.begin(prof.transpose_samples)
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

        bmg.pd.finish(prof.transpose_samples)
        return samples

    def _build_mcsamples(
        self, bmg, samples, query_to_query_id, num_samples: int
    ) -> MonteCarloSamples:
        bmg.pd.begin(prof.build_mcsamples)

        result: Dict[RVIdentifier, torch.Tensor] = {}
        for (rv, query) in bmg._rv_to_query.items():
            if isinstance(query.operator, bn.ConstantNode):
                # TODO: Test this with tensor and normal constants
                result[rv] = torch.tensor([[query.operator.value] * num_samples])
            else:
                query_id = query_to_query_id[query]
                result[rv] = samples[query_id]
        mcsamples = MonteCarloSamples(result)

        bmg.pd.finish(prof.build_mcsamples)

        return mcsamples

    def _infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        inference_type: InferenceType = InferenceType.NMC,  # pyre-ignore
        produce_report: bool = True,
    ) -> Tuple[MonteCarloSamples, PerformanceReport]:
        bmg = self._accumulate_graph(queries, observations)

        report = pr.PerformanceReport()
        # TODO: Refactor performance data; should be owned by
        # BMGInference, not graph accumulator.

        bmg.pd.begin(prof.infer)

        generated_graph = to_bmg_graph(bmg)
        g = generated_graph.graph
        query_to_query_id = generated_graph.query_to_query_id

        samples = []

        # BMG requires that we have at least one query.
        if len(query_to_query_id) != 0:
            g.collect_performance_data(produce_report)
            bmg.pd.begin(prof.graph_infer)
            raw = g.infer(num_samples, inference_type)
            bmg.pd.finish(prof.graph_infer)
            if produce_report:
                bmg.pd.begin(prof.deserialize_perf_report)
                js = g.performance_report()
                report = pr.json_to_perf_report(js)
                bmg.pd.finish(prof.deserialize_perf_report)
            assert len(raw) == num_samples
            samples = self._transpose_samples(bmg, raw)

        mcsamples = self._build_mcsamples(bmg, samples, query_to_query_id, num_samples)

        bmg.pd.finish(prof.infer)
        report.profiler_report = bmg.pd.to_report()  # pyre-ignore

        return mcsamples, report

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
        num_samples: int,
        inference_type: InferenceType = InferenceType.NMC,
    ) -> MonteCarloSamples:
        # TODO: Add num_chains
        # TODO: Add verbose level
        # TODO: Add logging
        samples, _ = self._infer(
            queries, observations, num_samples, inference_type, False
        )
        return samples

    def to_dot(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
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
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> str:
        """Produce a string containing a C++ program fragment which
        produces the graph deduced from the model."""
        bmg = self._accumulate_graph(queries, observations)
        return to_bmg_cpp(bmg).code

    def to_python(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ) -> str:
        """Produce a string containing a Python program fragment which
        produces the graph deduced from the model."""
        bmg = self._accumulate_graph(queries, observations)
        return to_bmg_python(bmg).code


# TODO: Add a to_graph API here; make a map from
# query RVs to query ids and return it along with the graph.
