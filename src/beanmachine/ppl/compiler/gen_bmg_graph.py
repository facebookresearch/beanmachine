# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Set

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.profiler as prof
import torch
from beanmachine.graph import Graph
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_node_types import (
    dist_type,
    factor_type,
    operator_type,
)
from beanmachine.ppl.compiler.bmg_types import _size_to_rc
from beanmachine.ppl.compiler.fix_problems import (
    default_skip_optimizations,
    fix_problems,
)


def _reshape(t: torch.Tensor):
    # Note that we take the transpose; BMG expects columns,
    # BM provides rows.
    r, c = _size_to_rc(t.size())
    return t.reshape(r, c).transpose(0, 1)


class GeneratedGraph:
    graph: Graph
    bmg: BMGraphBuilder
    node_to_graph_id: Dict[bn.BMGNode, int]
    query_to_query_id: Dict[bn.Query, int]

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.graph = Graph()
        self.bmg = bmg
        self.node_to_graph_id = {}
        self.query_to_query_id = {}

    def _add_observation(self, node: bn.Observation) -> None:
        self.graph.observe(self.node_to_graph_id[node.observed], node.value)

    def _add_query(self, node: bn.Query) -> None:
        query_id = self.graph.query(self.node_to_graph_id[node.operator])
        self.query_to_query_id[node] = query_id

    def _inputs(self, node: bn.BMGNode) -> List[int]:
        return [self.node_to_graph_id[x] for x in node.inputs]

    def _add_factor(self, node: bn.FactorNode) -> None:
        graph_id = self.graph.add_factor(factor_type(node), self._inputs(node))
        self.node_to_graph_id[node] = graph_id

    def _add_distribution(self, node: bn.DistributionNode) -> None:
        distr_type, elt_type = dist_type(node)
        graph_id = self.graph.add_distribution(distr_type, elt_type, self._inputs(node))
        self.node_to_graph_id[node] = graph_id

    def _add_operator(self, node: bn.OperatorNode) -> None:
        graph_id = self.graph.add_operator(operator_type(node), self._inputs(node))
        self.node_to_graph_id[node] = graph_id

    def _add_constant(self, node: bn.ConstantNode) -> None:  # noqa
        t = type(node)
        v = node.value
        if t is bn.PositiveRealNode:
            graph_id = self.graph.add_constant_pos_real(float(v))
        elif t is bn.NegativeRealNode:
            graph_id = self.graph.add_constant_neg_real(float(v))
        elif t is bn.ProbabilityNode:
            graph_id = self.graph.add_constant_probability(float(v))
        elif t is bn.BooleanNode:
            graph_id = self.graph.add_constant_bool(bool(v))
        elif t is bn.NaturalNode:
            graph_id = self.graph.add_constant_natural(int(v))
        elif t is bn.RealNode:
            graph_id = self.graph.add_constant_real(float(v))
        elif t is bn.ConstantPositiveRealMatrixNode:
            graph_id = self.graph.add_constant_pos_matrix(_reshape(v))
        elif t is bn.ConstantRealMatrixNode:
            graph_id = self.graph.add_constant_real_matrix(_reshape(v))
        elif t is bn.ConstantNegativeRealMatrixNode:
            graph_id = self.graph.add_constant_neg_matrix(_reshape(v))
        elif t is bn.ConstantProbabilityMatrixNode:
            graph_id = self.graph.add_constant_probability_matrix(_reshape(v))
        elif t is bn.ConstantSimplexMatrixNode:
            graph_id = self.graph.add_constant_col_simplex_matrix(_reshape(v))
        elif t is bn.ConstantNaturalMatrixNode:
            graph_id = self.graph.add_constant_natural_matrix(_reshape(v))
        elif t is bn.ConstantBooleanMatrixNode:
            graph_id = self.graph.add_constant_bool_matrix(_reshape(v))
        elif isinstance(v, torch.Tensor) and v.numel() != 1:
            graph_id = self.graph.add_constant_real_matrix(_reshape(v))
        else:
            graph_id = self.graph.add_constant_real(float(v))
        self.node_to_graph_id[node] = graph_id

    def _generate_node(self, node: bn.BMGNode) -> None:
        # We add all nodes that are reachable from a query, observation or
        # sample to the BMG graph such that inputs are always added before
        # outputs.
        #
        # TODO: We could consider traversing only nodes reachable from
        # observations or queries.
        #
        # There are three cases to consider:
        #
        # * Observations: there is no associated value returned by the graph
        #   when we add an observation, so there is nothing to track.
        #
        # * Query of an operator (or constant): The graph gives us the column
        #   index in the list of samples it returns for this query. We track it in
        #   query_to_query_id.
        #
        # * Any other node: the graph gives us the graph identifier of the new
        #   node. We need to know this for each node that will be used as an input
        #   later, so we track that in node_to_graph_id.

        if isinstance(node, bn.Observation):
            self._add_observation(node)
        elif isinstance(node, bn.Query):
            self._add_query(node)
        elif isinstance(node, bn.FactorNode):
            self._add_factor(node)
        elif isinstance(node, bn.DistributionNode):
            self._add_distribution(node)
        elif isinstance(node, bn.OperatorNode):
            self._add_operator(node)
        elif isinstance(node, bn.ConstantNode):
            self._add_constant(node)

    def _generate_graph(self, skip_optimizations: Set[str]) -> None:
        fix_problems(self.bmg, skip_optimizations).raise_errors()
        self.bmg._begin(prof.build_bmg_graph)
        for node in self.bmg.all_ancestor_nodes():
            self._generate_node(node)
        self.bmg._finish(prof.build_bmg_graph)


def to_bmg_graph(
    bmg: BMGraphBuilder, skip_optimizations: Set[str] = default_skip_optimizations
) -> GeneratedGraph:
    gg = GeneratedGraph(bmg)
    gg._generate_graph(skip_optimizations)
    return gg
