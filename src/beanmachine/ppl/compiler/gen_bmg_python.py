# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import beanmachine.ppl.compiler.bmg_nodes as bn
import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_node_types import (
    dist_type,
    factor_type,
    operator_type,
)
from beanmachine.ppl.compiler.bmg_types import _size_to_rc, SimplexMatrix
from beanmachine.ppl.compiler.fix_problems import fix_problems
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


def _tensor_to_python(t: torch.Tensor) -> str:
    if len(t.shape) == 0:
        return str(t.item())
    return "[" + ",".join(_tensor_to_python(c) for c in t) + "]"


def _matrix_to_python(value: torch.Tensor) -> str:
    r, c = _size_to_rc(value.size())
    v = value.reshape(r, c).transpose(0, 1)
    t = _tensor_to_python(v)
    return f"tensor({t})"


class GeneratedGraphPython:
    code: str
    _code: List[str]
    bmg: BMGraphBuilder
    node_to_graph_id: Dict[bn.BMGNode, int]
    query_to_query_id: Dict[bn.Query, int]

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.code = ""
        self._code = [
            "from beanmachine import graph",
            "from torch import tensor",
            "g = graph.Graph()",
        ]
        self.bmg = bmg
        self.node_to_graph_id = {}
        self.query_to_query_id = {}

    def _add_observation(self, node: bn.Observation) -> None:
        graph_id = self.node_to_graph_id[node.observed]
        self._code.append(f"g.observe(n{graph_id}, {node.value})")

    def _add_query(self, node: bn.Query) -> None:
        query_id = len(self.query_to_query_id)
        self.query_to_query_id[node] = query_id
        graph_id = self.node_to_graph_id[node.operator]
        self._code.append(f"q{query_id} = g.query(n{graph_id})")

    def _inputs(self, node: bn.BMGNode) -> str:
        inputs = ", ".join("n" + str(self.node_to_graph_id[x]) for x in node.inputs)
        return "[" + inputs + "]"

    def _add_factor(self, node: bn.FactorNode) -> None:
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        i = self._inputs(node)
        ft = str(factor_type(node))
        self._code.append(f"n{graph_id} = g.add_factor(")
        self._code.append(f"  graph.{ft},")
        self._code.append(f"  {i},")
        self._code.append(")")

    def _add_distribution(self, node: bn.DistributionNode) -> None:
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        i = self._inputs(node)
        if isinstance(node, bn.DirichletNode):
            t = LatticeTyper()[node]
            assert isinstance(t, SimplexMatrix)
            self._code.append(f"n{graph_id} = g.add_distribution(")
            self._code.append("  graph.DistributionType.DIRICHLET,")
            self._code.append("  graph.ValueType(")
            self._code.append("    graph.VariableType.COL_SIMPLEX_MATRIX,")
            self._code.append("    graph.AtomicType.PROBABILITY,")
            self._code.append(f"    {t.rows},")
            self._code.append(f"    {t.columns},")
            self._code.append("  ),")
            self._code.append(f"  {i},")
            self._code.append(")")
        else:
            distr_type, elt_type = dist_type(node)
            self.node_to_graph_id[node] = graph_id
            self._code.append(f"n{graph_id} = g.add_distribution(")
            self._code.append(f"  graph.{distr_type},")
            self._code.append(f"  graph.{elt_type},")
            self._code.append(f"  {i},")
            self._code.append(")")

    def _add_operator(self, node: bn.OperatorNode) -> None:
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        i = self._inputs(node)
        ot = str(operator_type(node))
        if len(node.inputs) <= 2:
            self._code.append(f"n{graph_id} = g.add_operator(graph.{ot}, {i})")
        else:
            self._code.append(f"n{graph_id} = g.add_operator(")
            self._code.append(f"  graph.{ot},")
            self._code.append(f"  {i},")
            self._code.append(")")

    def _add_constant(self, node: bn.ConstantNode) -> None:  # noqa
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        t = type(node)
        v = node.value
        if t is bn.PositiveRealNode:
            f = f"add_constant_pos_real({str(float(v))})"
        elif t is bn.NegativeRealNode:
            f = f"add_constant_neg_real({str(float(v))})"
        elif t is bn.ProbabilityNode:
            f = f"add_constant_probability({str(float(v))})"
        elif t is bn.BooleanNode:
            f = f"add_constant_bool({str(bool(v))})"
        elif t is bn.NaturalNode:
            f = f"add_constant_natural({str(int(v))})"
        elif t is bn.RealNode:
            f = f"add_constant_real({str(float(v))})"
        elif t is bn.ConstantPositiveRealMatrixNode:
            f = f"add_constant_pos_matrix({_matrix_to_python(v)})"
        elif t is bn.ConstantRealMatrixNode:
            f = f"add_constant_real_matrix({_matrix_to_python(v)})"
        elif t is bn.ConstantNegativeRealMatrixNode:
            f = f"add_constant_neg_matrix({_matrix_to_python(v)})"
        elif t is bn.ConstantProbabilityMatrixNode:
            f = f"add_constant_probability_matrix({_matrix_to_python(v)})"
        elif t is bn.ConstantSimplexMatrixNode:
            f = f"add_constant_col_simplex_matrix({_matrix_to_python(v)})"
        elif t is bn.ConstantNaturalMatrixNode:
            f = f"add_constant_natural_matrix({_matrix_to_python(v)})"
        elif t is bn.ConstantBooleanMatrixNode:
            f = f"add_constant_bool_matrix({_matrix_to_python(v)})"
        elif isinstance(v, torch.Tensor) and v.numel() != 1:
            f = f"add_constant_real_matrix({_matrix_to_python(v)})"
        else:
            f = f"add_constant_real({str(float(v))})"
        self._code.append(f"n{graph_id} = g.{f}")

    def _generate_node(self, node: bn.BMGNode) -> None:
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

    def _generate_python(self) -> None:
        fix_problems(self.bmg).raise_errors()
        for node in self.bmg.all_ancestor_nodes():
            self._generate_node(node)
        self.code = "\n".join(self._code)


def to_bmg_python(bmg: BMGraphBuilder) -> GeneratedGraphPython:
    gg = GeneratedGraphPython(bmg)
    gg._generate_python()
    return gg
