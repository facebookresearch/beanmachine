# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

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


def _value_to_cpp_eigen(value: torch.Tensor, variable: str) -> str:
    # Torch tensors are row major but Eigen matrices are column major;
    # a torch Dirichlet distribution expects a row of parameters;
    # BMG expects a column.  That's why we swap rows with columns here.
    r, c = _size_to_rc(value.size())
    v = value.reshape(r, c).transpose(0, 1).contiguous()
    values = ", ".join(str(element) for element in v.reshape(-1).tolist())
    return f"Eigen::MatrixXd {variable}({c}, {r})\n{variable} << {values};"


def _value_to_cpp(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        values = ",".join(str(element) for element in value.reshape(-1).tolist())
        dims = ",".join(str(dim) for dim in value.shape)
        return f"torch::from_blob((float[]){{{values}}}, {{{dims}}})"
    return str(value).lower()


class GeneratedGraphCPP:
    code: str
    _code: List[str]
    bmg: BMGraphBuilder
    node_to_graph_id: Dict[bn.BMGNode, int]
    query_to_query_id: Dict[bn.Query, int]
    _observation_count: int

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.code = ""
        self._code = ["graph::Graph g;"]
        self.bmg = bmg
        self.node_to_graph_id = {}
        self.query_to_query_id = {}
        self._observation_count = 0

    def _add_observation(self, node: bn.Observation) -> None:
        graph_id = self.node_to_graph_id[node.observed]
        v = node.value
        if isinstance(v, torch.Tensor):
            o = f"o{self._observation_count}"
            self._observation_count += 1
            s = _value_to_cpp_eigen(v, o)
            self._code.append(s)
            self._code.append(f"g.observe([n{graph_id}], {o});")
        else:
            self._code.append(f"g.observe([n{graph_id}], {_value_to_cpp(v)});")

    def _add_query(self, node: bn.Query) -> None:
        query_id = len(self.query_to_query_id)
        self.query_to_query_id[node] = query_id
        graph_id = self.node_to_graph_id[node.operator]
        self._code.append(f"uint q{query_id} = g.query(n{graph_id});")

    def _inputs(self, node: bn.BMGNode) -> str:
        inputs = ", ".join("n" + str(self.node_to_graph_id[x]) for x in node.inputs)
        return "std::vector<uint>({" + inputs + "})"

    def _add_factor(self, node: bn.FactorNode) -> None:
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        i = self._inputs(node)
        ft = str(factor_type(node)).replace(".", "::")
        self._code.append(f"uint n{graph_id} = g.add_factor(")
        self._code.append(f"  graph::{ft},")
        self._code.append(f"  {i});")

    def _add_distribution(self, node: bn.DistributionNode) -> None:
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        i = self._inputs(node)
        if isinstance(node, bn.DirichletNode):
            t = LatticeTyper()[node]
            assert isinstance(t, SimplexMatrix)
            self._code.append(f"uint n{graph_id} = g.add_distribution(")
            self._code.append("  graph::DistributionType::DIRICHLET,")
            self._code.append("  graph::ValueType(")
            self._code.append("    graph::VariableType::COL_SIMPLEX_MATRIX,")
            self._code.append("    graph::AtomicType::PROBABILITY,")
            self._code.append(f"    {t.rows},")
            self._code.append(f"    {t.columns}")
            self._code.append("  ),")
            self._code.append(f"  {i});")
        else:
            distr_type, elt_type = dist_type(node)
            distr_type = str(distr_type).replace(".", "::")
            elt_type = str(elt_type).replace(".", "::")
            self.node_to_graph_id[node] = graph_id
            self._code.append(f"uint n{graph_id} = g.add_distribution(")
            self._code.append(f"  graph::{distr_type},")
            self._code.append(f"  graph::{elt_type},")
            self._code.append(f"  {i});")

    def _add_operator(self, node: bn.OperatorNode) -> None:
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        i = self._inputs(node)
        ot = str(operator_type(node)).replace(".", "::")
        self._code.append(f"uint n{graph_id} = g.add_operator(")
        if len(node.inputs) <= 2:
            self._code.append(f"  graph::{ot}, {i});")
        else:
            self._code.append(f"  graph::{ot},")
            self._code.append(f"  {i});")

    def _add_constant(self, node: bn.ConstantNode) -> None:  # noqa
        graph_id = len(self.node_to_graph_id)
        self.node_to_graph_id[node] = graph_id
        t = type(node)
        v = node.value
        m = ""
        if t is bn.PositiveRealNode:
            f = f"add_constant_pos_real({_value_to_cpp(float(v))})"
        elif t is bn.NegativeRealNode:
            f = f"add_constant_neg_real({_value_to_cpp(float(v))})"
        elif t is bn.ProbabilityNode:
            f = f"add_constant_probability({_value_to_cpp(float(v))})"
        elif t is bn.BooleanNode:
            f = f"add_constant({_value_to_cpp(bool(v))})"
        elif t is bn.NaturalNode:
            f = f"add_constant({_value_to_cpp(int(v))})"
        elif t is bn.RealNode:
            f = f"add_constant({_value_to_cpp(float(v))})"
        elif t is bn.ConstantTensorNode:
            f = f"add_constant({_value_to_cpp(v)})"
        elif t is bn.ConstantPositiveRealMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_pos_matrix(m{graph_id})"
        elif t is bn.ConstantRealMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_real_matrix(m{graph_id})"
        elif t is bn.ConstantNegativeRealMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_neg_matrix(m{graph_id})"
        elif t is bn.ConstantProbabilityMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_probability_matrix(m{graph_id})"
        elif t is bn.ConstantSimplexMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_col_simplex_matrix(m{graph_id})"
        elif t is bn.ConstantNaturalMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_natural_matrix(m{graph_id})"
        elif t is bn.ConstantBooleanMatrixNode:
            m = _value_to_cpp_eigen(v, f"m{graph_id}")
            f = f"add_constant_bool_matrix(m{graph_id})"
        else:
            f = "UNKNOWN"
        if m != "":
            self._code.append(m)
        self._code.append(f"uint n{graph_id} = g.{f};")

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

    def _generate_cpp(self) -> None:
        fix_problems(self.bmg).raise_errors()
        for node in self.bmg.all_ancestor_nodes():
            self._generate_node(node)
        self.code = "\n".join(self._code)


def to_bmg_cpp(bmg: BMGraphBuilder) -> GeneratedGraphCPP:
    gg = GeneratedGraphCPP(bmg)
    gg._generate_cpp()
    return gg
