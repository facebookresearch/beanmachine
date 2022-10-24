# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
from typing import Dict, List

from beanmachine.ppl.compiler import bmg_nodes as bn

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problems import fix_problems
from beanmachine.ppl.compiler.internal_error import InternalError

_node_type_to_distribution = {
    bn.BernoulliNode: "torch.distributions.Bernoulli",
    bn.BetaNode: "torch.distributions.Beta",
    bn.NormalNode: "torch.distributions.Normal",
}

_node_type_to_operator = {
    bn.MultiplicationNode: "torch.multiply",
    bn.AdditionNode: "torch.add",
    bn.DivisionNode: "torch.div",
    bn.ToRealNode: "",
}


class ToBMPython:
    code: str
    _code: List[str]
    bmg: BMGraphBuilder
    node_to_var_id: Dict[bn.BMGNode, int]
    node_to_func_id: Dict[bn.BMGNode, int]
    dist_to_rv_id: Dict[bn.BMGNode, int]
    no_dist_samples: Dict[bn.BMGNode, int]
    queries: List[str]
    observations: List[str]

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.code = ""
        self._code = [
            "import beanmachine.ppl as bm",
            "import torch",
        ]
        self.bmg = bmg
        self.node_to_var_id = {}
        self.node_to_func_id = {}
        self.dist_to_rv_id = {}
        self.no_dist_samples = defaultdict(lambda: 0)
        self.queries = []
        self.observations = []

    def _get_node_id_mapping(self, node: bn.BMGNode) -> str:
        if node in self.node_to_var_id:
            return f"v{self.node_to_var_id[node]}"
        elif node in self.node_to_func_id:
            return f"f{self.node_to_func_id[node]}()"
        else:
            raise InternalError("Unsupported node type {node}")

    def _get_id(self) -> int:
        return len(self.node_to_var_id) + len(self.node_to_func_id)

    def _no_dist_samples(self, node: bn.DistributionNode) -> int:
        return sum(isinstance(o, bn.SampleNode) for o in node.outputs.items)

    def _inputs(self, node: bn.BMGNode) -> str:
        input_seq = []
        for x in node.inputs:
            if isinstance(x, bn.SampleNode):
                input_seq.append(
                    f"{self._get_node_id_mapping(x)}.wrapper(*{self._get_node_id_mapping(x)}.arguments)"
                )
            else:
                input_seq.append(self._get_node_id_mapping(x))
        inputs = ", ".join(input_seq)
        return inputs

    def _add_constant(self, node: bn.ConstantNode) -> None:
        var_id = self._get_id()
        self.node_to_var_id[node] = var_id
        t = type(node)
        v = node.value
        if (
            t is bn.PositiveRealNode
            or t is bn.NegativeRealNode
            or t is bn.ProbabilityNode
            or t is bn.RealNode
        ):
            f = str(float(v))
        elif t is bn.NaturalNode:
            f = str(int(v))
        else:
            f = str(float(v))
        self._code.append(f"v{var_id} = {f}")

    def _add_distribution(self, node: bn.DistributionNode) -> None:
        distr_type = _node_type_to_distribution[type(node)]  # pyre-ignore
        i = self._inputs(node)
        no_dist_samples = self._no_dist_samples(node)
        rv_id = len(self.dist_to_rv_id)
        self.dist_to_rv_id[node] = rv_id
        if no_dist_samples > 1:
            param = "i"
        else:
            param = ""
        self._code.append(
            f"@bm.random_variable\ndef rv{rv_id}({param}):\n\treturn {distr_type}({i})"
        )

    def _add_operator(self, node: bn.OperatorNode) -> None:
        var_id = self._get_id()
        operator_type = _node_type_to_operator[type(node)]  # pyre-ignore
        i = self._inputs(node)
        has_samples = any(isinstance(x, bn.SampleNode) for x in node.inputs)
        if has_samples:
            self.node_to_func_id[node] = var_id
            self._code.append(
                f"@bm.functional\ndef f{var_id}():\n\treturn {operator_type}({i})"
            )
        else:
            self.node_to_var_id[node] = var_id
            self._code.append(f"v{var_id} = {operator_type}({i})")

    def _add_sample(self, node: bn.SampleNode) -> None:
        var_id = self._get_id()
        self.node_to_var_id[node] = var_id
        rv_id = self.dist_to_rv_id[node.operand]
        self.no_dist_samples[node.operand] += 1
        total_samples = self._no_dist_samples(node.operand)
        if total_samples > 1:
            param = f"{self.no_dist_samples[node.operand]}"
        else:
            param = ""
        self._code.append(f"v{var_id} = rv{rv_id}({param})")

    def _add_query(self, node: bn.Query) -> None:
        self.queries.append(f"{self._get_node_id_mapping(node.operator)}")

    def _add_observation(self, node: bn.Observation) -> None:
        val = node.value
        # We need this cast since BMG requires boolean observations to be True/False
        # TODO: Implement selective graph fixing depending on the backend
        if isinstance(val, bool):
            val = float(val)
        self.observations.append(
            f"{self._get_node_id_mapping(node.observed)} : torch.tensor({val})"
        )

    def _generate_python(self, node: bn.BMGNode) -> None:
        if isinstance(node, bn.ConstantNode):
            self._add_constant(node)
        elif isinstance(node, bn.DistributionNode):
            self._add_distribution(node)
        elif isinstance(node, bn.SampleNode):
            self._add_sample(node)
        elif isinstance(node, bn.OperatorNode):
            self._add_operator(node)
        elif isinstance(node, bn.Query):
            self._add_query(node)
        elif isinstance(node, bn.Observation):
            self._add_observation(node)

    def _generate_bm_python(self) -> str:
        bmg, error_report = fix_problems(self.bmg)
        self.bmg = bmg
        error_report.raise_errors()
        for node in self.bmg.all_ancestor_nodes():
            self._generate_python(node)
        self._code.append(f"queries = [{(','.join(self.queries))}]")
        self._code.append(f"observations = {{{','.join(self.observations)}}}")
        self.code = "\n".join(self._code)
        return self.code


def to_bm_python(bmg: BMGraphBuilder) -> str:
    bmp = ToBMPython(bmg)
    return bmp._generate_bm_python()
