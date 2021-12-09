# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert a builder into a program that produces a graph builder.
"""

from typing import Dict

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder


class GenerateBuilder:
    bmg: BMGraphBuilder
    _nodes: Dict[bn.BMGNode, int]

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.bmg = bmg
        self._nodes = {}

    def _factory_name(self, node: bn.BMGNode) -> str:
        return type(node).__name__

    def _to_id(self, index: int) -> str:
        return "n" + str(index)

    def _input_to_arg(self, node: bn.BMGNode) -> str:
        index = self._nodes[node]
        return self._to_id(index)

    def _inputs_to_args(self, node: bn.BMGNode) -> str:
        if isinstance(node, bn.ConstantNode):
            return str(node.value)
        arglist = ", ".join(self._input_to_arg(i) for i in node.inputs)
        if isinstance(node, bn.Observation):
            return f"{arglist}, {str(node.value)}"
        if isinstance(node, bn.LogSumExpNode):
            return f"[{arglist}]"
        if isinstance(node, bn.TensorNode):
            return f"[{arglist}], torch.Size([{str(len(node.inputs))}])"
        return arglist

    def _generate_builder(self) -> str:

        lines = [
            "import beanmachine.ppl.compiler.bmg_nodes as bn",
            "import torch",
            "from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder",
            "from torch import tensor",
            "",
            "bmg = BMGraphBuilder()",
        ]

        # Nodes should be sorted so that ancestors always come
        # before descendents.

        self._nodes = {}
        for index, node in enumerate(self.bmg.all_nodes()):
            self._nodes[node] = index

        for node, index in self._nodes.items():
            n = self._to_id(index)
            f = self._factory_name(node)
            a = self._inputs_to_args(node)
            lines.append(f"{n} = bmg.add_node(bn.{f}({a}))")
        return "\n".join(lines)


def generate_builder(bmg: BMGraphBuilder) -> str:
    return GenerateBuilder(bmg)._generate_builder()
