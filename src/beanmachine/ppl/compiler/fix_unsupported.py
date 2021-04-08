# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import PositiveReal
from beanmachine.ppl.compiler.error_report import BMGError, UnsupportedNode
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.graph_labels import get_edge_label


class UnsupportedNodeFixer(ProblemFixerBase):
    """This class takes a Bean Machine Graph builder and attempts to
    fix all uses of unsupported operators by replacing them with semantically
    equivalent nodes that are supported by BMG."""

    def __init__(self, bmg: BMGraphBuilder) -> None:
        ProblemFixerBase.__init__(self, bmg)

    def _replace_division(self, node: bn.DivisionNode) -> Optional[bn.BMGNode]:
        # BMG has no division node. We replace division by a constant with
        # a multiplication:
        #
        # x / const --> x * (1 / const)
        #
        # And we replace division by a non-constant with a power:
        #
        # x / y --> x * (y ** (-1))
        #
        r = node.right
        if isinstance(r, bn.ConstantNode):
            return self._bmg.add_multiplication(
                node.left, self._bmg.add_constant(1.0 / r.value)
            )
        neg1 = self._bmg.add_constant(-1.0)
        powr = self._bmg.add_power(r, neg1)
        return self._bmg.add_multiplication(node.left, powr)

    def _replace_uniform(self, node: bn.UniformNode) -> Optional[bn.BMGNode]:
        # TODO: Suppose we have something like Uniform(1.0, 2.0).  Can we replace that
        # with (Flat() + 1.0) ? The problem is that if there is an observation on
        # a sample of the original uniform, we need to modify the observation to
        # point to the sample, not the addition, and then we need to modify the value
        # of the observation. But this is doable. Revisit this question later.
        # For now, we can simply say that a uniform distribution over 0.0 to 1.0 can
        # be replaced with a flat.
        low = node.low
        high = node.high
        if (
            isinstance(low, bn.ConstantNode)
            and float(low.value) == 0.0
            and isinstance(high, bn.ConstantNode)
            and float(high.value) == 1.0
        ):
            return self._bmg.add_flat()
        return None

    def _replace_chi2(self, node: bn.Chi2Node) -> bn.BMGNode:
        # Chi2(x), which BMG does not support, is exactly equivalent
        # to Gamma(x * 0.5, 0.5), which BMG does support.
        half = self._bmg.add_constant_of_type(0.5, PositiveReal)
        mult = self._bmg.add_multiplication(node.df, half)
        return self._bmg.add_gamma(mult, half)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # TODO:
        # Not -> Complement
        # Index/Map -> IfThenElse
        if isinstance(n, bn.Chi2Node):
            return self._replace_chi2(n)
        if isinstance(n, bn.DivisionNode):
            return self._replace_division(n)
        if isinstance(n, bn.UniformNode):
            return self._replace_uniform(n)
        return None

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return not n._supported_in_bmg()

    def _get_error(self, n: bn.BMGNode, index: int) -> Optional[BMGError]:
        # TODO: The edge labels used to visualize the graph in DOT
        # are not necessarily the best ones for displaying errors.
        # Consider fixing this.
        return UnsupportedNode(n.inputs[index], n, get_edge_label(n, index))
