# Copyright (c) Facebook, Inc. and its affiliates.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import AdditionNode, BMGNode, NegateNode
from beanmachine.ppl.compiler.bmg_types import One


class AdditionFixer:
    """This class takes a Bean Machine Graph builder and attempts to
    rewrite reachable additions of the form:

    * add(1, negate(prob)) or add(negate(prob), 1) -> complement(prob)
    * add(1, negate(bool)) or add(negate(bool), 1) -> complement(bool)"""

    bmg: BMGraphBuilder

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.bmg = bmg

    def _addition_to_complement(self, node: AdditionNode) -> BMGNode:
        assert node.can_be_complement
        # We have 1+(-x) or (-x)+1 where x is either P or B, and require
        # a P or B. Complement(x) is of the same type as x if x is P or B.
        if node.left.inf_type == One:
            other = node.right
        else:
            assert node.right.inf_type == One
            other = node.left
        assert isinstance(other, NegateNode)
        return self.bmg.add_complement(other.operand)

    def additions_to_complements(self) -> None:
        replacements = {}
        nodes = self.bmg._traverse_from_roots()
        for node in nodes:
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                if not isinstance(c, AdditionNode):
                    continue
                assert isinstance(c, AdditionNode)
                if not c.can_be_complement:
                    continue
                if c in replacements:
                    node.inputs[i] = replacements[c]
                    continue
                replacement = self._addition_to_complement(c)
                node.inputs[i] = replacement
                replacements[c] = replacement
