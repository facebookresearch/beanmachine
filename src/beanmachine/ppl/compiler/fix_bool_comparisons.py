# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import (
    BMGNode,
    ComparisonNode,
    EqualNode,
    GreaterThanEqualNode,
    LessThanEqualNode,
    NotEqualNode,
)
from beanmachine.ppl.compiler.bmg_types import Boolean, One, Zero, supremum


class BoolComparisonFixer:
    """This class takes a Bean Machine Graph builder and replaces all comparison
    operators whose operands are bool with semantically equivalent IF nodes."""

    bmg: BMGraphBuilder

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.bmg = bmg

    def _is_bool_comparison(self, node: BMGNode) -> bool:
        return (
            isinstance(node, ComparisonNode)
            and supremum(node.left.inf_type, node.right.inf_type, Boolean) == Boolean
        )

    def _replace_bool_equals(self, node: EqualNode) -> BMGNode:
        # 1 == y        -->  y
        # x == 1        -->  x
        # 0 == y        -->  not y
        # x == 0        -->  not x
        # x == y        -->  if x then y else not y
        if node.left.inf_type == One:
            return node.right
        if node.right.inf_type == One:
            return node.left
        if node.left.inf_type == Zero:
            return self.bmg.add_complement(node.right)
        if node.right.inf_type == Zero:
            return self.bmg.add_complement(node.left)
        alt = self.bmg.add_complement(node.right)
        return self.bmg.add_if_then_else(node.left, node.right, alt)

    def _replace_bool_not_equals(self, node: NotEqualNode) -> BMGNode:
        # 1 != y        -->  not y
        # x != 1        -->  not x
        # 0 != y        -->  y
        # x != 0        -->  x
        # x != y        -->  if x then not y else y
        if node.left.inf_type == One:
            return self.bmg.add_complement(node.right)
        if node.right.inf_type == One:
            return self.bmg.add_complement(node.left)
        if node.left.inf_type == Zero:
            return node.right
        if node.right.inf_type == Zero:
            return node.left
        cons = self.bmg.add_complement(node.right)
        return self.bmg.add_if_then_else(node.left, cons, node.right)

    def _replace_bool_gte(self, node: GreaterThanEqualNode) -> BMGNode:
        # 1 >= y        -->  true
        # x >= 1        -->  x
        # 0 >= y        -->  not y
        # x >= 0        -->  true
        # x >= y        -->  if x then true else not y
        if node.left.inf_type == One:
            return self.bmg.add_constant_of_type(True, Boolean)
        if node.right.inf_type == One:
            return node.left
        if node.left.inf_type == Zero:
            return self.bmg.add_complement(node.right)
        if node.right.inf_type == Zero:
            return self.bmg.add_constant_of_type(True, Boolean)
        cons = self.bmg.add_constant_of_type(True, Boolean)
        alt = self.bmg.add_complement(node.right)
        return self.bmg.add_if_then_else(node.left, cons, alt)

    def _replace_bool_lte(self, node: LessThanEqualNode) -> BMGNode:
        # 1 <= y        -->  y
        # x <= 1        -->  true
        # 0 <= y        -->  true
        # x <= 0        -->  not x
        # x <= y        -->  if x then y else true
        if node.left.inf_type == One:
            return node.right
        if node.right.inf_type == One:
            return self.bmg.add_constant_of_type(True, Boolean)
        if node.left.inf_type == Zero:
            return self.bmg.add_constant_of_type(True, Boolean)
        if node.right.inf_type == Zero:
            return self.bmg.add_complement(node.left)
        alt = self.bmg.add_constant_of_type(True, Boolean)
        return self.bmg.add_if_then_else(node.left, node.right, alt)

    def _replace_bool_comparison(self, node: ComparisonNode) -> Optional[BMGNode]:
        # TODO: x > y   -->  if x then not y else false
        # TODO: x < y   -->  if x then false else y
        if isinstance(node, EqualNode):
            return self._replace_bool_equals(node)
        if isinstance(node, NotEqualNode):
            return self._replace_bool_not_equals(node)
        if isinstance(node, GreaterThanEqualNode):
            return self._replace_bool_gte(node)
        if isinstance(node, LessThanEqualNode):
            return self._replace_bool_lte(node)

        return None

    def fix_bool_comparisons(self) -> None:
        replacements = {}
        nodes = self.bmg._traverse_from_roots()
        for node in nodes:
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                if not self._is_bool_comparison(c):
                    continue
                if c in replacements:
                    node.inputs[i] = replacements[c]
                    continue
                assert isinstance(c, ComparisonNode)
                replacement = self._replace_bool_comparison(c)
                if replacement is not None:
                    replacements[c] = replacement
                    node.inputs[i] = replacement
