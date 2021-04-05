# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import Boolean, One, Zero, supremum
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase


class BoolComparisonFixer(ProblemFixerBase):
    """This class takes a Bean Machine Graph builder and replaces all comparison
    operators whose operands are bool with semantically equivalent IF nodes."""

    def __init__(self, bmg: BMGraphBuilder) -> None:
        ProblemFixerBase.__init__(self, bmg)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return (
            isinstance(n, bn.ComparisonNode)
            and supremum(n.left.inf_type, n.right.inf_type, Boolean) == Boolean
        )

    def _replace_bool_equals(self, node: bn.EqualNode) -> bn.BMGNode:
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
            return self._bmg.add_complement(node.right)
        if node.right.inf_type == Zero:
            return self._bmg.add_complement(node.left)
        alt = self._bmg.add_complement(node.right)
        return self._bmg.add_if_then_else(node.left, node.right, alt)

    def _replace_bool_not_equals(self, node: bn.NotEqualNode) -> bn.BMGNode:
        # 1 != y        -->  not y
        # x != 1        -->  not x
        # 0 != y        -->  y
        # x != 0        -->  x
        # x != y        -->  if x then not y else y
        if node.left.inf_type == One:
            return self._bmg.add_complement(node.right)
        if node.right.inf_type == One:
            return self._bmg.add_complement(node.left)
        if node.left.inf_type == Zero:
            return node.right
        if node.right.inf_type == Zero:
            return node.left
        cons = self._bmg.add_complement(node.right)
        return self._bmg.add_if_then_else(node.left, cons, node.right)

    def _replace_bool_gte(self, node: bn.GreaterThanEqualNode) -> bn.BMGNode:
        # 1 >= y        -->  true
        # x >= 1        -->  x
        # 0 >= y        -->  not y
        # x >= 0        -->  true
        # x >= y        -->  if x then true else not y
        if node.left.inf_type == One:
            return self._bmg.add_constant_of_type(True, Boolean)
        if node.right.inf_type == One:
            return node.left
        if node.left.inf_type == Zero:
            return self._bmg.add_complement(node.right)
        if node.right.inf_type == Zero:
            return self._bmg.add_constant_of_type(True, Boolean)
        cons = self._bmg.add_constant_of_type(True, Boolean)
        alt = self._bmg.add_complement(node.right)
        return self._bmg.add_if_then_else(node.left, cons, alt)

    def _replace_bool_gt(self, node: bn.GreaterThanNode) -> bn.BMGNode:
        # 1 > y        -->  not y
        # x > 1        -->  false
        # 0 > y        -->  false
        # x > 0        -->  x
        # x > y        -->  if x then not y else false
        if node.left.inf_type == One:
            return self._bmg.add_complement(node.right)
        if node.right.inf_type == One:
            return self._bmg.add_constant_of_type(False, Boolean)
        if node.left.inf_type == Zero:
            return self._bmg.add_constant_of_type(False, Boolean)
        if node.right.inf_type == Zero:
            return node.left
        cons = self._bmg.add_complement(node.right)
        alt = self._bmg.add_constant_of_type(False, Boolean)
        return self._bmg.add_if_then_else(node.left, cons, alt)

    def _replace_bool_lte(self, node: bn.LessThanEqualNode) -> bn.BMGNode:
        # 1 <= y        -->  y
        # x <= 1        -->  true
        # 0 <= y        -->  true
        # x <= 0        -->  not x
        # x <= y        -->  if x then y else true
        if node.left.inf_type == One:
            return node.right
        if node.right.inf_type == One:
            return self._bmg.add_constant_of_type(True, Boolean)
        if node.left.inf_type == Zero:
            return self._bmg.add_constant_of_type(True, Boolean)
        if node.right.inf_type == Zero:
            return self._bmg.add_complement(node.left)
        alt = self._bmg.add_constant_of_type(True, Boolean)
        return self._bmg.add_if_then_else(node.left, node.right, alt)

    def _replace_bool_lt(self, node: bn.LessThanNode) -> bn.BMGNode:
        # 1 < y        -->  false
        # x < 1        -->  not x
        # 0 < y        -->  y
        # x < 0        -->  false
        # x < y        -->  if x then false else y
        if node.left.inf_type == One:
            return self._bmg.add_constant_of_type(False, Boolean)
        if node.right.inf_type == One:
            return self._bmg.add_complement(node.left)
        if node.left.inf_type == Zero:
            return node.right
        if node.right.inf_type == Zero:
            return self._bmg.add_constant_of_type(False, Boolean)
        cons = self._bmg.add_constant_of_type(False, Boolean)
        return self._bmg.add_if_then_else(node.left, cons, node.right)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # TODO: Should we treat "x is y" the same as "x == y" when they are
        # bools, or should that be an error?
        if isinstance(n, bn.EqualNode):
            return self._replace_bool_equals(n)
        if isinstance(n, bn.NotEqualNode):
            return self._replace_bool_not_equals(n)
        if isinstance(n, bn.GreaterThanEqualNode):
            return self._replace_bool_gte(n)
        if isinstance(n, bn.GreaterThanNode):
            return self._replace_bool_gt(n)
        if isinstance(n, bn.LessThanEqualNode):
            return self._replace_bool_lte(n)
        if isinstance(n, bn.LessThanNode):
            return self._replace_bool_lt(n)

        return None
