# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import Boolean
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    node_fixer_first_match,
    NodeFixer,
    NodeFixerResult,
    type_guard,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class BoolComparisonFixer:
    """This class takes a Bean Machine Graph builder and replaces all comparison
    operators whose operands are bool with semantically equivalent IF nodes."""

    _bmg: BMGraphBuilder
    _typer: LatticeTyper

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        self._bmg = bmg
        self._typer = typer

    def _both_bool(self, n: bn.BMGNode) -> bool:
        return self._typer.is_bool(n.left) and self._typer.is_bool(  # pyre-ignore
            n.right  # pyre-ignore
        )

    def _replace_bool_equals(self, node: bn.EqualNode) -> NodeFixerResult:
        # 1 == y        -->  y
        # x == 1        -->  x
        # 0 == y        -->  not y
        # x == 0        -->  not x
        # x == y        -->  if x then y else not y
        if not self._both_bool(node):
            return Inapplicable
        if bn.is_one(node.left):
            return node.right
        if bn.is_one(node.right):
            return node.left
        if bn.is_zero(node.left):
            return self._bmg.add_complement(node.right)
        if bn.is_zero(node.right):
            return self._bmg.add_complement(node.left)
        alt = self._bmg.add_complement(node.right)
        return self._bmg.add_if_then_else(node.left, node.right, alt)

    def _replace_bool_not_equals(self, node: bn.NotEqualNode) -> NodeFixerResult:
        # 1 != y        -->  not y
        # x != 1        -->  not x
        # 0 != y        -->  y
        # x != 0        -->  x
        # x != y        -->  if x then not y else y
        if not self._both_bool(node):
            return Inapplicable
        if bn.is_one(node.left):
            return self._bmg.add_complement(node.right)
        if bn.is_one(node.right):
            return self._bmg.add_complement(node.left)
        if bn.is_zero(node.left):
            return node.right
        if bn.is_zero(node.right):
            return node.left
        cons = self._bmg.add_complement(node.right)
        return self._bmg.add_if_then_else(node.left, cons, node.right)

    def _replace_bool_gte(self, node: bn.GreaterThanEqualNode) -> NodeFixerResult:
        # 1 >= y        -->  true
        # x >= 1        -->  x
        # 0 >= y        -->  not y
        # x >= 0        -->  true
        # x >= y        -->  if x then true else not y
        if not self._both_bool(node):
            return Inapplicable
        if bn.is_one(node.left):
            return self._bmg.add_constant_of_type(True, Boolean)
        if bn.is_one(node.right):
            return node.left
        if bn.is_zero(node.left):
            return self._bmg.add_complement(node.right)
        if bn.is_zero(node.right):
            return self._bmg.add_constant_of_type(True, Boolean)
        cons = self._bmg.add_constant_of_type(True, Boolean)
        alt = self._bmg.add_complement(node.right)
        return self._bmg.add_if_then_else(node.left, cons, alt)

    def _replace_bool_gt(self, node: bn.GreaterThanNode) -> NodeFixerResult:
        # 1 > y        -->  not y
        # x > 1        -->  false
        # 0 > y        -->  false
        # x > 0        -->  x
        # x > y        -->  if x then not y else false
        if not self._both_bool(node):
            return Inapplicable
        if bn.is_one(node.left):
            return self._bmg.add_complement(node.right)
        if bn.is_one(node.right):
            return self._bmg.add_constant_of_type(False, Boolean)
        if bn.is_zero(node.left):
            return self._bmg.add_constant_of_type(False, Boolean)
        if bn.is_zero(node.right):
            return node.left
        cons = self._bmg.add_complement(node.right)
        alt = self._bmg.add_constant_of_type(False, Boolean)
        return self._bmg.add_if_then_else(node.left, cons, alt)

    def _replace_bool_lte(self, node: bn.LessThanEqualNode) -> NodeFixerResult:
        # 1 <= y        -->  y
        # x <= 1        -->  true
        # 0 <= y        -->  true
        # x <= 0        -->  not x
        # x <= y        -->  if x then y else true
        if not self._both_bool(node):
            return Inapplicable
        if bn.is_one(node.left):
            return node.right
        if bn.is_one(node.right):
            return self._bmg.add_constant_of_type(True, Boolean)
        if bn.is_zero(node.left):
            return self._bmg.add_constant_of_type(True, Boolean)
        if bn.is_zero(node.right):
            return self._bmg.add_complement(node.left)
        alt = self._bmg.add_constant_of_type(True, Boolean)
        return self._bmg.add_if_then_else(node.left, node.right, alt)

    def _replace_bool_lt(self, node: bn.LessThanNode) -> NodeFixerResult:
        # 1 < y        -->  false
        # x < 1        -->  not x
        # 0 < y        -->  y
        # x < 0        -->  false
        # x < y        -->  if x then false else y
        if not self._both_bool(node):
            return Inapplicable
        if bn.is_one(node.left):
            return self._bmg.add_constant_of_type(False, Boolean)
        if bn.is_one(node.right):
            return self._bmg.add_complement(node.left)
        if bn.is_zero(node.left):
            return node.right
        if bn.is_zero(node.right):
            return self._bmg.add_constant_of_type(False, Boolean)
        cons = self._bmg.add_constant_of_type(False, Boolean)
        return self._bmg.add_if_then_else(node.left, cons, node.right)


def bool_comparison_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    bcf = BoolComparisonFixer(bmg, typer)
    # TODO: Should we treat "x is y" the same as "x == y" when they are
    # bools, or should that be an error?

    return node_fixer_first_match(
        [
            type_guard(bn.EqualNode, bcf._replace_bool_equals),
            type_guard(bn.GreaterThanEqualNode, bcf._replace_bool_gte),
            type_guard(bn.GreaterThanNode, bcf._replace_bool_gt),
            type_guard(bn.LessThanEqualNode, bcf._replace_bool_lte),
            type_guard(bn.LessThanNode, bcf._replace_bool_lt),
            type_guard(bn.NotEqualNode, bcf._replace_bool_not_equals),
        ]
    )
