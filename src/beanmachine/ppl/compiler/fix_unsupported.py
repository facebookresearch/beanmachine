# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import (
    BMGNode,
    Chi2Node,
    ComparisonNode,
    ConstantNode,
    DivisionNode,
    EqualNode,
    GreaterThanEqualNode,
    NotEqualNode,
    UniformNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    One,
    PositiveReal,
    Zero,
    supremum,
)
from beanmachine.ppl.compiler.error_report import ErrorReport, UnsupportedNode


class UnsupportedNodeFixer:
    """This class takes a Bean Machine Graph builder and attempts to
    fix all uses of unsupported operators by replacing them with semantically
    equivalent nodes that are supported by BMG."""

    errors: ErrorReport
    bmg: BMGraphBuilder

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.errors = ErrorReport()
        self.bmg = bmg

    def _replace_division(self, node: DivisionNode) -> Optional[BMGNode]:
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
        if isinstance(r, ConstantNode):
            return self.bmg.add_multiplication(
                node.left, self.bmg.add_constant(1.0 / r.value)
            )
        neg1 = self.bmg.add_constant(-1.0)
        powr = self.bmg.add_power(r, neg1)
        return self.bmg.add_multiplication(node.left, powr)

    def _replace_uniform(self, node: UniformNode) -> Optional[BMGNode]:
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
            isinstance(low, ConstantNode)
            and float(low.value) == 0.0
            and isinstance(high, ConstantNode)
            and float(high.value) == 1.0
        ):
            return self.bmg.add_flat()
        return None

    def _replace_chi2(self, node: Chi2Node) -> BMGNode:
        # Chi2(x), which BMG does not support, is exactly equivalent
        # to Gamma(x * 0.5, 0.5), which BMG does support.
        half = self.bmg.add_constant_of_type(0.5, PositiveReal)
        mult = self.bmg.add_multiplication(node.df, half)
        return self.bmg.add_gamma(mult, half)

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

    def _replace_bool_comparison(self, node: ComparisonNode) -> Optional[BMGNode]:
        # TODO: x > y   -->  if x then not y else false
        # TODO: x < y   -->  if x then false else y
        # TODO: x <= y  -->  if x then y else true
        if isinstance(node, EqualNode):
            return self._replace_bool_equals(node)
        if isinstance(node, NotEqualNode):
            return self._replace_bool_not_equals(node)
        if isinstance(node, GreaterThanEqualNode):
            return self._replace_bool_gte(node)

        return None

    def _replace_unsupported_node(self, node: BMGNode) -> Optional[BMGNode]:
        # TODO:
        # Not -> Complement
        # Index/Map -> IfThenElse
        if isinstance(node, Chi2Node):
            return self._replace_chi2(node)
        if isinstance(node, DivisionNode):
            return self._replace_division(node)
        if isinstance(node, UniformNode):
            return self._replace_uniform(node)
        if self._is_bool_comparison(node):
            assert isinstance(node, ComparisonNode)
            return self._replace_bool_comparison(node)
        return None

    def fix_unsupported_nodes(self) -> None:
        replacements = {}
        reported = set()
        nodes = self.bmg._traverse_from_roots()
        for node in nodes:
            for i in range(len(node.inputs)):
                c = node.inputs[i]
                if c._supported_in_bmg():
                    continue
                # We have an unsupported node. Have we already worked out its
                # replacement node?
                if c in replacements:
                    node.inputs[i] = replacements[c]
                    continue
                # We have an unsupported node; have we already reported it as
                # having no replacement?
                if c in reported:
                    continue
                # We have an unsupported node and we don't know what to do.
                replacement = self._replace_unsupported_node(c)
                if replacement is None:
                    self.errors.add_error(UnsupportedNode(c, node, node.edges[i]))
                    reported.add(c)
                else:
                    replacements[c] = replacement
                    node.inputs[i] = replacement
