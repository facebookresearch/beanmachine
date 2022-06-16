# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_node_types import is_supported_by_bmg
from beanmachine.ppl.compiler.bmg_types import PositiveReal
from beanmachine.ppl.compiler.error_report import (
    BadMatrixMultiplication,
    BMGError,
    UnsupportedNode,
    UntypableNode,
)
from beanmachine.ppl.compiler.fix_problem import (
    edge_error_pass,
    GraphFixer,
    node_error_pass,
    node_fixer_first_match,
    NodeFixer,
    type_guard,
)
from beanmachine.ppl.compiler.graph_labels import get_edge_label
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class UnsupportedNodeFixer:
    """This class takes a Bean Machine Graph builder and attempts to
    fix all uses of unsupported operators by replacing them with semantically
    equivalent nodes that are supported by BMG."""

    _bmg: BMGraphBuilder
    _typer: LatticeTyper

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        self._bmg = bmg
        self._typer = typer

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

    def _replace_exp2(self, node: bn.Exp2Node) -> bn.BMGNode:
        two = self._bmg.add_constant(2.0)
        return self._bmg.add_power(two, node.operand)

    def _replace_log10(self, node: bn.Log10Node) -> bn.BMGNode:
        # log10(x) = log(x) * (1/log(10))
        c = self._bmg.add_constant(1.0 / math.log(10))
        ln = self._bmg.add_log(node.operand)
        return self._bmg.add_multiplication(ln, c)

    def _replace_log1p(self, node: bn.Log1pNode) -> bn.BMGNode:
        # log1p(x) = log(1+x)
        # TODO: If we added a log1p node to BMG then it could get
        # more accurate results.
        one = self._bmg.add_constant(1.0)
        add = self._bmg.add_addition(one, node.operand)
        return self._bmg.add_log(add)

    def _replace_log2(self, node: bn.Log10Node) -> bn.BMGNode:
        # log2(x) = log(x) * (1/log(2))
        c = self._bmg.add_constant(1.0 / math.log(2))
        ln = self._bmg.add_log(node.operand)
        return self._bmg.add_multiplication(ln, c)

    def _replace_squareroot(self, node: bn.SquareRootNode) -> bn.BMGNode:
        half = self._bmg.add_constant(0.5)
        return self._bmg.add_power(node.operand, half)

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

    def _replace_index_one_column(self, node: bn.IndexNode) -> bn.BMGNode:
        left = node.left
        right = node.right
        typer = self._typer
        assert isinstance(typer, LatticeTyper)

        # It is possible during rewrites to end up with a constant for both
        # operands of the index; in that case, fold the index away entirely.

        # If we have a ToMatrix indexed by a constant, we can similarly
        # eliminate the indexing operation.

        # If we have Index(ColumnIndex(ToMatrix(elements), Const1), Const2)
        # then we can again eliminate the indexing altogether.

        if isinstance(right, bn.ConstantNode) and typer.is_natural(right):
            r = int(right.value)
            if isinstance(left, bn.ConstantNode):
                return self._bmg.add_constant(left.value[r])
            if isinstance(left, bn.ToMatrixNode):
                return left.inputs[r + 2]
            if isinstance(left, bn.ColumnIndexNode):
                collection = left.left
                if isinstance(collection, bn.ToMatrixNode):
                    column_index = left.right
                    if isinstance(column_index, bn.ConstantNode) and typer.is_natural(
                        column_index
                    ):
                        c = int(column_index.value)
                        rows = int(collection.rows.value)
                        return collection.inputs[rows * c + r + 2]

        # We cannot optimize it away; add a vector index operation.
        return self._bmg.add_vector_index(left, right)

    def _replace_index_multi_column(self, node: bn.IndexNode) -> bn.BMGNode:
        left = node.left
        right = node.right
        typer = self._typer
        assert isinstance(typer, LatticeTyper)

        # It is possible during rewrites to end up with a constant for both
        # operands of the index; in that case, fold the index away entirely.

        if isinstance(right, bn.ConstantNode) and typer.is_natural(right):
            r = int(right.value)
            if isinstance(left, bn.ConstantNode):
                return self._bmg.add_constant(left.value[r])
            # TODO: If left is a ToMatrixNode then we can construct a second
            # ToMatrixNode that has just the entries we need.

        # We cannot optimize it away.
        return self._bmg.add_column_index(left, right)

    def _replace_index(self, node: bn.IndexNode) -> Optional[bn.BMGNode]:
        # * If we have an index into a one-column matrix, replace it with
        #   a vector index.
        # * If we have an index into a multi-column matrix, replace it with
        #   a column index
        left = node.left
        node_type = self._typer[left]
        if not isinstance(node_type, bt.BMGMatrixType):
            return None
        if node_type.columns == 1:
            return self._replace_index_one_column(node)
        return self._replace_index_multi_column(node)

    def _replace_item(self, node: bn.ItemNode) -> Optional[bn.BMGNode]:
        # "item()" is an identity for our purposes. We just remove it.
        return node.inputs[0]

    def _replace_lse(self, node: bn.LogSumExpTorchNode) -> Optional[bn.BMGNode]:
        # We only support compiling models where dim=0 and keepDims=False.
        if not bn.is_zero(node.inputs[1]) or not bn.is_zero(node.inputs[2]):
            return None

        # We require that the input to LSE be a single column.
        operand = node.inputs[0]
        operand_type = self._typer[operand]
        if not isinstance(operand_type, bt.BMGMatrixType) or operand_type.columns != 1:
            return None

        # If the input is a TO_MATRIX operation then we can just feed its inputs
        # directly into the n-ary LSE BMG node.
        if isinstance(operand, bn.ToMatrixNode):
            # The first two inputs are the size.
            assert len(operand.inputs) >= 3
            # If the matrix is a singleton then logsumexp is an identity.
            if len(operand.inputs) == 3:
                return operand.inputs[2]
            elements = operand.inputs.inputs[2:]
            assert isinstance(elements, list)
            return self._bmg.add_logsumexp(*elements)

        # Otherwise, just generate the vector LSE BMG node.
        return self._bmg.add_logsumexp_vector(operand)

    def _replace_lae(self, node: bn.LogAddExpNode) -> Optional[bn.BMGNode]:
        return self._bmg.add_logsumexp(*node.inputs)

    def _replace_tensor(self, node: bn.TensorNode) -> Optional[bn.BMGNode]:
        # Replace a 1-d or 2-d tensor with a TO_MATRIX node.
        size = node._size
        if len(size) > 2:
            return None
        # This is the row and column count of the torch tensor.
        # In BMG, matrices are column-major, so we'll swap them.
        r, c = bt._size_to_rc(size)
        # ToMatrixNode requires naturals.
        rows = self._bmg.add_natural(c)
        cols = self._bmg.add_natural(r)
        tm = self._bmg.add_to_matrix(rows, cols, *node.inputs.inputs)
        return tm

    def _replace_bool_switch(self, node: bn.SwitchNode) -> bn.BMGNode:
        # If the switched value is a Boolean then we can turn this into an if-then-else.
        assert (len(node.inputs) - 1) / 2 == 2
        assert isinstance(node.inputs[1], bn.ConstantNode)
        assert isinstance(node.inputs[3], bn.ConstantNode)
        if bn.is_zero(node.inputs[1]):
            assert bn.is_one(node.inputs[3])
            return self._bmg.add_if_then_else(
                node.inputs[0], node.inputs[4], node.inputs[2]
            )
        else:
            assert bn.is_one(node.inputs[1])
            assert bn.is_zero(node.inputs[3])
            return self._bmg.add_if_then_else(
                node.inputs[0], node.inputs[2], node.inputs[4]
            )

    def _replace_natural_switch(self, node: bn.SwitchNode) -> Optional[bn.BMGNode]:
        # If:
        #
        # * the switched value is natural, and
        # * the cases are 0, 1, 2, ... n

        # then we can generate Choice(choice, [stochastic_values]).
        #
        # TODO: If we have a contiguous set of cases, say {2, 3, 4}, then
        # we could generate the choice from the elements and the index could
        # be the choice node minus two.
        #
        # TODO: If we have a slightly noncontiguous set of cases, say {0, 1, 3},
        # then we can generate a choice with a dummy value of the appropriate type
        # in the missing place.
        #
        # TODO: If we have arbitrary natural cases, say 1, 10, 101, then we could
        # add an integer equality operation to BMG and generate a nested IfThenElse.

        # Do we have contiguous cases 0, ..., n?
        num_cases = (len(node.inputs) - 1) // 2

        cases = set()
        for i in range(num_cases):
            c = node.inputs[i * 2 + 1]
            assert isinstance(c, bn.ConstantNode)
            cases.add(int(c.value))

        if min(cases) != 0 or max(cases) != num_cases - 1 or len(cases) != num_cases:
            return None

        # We're all set; generate a choice.
        values = [None] * num_cases
        for i in range(num_cases):
            c = node.inputs[i * 2 + 1]
            assert isinstance(c, bn.ConstantNode)
            v = node.inputs[i * 2 + 2]
            values[int(c.value)] = v  # pyre-ignore
        assert None not in values
        return self._bmg.add_choice(node.inputs[0], *values)

    def _replace_switch(self, node: bn.SwitchNode) -> Optional[bn.BMGNode]:
        # inputs[0] is the value used to perform the switch; there are
        # then pairs of constants and values.  It should be impossible
        # to have an even number of inputs.
        assert len(node.inputs) % 2 == 1

        choice = node.inputs[0]

        num_cases = (len(node.inputs) - 1) // 2
        # It should be impossible to have a switch with no cases.
        assert num_cases > 0

        # It is possible but weird to have a switch with exactly one case.
        # In this scenario we can eliminate the switch entirely by simply
        # replacing it with its lone case value.

        # TODO: Consider producing a warning for this situation, because
        # the user's model is probably wrong if they think they are stochastically
        # choosing a random variable but always get the same one.

        if num_cases == 1:
            assert isinstance(node.inputs[1], bn.ConstantNode)
            return node.inputs[2]

        # There are at least two cases.  We should never have two cases to choose from
        # but a constant choice!

        assert not isinstance(choice, bn.ConstantNode)

        tc = self._typer[choice]
        if tc == bt.Boolean:
            return self._replace_bool_switch(node)
        if tc == bt.Natural:
            return self._replace_natural_switch(node)

        # TODO: Generate a better error message for switches that we cannot yet
        # turn into BMG nodes.
        return None


def unsupported_node_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    usnf = UnsupportedNodeFixer(bmg, typer)
    return node_fixer_first_match(
        [
            type_guard(bn.Chi2Node, usnf._replace_chi2),
            type_guard(bn.DivisionNode, usnf._replace_division),
            type_guard(bn.Exp2Node, usnf._replace_exp2),
            type_guard(bn.IndexNode, usnf._replace_index),
            type_guard(bn.ItemNode, usnf._replace_item),
            type_guard(bn.Log10Node, usnf._replace_log10),
            type_guard(bn.Log1pNode, usnf._replace_log1p),
            type_guard(bn.Log2Node, usnf._replace_log2),
            type_guard(bn.LogSumExpTorchNode, usnf._replace_lse),
            type_guard(bn.LogAddExpNode, usnf._replace_lae),
            type_guard(bn.SquareRootNode, usnf._replace_squareroot),
            type_guard(bn.SwitchNode, usnf._replace_switch),
            type_guard(bn.TensorNode, usnf._replace_tensor),
            type_guard(bn.UniformNode, usnf._replace_uniform),
        ]
    )


# TODO: We should make a rewriter that detects stochastic index
# into list.  We will need to detect if the list is (1) all
# numbers, in which case we can make a constant matrix out of it,
# (2) mix of numbers and stochastic elements, in which case we can
# make it into a TO_MATRIX node, or (3) wrong shape or contents,
# in which case we must give an error.  We will likely want to
# move this check for unsupported constant value to AFTER that rewrite.


def unsupported_node_reporter(bmg: BMGraphBuilder) -> GraphFixer:
    def _error_for_unsupported_node(n: bn.BMGNode, index: int) -> Optional[BMGError]:
        # TODO: The edge labels used to visualize the graph in DOT
        # are not necessarily the best ones for displaying errors.
        # Consider fixing this.

        # Constants that can be converted to constant nodes of the appropriate type
        # will be converted in the requirements checking pass. Here we just detect
        # constants that cannot possibly be supported because they are the wrong
        # dimensionality.

        parent = n.inputs[index]

        if isinstance(parent, bn.ConstantNode):
            t = bt.type_of_value(parent.value)
            if t != bt.Tensor and t != bt.Untypable:
                return None

        if is_supported_by_bmg(parent):
            return None

        return UnsupportedNode(
            parent,
            n,
            get_edge_label(n, index),
            bmg.execution_context.node_locations(parent),
        )

    return edge_error_pass(bmg, _error_for_unsupported_node)


def bad_matmul_reporter(bmg: BMGraphBuilder) -> GraphFixer:
    typer = LatticeTyper()

    def get_error(node: bn.BMGNode) -> Optional[BMGError]:
        if not isinstance(node, bn.MatrixMultiplicationNode):
            return None

        # If an operand is bad then we'll produce an error later. This
        # pass just looks for row/column mismatches.
        lt = typer[node.left]
        if not isinstance(lt, bt.BMGMatrixType):
            return None
        rt = typer[node.right]
        if not isinstance(rt, bt.BMGMatrixType):
            return None
        if lt.columns == rt.rows:
            return None
        return BadMatrixMultiplication(
            node, lt, rt, bmg.execution_context.node_locations(node)
        )

    return node_error_pass(bmg, get_error)


def untypable_node_reporter(bmg: BMGraphBuilder) -> GraphFixer:
    # By the time this pass runs every node in the graph should
    # be a valid BMG node (except constants, which are rewritten later)
    # and therefore should have a valid type. Later passes rely on this
    # invariant, so if it is violated, let's find out now and stop
    # the compiler rather than producing some strange error later on.
    typer = LatticeTyper()

    def get_error(node: bn.BMGNode) -> Optional[BMGError]:
        # If a node is untypable then all its descendants will be too.
        # We do not want to produce a cascading error here so let's just
        # report the nodes that are untypable who have no untypable parents.
        if isinstance(node, bn.ConstantNode):
            return None
        t = typer[node]
        if t != bt.Untypable and t != bt.Tensor:
            return None
        for i in node.inputs:
            t = typer[i]
            if t == bt.Untypable or t == bt.Tensor:
                return None
        return UntypableNode(node, bmg.execution_context.node_locations(node))

    return node_error_pass(bmg, get_error)
