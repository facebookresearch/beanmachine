# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_node_types import is_supported_by_bmg
from beanmachine.ppl.compiler.bmg_types import PositiveReal
from beanmachine.ppl.compiler.error_report import BMGError, UnsupportedNode
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.graph_labels import get_edge_label
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class UnsupportedNodeFixer(ProblemFixerBase):
    """This class takes a Bean Machine Graph builder and attempts to
    fix all uses of unsupported operators by replacing them with semantically
    equivalent nodes that are supported by BMG."""

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

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

    def _replace_index(self, node: bn.IndexNode) -> Optional[bn.BMGNode]:
        # * If we have an index into a one-column matrix, replace it with
        #   a vector index.
        # * If we have an index into a multi-column matrix, replace it with
        #   a column index
        # TODO: Consider if there are more optimizations we can make here
        # if either operand is a constant.
        typer = self._typer
        assert isinstance(typer, LatticeTyper)
        right = node.right
        left = node.left
        node_type = typer[left]
        if isinstance(node_type, bt.BMGMatrixType):
            if node_type.columns == 1:
                if (
                    isinstance(left, bn.ToMatrixNode)
                    and isinstance(right, bn.ConstantNode)
                    and typer.is_natural(right)
                ):
                    return left.inputs[right.value + 2]
                return self._bmg.add_vector_index(left, right)
            return self._bmg.add_column_index(left, right)
        return None

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

    def _replace_tensor(self, node: bn.TensorNode) -> Optional[bn.BMGNode]:
        # Replace a 1-d or 2-d tensor with a TO_MATRIX node.
        size = node.size
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

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # TODO:
        # Not -> Complement
        # Index/Map -> IfThenElse
        if isinstance(n, bn.Chi2Node):
            return self._replace_chi2(n)
        if isinstance(n, bn.DivisionNode):
            return self._replace_division(n)
        if isinstance(n, bn.IndexNode):
            return self._replace_index(n)
        if isinstance(n, bn.LogSumExpTorchNode):
            return self._replace_lse(n)
        if isinstance(n, bn.TensorNode):
            return self._replace_tensor(n)
        if isinstance(n, bn.UniformNode):
            return self._replace_uniform(n)
        return None

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        # Untyped constant nodes will be replaced in the requirements checking pass.
        return not isinstance(n, bn.UntypedConstantNode) and not is_supported_by_bmg(n)

    def _get_error(self, n: bn.BMGNode, index: int) -> Optional[BMGError]:
        # TODO: The edge labels used to visualize the graph in DOT
        # are not necessarily the best ones for displaying errors.
        # Consider fixing this.
        return UnsupportedNode(n.inputs[index], n, get_edge_label(n, index))
