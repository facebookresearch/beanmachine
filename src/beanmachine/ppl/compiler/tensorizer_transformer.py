# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# Tensorizing and detensorizing
#
# See the comment at the top of devectorizer_transformer.py for a high-level description of
# what this class is for and how it works with the devectorizer.

import typing
from enum import Enum
from typing import Callable, List

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.copy_and_replace import (
    Cloner,
    NodeTransformer,
    TransformAssessment,
)
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_unsupported import UnsupportedNodeFixer
from beanmachine.ppl.compiler.size_assessment import SizeAssessment
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer, Unsized


# The tensorizing transformation does not need to know the *semantic* type of a node;
# that is, whether it is a bool, natural, probability, positive real, and so on. But
# we do need information about what the *tensor shape* was in the original PyTorch
# model.
class ElementType(Enum):
    # The node represents a multidimensional tensor that cannot be expressed in BMG.
    TENSOR = 1
    # The node represents a single value.
    SCALAR = 2
    # The node represents multiple values that can be expressed in a BMG 2-d matrix.
    MATRIX = 3
    # We were unable to deduce the size in the original Python model.
    UNKNOWN = 4


def _always(node):
    return True


class Tensorizer(NodeTransformer):
    # A node transformer exposes two operations to its caller:
    # * assess_node takes a node and returns an assessement of whether it can be
    #   transformed.
    # * transform_node takes a node and either returns a copy, or a new node to
    #   replace the given node.
    #
    # This transformer determines whether a node in the graph accumulated from the
    # original Python model should be transformed into a matrix-aware BMG node.
    def __init__(self, cloner: Cloner, sizer: Sizer):
        self.cloner = cloner
        self.sizer = sizer
        self.size_assessor = SizeAssessment(self.sizer)
        self.transform_cache = {}

        self.can_be_transformed_map = {
            bn.AdditionNode: _always,
            bn.MultiplicationNode: self.mult_can_be_tensorized,
            bn.DivisionNode: self.div_can_be_tensorized,
            bn.ComplementNode: _always,
            bn.ExpNode: _always,
            bn.LogNode: _always,
            bn.Log1mexpNode: _always,
            bn.NegateNode: self.negate_can_be_tensorized,
            bn.SumNode: _always,
        }
        self.transform_map = {
            bn.AdditionNode: lambda node, inputs: self._tensorize_addition(
                node, inputs, self.cloner.bmg.add_matrix_addition
            ),
            bn.MultiplicationNode: self._tensorize_multiply,
            bn.DivisionNode: self._tensorize_div,
            bn.ComplementNode: lambda node, inputs: self._tensorize_unary_elementwise(
                node, inputs, self.cloner.bmg.add_matrix_complement
            ),
            bn.ExpNode: lambda node, inputs: self._tensorize_unary_elementwise(
                node, inputs, self.cloner.bmg.add_matrix_exp
            ),
            bn.LogNode: lambda node, inputs: self._tensorize_unary_elementwise(
                node, inputs, self.cloner.bmg.add_matrix_log
            ),
            bn.Log1mexpNode: lambda node, inputs: self._tensorize_unary_elementwise(
                node, inputs, self.cloner.bmg.add_matrix_log1mexp
            ),
            bn.NegateNode: lambda node, inputs: self._tensorize_unary_elementwise(
                node, inputs, self.cloner.bmg.add_matrix_negate
            ),
            bn.SumNode: self._tensorize_sum,
        }

    def _tensorize_div(
        self, node: bn.DivisionNode, new_inputs: List[bn.BMGNode]
    ) -> bn.BMGNode:
        # If we have DIV(matrix, scalar) then we transform that into
        # MATRIX_SCALE(matrix, DIV(1, scalar)).
        assert len(node.inputs.inputs) == 2
        tensor_input = new_inputs[0]
        scalar_input = new_inputs[1]
        if self._element_type(tensor_input) is not ElementType.MATRIX:
            raise ValueError("Expected a matrix as first operand")
        if self._element_type(scalar_input) is not ElementType.SCALAR:
            raise ValueError("Expected a scalar as second operand")
        one = self.cloner.bmg.add_pos_real(1.0)
        new_scalar = self.cloner.bmg.add_division(one, scalar_input)
        return self.cloner.bmg.add_matrix_scale(new_scalar, tensor_input)

    def _tensorize_sum(
        self, node: bn.SumNode, new_inputs: List[bn.BMGNode]
    ) -> bn.BMGNode:
        # TODO: Ensure that we correctly insert any necessary broadcasting nodes
        # in the requirements-fixing pass.
        assert len(new_inputs) >= 1
        if any(
            self._element_type(operand) == ElementType.MATRIX
            for operand in node.inputs.inputs
        ):
            current = new_inputs[0]
            for i in range(1, len(new_inputs)):
                current = self.cloner.bmg.add_matrix_addition(current, new_inputs[i])
            return self.cloner.bmg.add_matrix_sum(current)
        return self.cloner.bmg.add_sum(*new_inputs)

    def _tensorize_multiply(
        self, node: bn.MultiplicationNode, new_inputs: List[bn.BMGNode]
    ) -> bn.BMGNode:
        # Note that this function handles *elementwise* multiplication of tensors, not
        # matrix multiplication.  There are three cases to consider.
        if len(new_inputs) != 2:
            raise ValueError(
                "Cannot transform a mult into a tensor mult because there are not two operands"
            )
        lhs_sz = self.sizer[new_inputs[0]]
        rhs_sz = self.sizer[new_inputs[1]]
        if lhs_sz == Unsized or rhs_sz == Unsized:
            raise ValueError(
                f"cannot multiply an unsized quantity. Operands: {new_inputs[0]} and {new_inputs[1]}"
            )
        # Case one: MULT(matrix, matrix) --> ELEMENTWISEMULT(matrix, matrix)
        # TODO: Ensure that the requirements fixing pass correctly inserts broadcast operators.
        lhs_is_scalar = is_scalar(lhs_sz)
        rhs_is_scalar = is_scalar(rhs_sz)
        if not lhs_is_scalar and not rhs_is_scalar:
            return self.cloner.bmg.add_elementwise_multiplication(
                new_inputs[0], new_inputs[1]
            )
        # Case two: MULT(scalar, scalar) stays just that.
        if lhs_is_scalar and not rhs_is_scalar:
            scalar_parent_image = new_inputs[0]
            tensor_parent_image = new_inputs[1]
            assert not is_scalar(rhs_sz)
        elif rhs_is_scalar and not lhs_is_scalar:
            tensor_parent_image = new_inputs[0]
            scalar_parent_image = new_inputs[1]
            assert is_scalar(rhs_sz)
        else:
            return self.cloner.bmg.add_multiplication(new_inputs[0], new_inputs[1])
        # Case three: MULT(matrix, scalar) or MULT(scalar, matrix) --> MATRIX_SCALE(matrix, scalar)
        return self.cloner.bmg.add_matrix_scale(
            scalar_parent_image, tensor_parent_image
        )

    def _tensorize_unary_elementwise(
        self,
        node: bn.UnaryOperatorNode,
        new_inputs: List[bn.BMGNode],
        creator: Callable,
    ) -> bn.BMGNode:
        # Unary operators such as exp, log, and so on, are straightforward. If the operand is
        # a matrix, generate the matrix-aware node. Otherwise leave it alone.
        assert len(new_inputs) == 1
        if self._element_type(new_inputs[0]) == ElementType.MATRIX:
            return creator(new_inputs[0])
        else:
            return self.cloner.clone(node, new_inputs)

    def _tensorize_addition(
        self,
        node: bn.AdditionNode,
        new_inputs: List[bn.BMGNode],
        creator: Callable,
    ) -> bn.BMGNode:
        # If we have matrix + matrix, scalar + matrix or matrix + scalar, generate
        # a matrix add.  In the latter cases, the requirements fixing pass will insert a
        # matrix fill node to convert the scalar to a matrix of the appropriate size.
        assert len(new_inputs) == 2
        if (
            self._element_type(new_inputs[0]) == ElementType.MATRIX
            or self._element_type(new_inputs[1]) == ElementType.MATRIX
        ):
            return creator(new_inputs[0], new_inputs[1])
        else:
            return self.cloner.clone(node, new_inputs)

    def _element_type(self, node: bn.BMGNode) -> ElementType:
        size = self.sizer[node]
        if size == Unsized:
            return ElementType.UNKNOWN
        length = len(size)
        if length == 0 or is_scalar(size):
            return ElementType.SCALAR
        if length == 1 and size[0] > 1:
            return ElementType.MATRIX
        elif length == 2:
            return ElementType.MATRIX
        else:
            return ElementType.TENSOR

    def div_can_be_tensorized(self, node: bn.DivisionNode) -> bool:
        if len(node.inputs.inputs) == 2:
            return (
                self._element_type(node.inputs.inputs[0]) == ElementType.MATRIX
                and self._element_type(node.inputs.inputs[1]) == ElementType.SCALAR
            )
        return False

    def negate_can_be_tensorized(self, node: bn.NegateNode) -> bool:
        # We want to fix unsupported nodes first before we tensorize them.
        # For example, when we have log1p(-Sample(Beta(...))) this gets changed to log(1-Sample(Beta(...))),
        # which is log(complement(Sample(Beta(...)))). But if we run the tensorizer, it does negate first
        # and then the log1p fixing. In this case, we get log(1+MatrixNegate(Sample(Beta(...)))).
        # There is no way to indicate that the computation is always positive real.
        # Therefore, this leads to the compiler thinking the requirements are violated.
        # We can avoid this by converting unsupported nodes to supported nodes and then tensorizing them.
        # TODO: This will also allow us to carry out fixpoint between tensorizer and other fixers.
        # TODO: We may want to do the same for other operators that can be tensorized.
        if any(
            isinstance(i, node_type)
            for node_type in UnsupportedNodeFixer._unsupported_nodes
            for i in node.outputs.items
        ):
            return False
        return True

    def mult_can_be_tensorized(self, node: bn.MultiplicationNode) -> bool:
        return len(node.inputs.inputs) == 2

    # a node can be tensorized if all its parents satisfy the type requirements
    def can_be_tensorized(self, original_node: bn.BMGNode) -> bool:
        if self.can_be_transformed_map.__contains__(type(original_node)):
            return self.can_be_transformed_map[type(original_node)](original_node)
        else:
            return False

    def assess_node(
        self, node: bn.BMGNode, original: BMGraphBuilder
    ) -> TransformAssessment:
        report = ErrorReport()
        error = self.size_assessor.size_error(node, self.cloner.bmg_original)
        if error is not None:
            report.add_error(error)
        return TransformAssessment(self.can_be_tensorized(node), report)

    # a node is either replaced 1-1, 1-many, or deleted
    def transform_node(
        self, node: bn.BMGNode, new_inputs: List[bn.BMGNode]
    ) -> typing.Optional[typing.Union[bn.BMGNode, List[bn.BMGNode]]]:
        if self.transform_map.__contains__(type(node)):
            return self.transform_map[type(node)](node, new_inputs)
        else:
            return self.cloner.clone(node, new_inputs)
