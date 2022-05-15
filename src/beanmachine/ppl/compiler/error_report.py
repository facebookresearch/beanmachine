# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module declares types used for error detection and reporting
when compiling Bean Machine models to Bean Machine Graph."""

from abc import ABC
from typing import List, Set

from beanmachine.ppl.compiler.bmg_nodes import (
    BMGNode,
    MatrixMultiplicationNode,
    Observation,
    SampleNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    BMGLatticeType,
    BMGMatrixType,
    Requirement,
    requirement_to_type,
)
from beanmachine.ppl.compiler.execution_context import FunctionCall
from beanmachine.ppl.compiler.graph_labels import get_node_error_label
from beanmachine.ppl.utils.a_or_an import a_or_an, A_or_An


class BMGError(ABC):
    pass


class Violation(BMGError):
    node: BMGNode
    node_type: BMGLatticeType
    requirement: Requirement
    consumer: BMGNode
    edge: str
    node_locations: Set[FunctionCall]

    def __init__(
        self,
        node: BMGNode,
        node_type: BMGLatticeType,
        requirement: Requirement,
        consumer: BMGNode,
        edge: str,
        node_locations: Set[FunctionCall],
    ) -> None:
        self.node = node
        self.node_type = node_type
        self.requirement = requirement
        self.consumer = consumer
        self.edge = edge
        self.node_locations = node_locations

    def __str__(self) -> str:
        r = self.requirement
        t = requirement_to_type(r)
        assert isinstance(t, BMGLatticeType)
        # TODO: Fix this error message for the case where we require
        # a matrix but we can only get a scalar value
        consumer_label = get_node_error_label(self.consumer)
        msg = (
            f"The {self.edge} of {a_or_an(consumer_label)} "
            + f"is required to be {a_or_an(t.long_name)} "
            + f"but is {a_or_an(self.node_type.long_name)}."
        )
        if len(self.node_locations) > 0:
            msg += f"\nThe {consumer_label} was created in function call "
            msg += ", ".join(sorted(str(loc) for loc in self.node_locations))
            msg += "."
        return msg


class ImpossibleObservation(BMGError):
    node: Observation
    distribution_type: BMGLatticeType

    def __init__(self, node: Observation, distribution_type: BMGLatticeType) -> None:
        self.node = node
        self.distribution_type = distribution_type

    def __str__(self) -> str:
        v = self.node.value
        s = self.node.observed
        assert isinstance(s, SampleNode)
        d = get_node_error_label(s.operand)
        t = self.distribution_type.long_name
        msg = (
            f"{A_or_An(d)} distribution is observed to have value {v} "
            + f"but only produces samples of type {t}."
        )
        return msg


class UnsupportedNode(BMGError):
    # Graph node "consumer" has a parent "node" which is not supported by BMG.
    # Give an error describing the offending node, the consumer which uses its
    # value, and the label of the edge connecting them.
    node: BMGNode
    consumer: BMGNode
    edge: str
    node_locations: Set[FunctionCall]

    def __init__(
        self,
        node: BMGNode,
        consumer: BMGNode,
        edge: str,
        node_locations: Set[FunctionCall],
    ) -> None:
        self.node = node
        self.consumer = consumer
        self.edge = edge
        self.node_locations = node_locations

    def __str__(self) -> str:
        # TODO: Improve wording and diagnosis.
        msg = f"The model uses {a_or_an(get_node_error_label(self.node))} "
        msg += "operation unsupported by Bean Machine Graph."

        if len(self.node_locations) > 0:
            msg += "\nThe unsupported node was created in function call "
            msg += ", ".join(sorted(str(loc) for loc in self.node_locations))
            msg += "."
        else:
            msg += f"\nThe unsupported node is the {self.edge} "
            msg += f"of {a_or_an(get_node_error_label(self.consumer))}."

        return msg


class BadMatrixMultiplication(BMGError):
    node: MatrixMultiplicationNode
    left_type: BMGMatrixType
    right_type: BMGMatrixType
    node_locations: Set[FunctionCall]

    def __init__(
        self,
        node: MatrixMultiplicationNode,
        left_type: BMGMatrixType,
        right_type: BMGMatrixType,
        node_locations: Set[FunctionCall],
    ) -> None:
        self.node = node
        self.left_type = left_type
        self.right_type = right_type
        self.node_locations = node_locations

    def __str__(self) -> str:
        # TODO: Improve wording and diagnosis.
        msg = f"The model uses {a_or_an(get_node_error_label(self.node))} "
        msg += "operation unsupported by Bean Machine Graph.\nThe dimensions of the"
        msg += f" operands are {self.left_type.rows}x{self.left_type.columns} and "
        msg += f"{self.right_type.rows}x{self.right_type.columns}."

        if len(self.node_locations) > 0:
            msg += "\nThe unsupported node was created in function call "
            msg += ", ".join(sorted(str(loc) for loc in self.node_locations))
            msg += "."

        return msg


class UntypableNode(BMGError):
    node: BMGNode
    node_locations: Set[FunctionCall]

    def __init__(
        self,
        node: BMGNode,
        node_locations: Set[FunctionCall],
    ) -> None:
        self.node = node
        self.node_locations = node_locations

    def __str__(self) -> str:
        msg = "INTERNAL COMPILER ERROR: Untypable node\n"
        msg += "(This indicates a defect in the compiler, not in the model.)\n"
        msg = f"The model uses {a_or_an(get_node_error_label(self.node))} node.\n"
        msg += "The compiler is unable to determine its type in the Bean Machine Graph"
        msg += " type system."

        if len(self.node_locations) > 0:
            msg += "\nThe untypable node was created in function call "
            msg += ", ".join(sorted(str(loc) for loc in self.node_locations))
            msg += "."

        return msg


class ErrorReport:

    errors: List[BMGError]

    def __init__(self) -> None:
        self.errors = []

    def add_error(self, error: BMGError) -> None:
        self.errors.append(error)

    def raise_errors(self) -> None:
        if len(self.errors) != 0:
            # TODO: Better error
            raise ValueError(str(self))

    def any(self) -> bool:
        return len(self.errors) != 0

    def __str__(self) -> str:
        return "\n".join(sorted(str(e) for e in self.errors))
