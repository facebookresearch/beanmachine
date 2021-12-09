# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module declares types used for error detection and reporting
when compiling Bean Machine models to Bean Machine Graph."""

from abc import ABC
from typing import List

from beanmachine.ppl.compiler.bmg_nodes import BMGNode, Observation, SampleNode
from beanmachine.ppl.compiler.bmg_types import (
    BMGLatticeType,
    Requirement,
    requirement_to_type,
)
from beanmachine.ppl.compiler.graph_labels import get_node_label


# TODO: We use the DOT label of the node in the error messages here,
# but we can probably do a better job. These labels are designed for
# displaying a graph, not an error message.


class BMGError(ABC):
    pass


class Violation(BMGError):
    node: BMGNode
    node_type: BMGLatticeType
    requirement: Requirement
    consumer: BMGNode
    edge: str

    def __init__(
        self,
        node: BMGNode,
        node_type: BMGLatticeType,
        requirement: Requirement,
        consumer: BMGNode,
        edge: str,
    ) -> None:
        self.node = node
        self.node_type = node_type
        self.requirement = requirement
        self.consumer = consumer
        self.edge = edge

    def __str__(self) -> str:
        r = self.requirement
        t = requirement_to_type(r)
        assert isinstance(t, BMGLatticeType)
        # TODO: Fix this error message for the case where we require
        # a matrix but we can only get a scalar value
        msg = (
            f"The {self.edge} of a {get_node_label(self.consumer)} "
            + f"is required to be a {t.long_name} "
            + f"but is a {self.node_type.long_name}."
        )
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
        d = get_node_label(s.operand)
        t = self.distribution_type.long_name
        msg = (
            f"A {d} distribution is observed to have value {v} "
            + f"but only produces samples of type {t}."
        )
        return msg


class UnsupportedNode(BMGError):
    node: BMGNode
    consumer: BMGNode
    edge: str

    def __init__(self, node: BMGNode, consumer: BMGNode, edge: str) -> None:
        self.node = node
        self.consumer = consumer
        self.edge = edge

    def __str__(self) -> str:
        msg = (
            # TODO: Improve wording and diagnosis.
            f"The model uses a {get_node_label(self.node)} operation unsupported by "
            + f"Bean Machine Graph.\nThe unsupported node is the {self.edge} "
            + f"of a {get_node_label(self.consumer)}."
        )
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
