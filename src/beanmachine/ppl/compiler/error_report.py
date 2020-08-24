# Copyright (c) Facebook, Inc. and its affiliates.

"""This module declares types used for error detection and reporting
when compiling Bean Machine models to Bean Machine Graph."""

from abc import ABC
from typing import List

from beanmachine.ppl.compiler.bmg_nodes import BMGNode
from beanmachine.ppl.compiler.bmg_types import BMGLatticeType, Requirement, UpperBound


class BMGError(ABC):
    pass


class Violation(BMGError):
    node: BMGNode
    requirement: Requirement
    consumer: BMGNode
    edge: str

    def __init__(
        self, node: BMGNode, requirement: Requirement, consumer: BMGNode, edge: str
    ) -> None:
        self.node = node
        self.requirement = requirement
        self.consumer = consumer
        self.edge = edge

    def __str__(self) -> str:
        r = self.requirement
        t = r.bound if isinstance(r, UpperBound) else r
        assert isinstance(t, BMGLatticeType)
        msg = (
            f"The {self.edge} of a {self.consumer.label} "
            + f"is required to be a {t.long_name} "
            + f"but is a {self.node.inf_type.long_name}."
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
            f"The model uses a {self.node.label} operation unsupported by "
            + f"Bean Machine Graph.\nThe unsupported node is the {self.edge} "
            + f"of a {self.consumer.label}."
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
