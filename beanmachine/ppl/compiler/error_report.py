# Copyright (c) Facebook, Inc. and its affiliates.

"""This module declares types used for error detection and reporting
when compiling Bean Machine models to Bean Machine Graph."""

from abc import ABC
from typing import List

from beanmachine.ppl.compiler.bmg_nodes import BMGNode
from beanmachine.ppl.compiler.bmg_types import Requirement, UpperBound, name_of_type


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
        t = (
            self.requirement.bound
            if isinstance(self.requirement, UpperBound)
            else self.requirement
        )
        assert isinstance(t, type)
        msg = (
            f"The {self.edge} of a {self.consumer.label} "
            + f"is required to be a {name_of_type(t)} "
            + f"but is a {name_of_type(self.node.inf_type)}."
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

    def __str__(self) -> str:
        return "\n".join(sorted(str(e) for e in self.errors))
