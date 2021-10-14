from __future__ import annotations

import dataclasses
from typing import Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier


@dataclasses.dataclass(frozen=True)
class Variable:
    transformed_value: torch.Tensor
    transform: dist.Transform
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)
    children: Set[RVIdentifier] = dataclasses.field(default_factory=set)

    def replace(self, **changes) -> Variable:
        """Return a new Variable object with fields replaced by the changes"""
        return dataclasses.replace(self, **changes)
