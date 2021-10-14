from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier


@dataclasses.dataclass(frozen=True)
class Variable:
    transformed_value: torch.Tensor
    transform: dist.Transform
    distribution: dist.Distribution
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)
    children: Set[RVIdentifier] = dataclasses.field(default_factory=set)

    @cached_property
    def log_prob(self) -> torch.Tensor:
        try:
            y = self.transformed_value
            x = self.transform.inv(y)
            return self.distribution.log_prob(x) - self.transform.log_abs_det_jacobian(
                x, y
            )
        except (RuntimeError, ValueError):
            return torch.tensor(float("-inf"), device=self.transformed_value.device)

    def replace(self, **changes) -> Variable:
        """Return a new Variable object with fields replaced by the changes"""
        return dataclasses.replace(self, **changes)
