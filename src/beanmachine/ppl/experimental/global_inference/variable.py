from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier


@dataclasses.dataclass(frozen=True)
class Variable:
    value: torch.Tensor
    distribution: dist.Distribution
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)
    children: Set[RVIdentifier] = dataclasses.field(default_factory=set)

    @cached_property
    def log_prob(self) -> torch.Tensor:
        try:
            return self.distribution.log_prob(self.value)
        except (RuntimeError, ValueError):
            dtype = (
                self.value.dtype
                if torch.is_floating_point(self.value)
                else torch.float32
            )
            return torch.tensor(float("-inf"), device=self.value.device, dtype=dtype)

    def replace(self, **changes) -> Variable:
        """Return a new Variable object with fields replaced by the changes"""
        return dataclasses.replace(self, **changes)
