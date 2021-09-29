import dataclasses
from typing import Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier


@dataclasses.dataclass
class Variable:
    transformed_value: torch.Tensor
    transform: dist.Transform
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)
    children: Set[RVIdentifier] = dataclasses.field(default_factory=set)

    def copy(self):
        return dataclasses.replace(
            self, parents=self.parents.copy(), children=self.children.copy()
        )
