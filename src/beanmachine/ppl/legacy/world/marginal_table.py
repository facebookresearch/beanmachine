# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Set

import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier


@dataclasses.dataclass
class Entry:
    """
    An entry in the fom of:
    - RVIdentifier
    - Maximum support of discrete variable
    - Set of parents (including itself)
    - The probabilities associated with the cartesian product of the enumerated
      parents in a PyTorch Tensor with each entry's indices corresponding to the
      value of its parents
    - The above tensor flattened
    """

    var: RVIdentifier
    cardinality: int
    parents: Set
    values: torch.Tensor

    def __str__(self) -> str:
        out = "Variable: " + str(self.var) + ", "
        out += "Cardinality: " + str(self.cardinality) + ", "
        out += "parents: " + str(self.parents) + ", "
        out += "values: " + str(self.values)
        return out

    def __eq__(self, other) -> bool:
        return (  # pyre-ignore [7]
            self.var == other.var
            and self.cardinality == other.cardinality
            and self.parents == other.parents
            and (self.values == other.values).all().item()
        )

    def __hash__(self) -> int:
        return hash((self.var, self.cardinality))


@dataclasses.dataclass
class Table:
    """
    A table representing the marginal distribution of a (discrete) factor graph
    """

    entries: Set[Entry] = dataclasses.field(default_factory=set)

    def add_entry(self, entry) -> None:
        self.entries.add(entry)

    def __eq__(self, other) -> bool:
        return self.entries == other.entries
