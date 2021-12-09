# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
from typing import Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch.distributions.utils import lazy_property


@dataclasses.dataclass
class Variable:
    """
    Primitive used for maintaining metadata of random variables. Usually used
    in conjunction with `World` during inference.

    Attributes:
      value (torch.Tensor): Sampled value of random variable
      distribution (torch.distributions.Distribution): Distribution random variable was sampled from
      parents (set): Set containing the RVIdentifiers of the parents of the random variable
      children (set): Set containing the RVIdentifiers of the children of the random variable
    """

    value: torch.Tensor
    distribution: dist.Distribution
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)
    children: Set[RVIdentifier] = dataclasses.field(default_factory=set)

    @lazy_property
    def log_prob(self) -> torch.Tensor:
        """
        Returns
             The logprob of the `value` of the value given the distribution.
        """
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
