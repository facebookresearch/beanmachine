# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass, fields
from typing import Optional, Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.utils import RVIdentifier, float_types
from torch import Tensor
from torch.distributions import Distribution


@dataclass
class Variable(object):
    """
    Represents each random variable instantiation in the World:

    Its fields are:
        distribution: the distribution from the variable is drawn from
        value: a sample value drawn from the random variable distribution
               or the observation value
        parent: set of parents
        children: set of children
        log_prob: log probability of value given the distribution

    for instance, for the following randodm variable:

    @sample
    def bar(self):
        if not self.foo():
            return dist.Bernoulli(torch.tensor(0.1))
        else:
            return dist.Bernoulli(torch.tensor(0.9))


    we have the following:

    Variable(
     distribution=Bernoulli(probs: 0.8999999761581421,
                            logits: 2.1972243785858154),
     value=tensor(0.),
     parent={RandomVariable(function=<function foo at 0x7fa9c87eabf8>, arguments=())},
     children=set(),
     log_prob=tensor(-2.3026)
    )
    """

    distribution: Distribution
    value: Tensor
    parent: Set[Optional[RVIdentifier]]
    children: Set[Optional[RVIdentifier]]
    log_prob: Tensor
    proposal_distribution: Distribution
    extended_val: Tensor

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                value is not None
                and field.name not in ("parent", "children")
                and not isinstance(value, field.type)
            ):
                raise ValueError(
                    f"Expected {field.name} to be of {field.type}"
                    f", but got {repr(value)}"
                )
            if field.name in ("parent", "children") and not isinstance(value, Set):
                raise ValueError(
                    f"Expected {field.name} to be of {field.type}"
                    f", but got {repr(value)}"
                )

    def __str__(self) -> str:
        return str(self.value.item()) + " from " + str(self.distribution)

    def initialize_value(self, obs: Optional[Tensor]) -> None:
        """
        Initialized the Variable value

        :param is_obs: the boolean representing whether the node is an
        observation or not
        """
        distribution = self.distribution
        # pyre-fixme
        sample_val = distribution.sample()
        # pyre-fixme
        support = distribution.support
        if obs is not None:
            self.value = obs
            return
        elif isinstance(support, dist.constraints._Real):
            self.value = torch.zeros(sample_val.shape, dtype=sample_val.dtype)
        elif isinstance(support, dist.constraints._Simplex):
            self.value = torch.ones(sample_val.shape, dtype=sample_val.dtype)
            self.value /= sample_val.shape[-1]
        elif isinstance(support, dist.constraints._GreaterThan):
            self.value = torch.ones(sample_val.shape, dtype=sample_val.dtype)
        else:
            self.value = sample_val

        if isinstance(distribution, dist.Beta):
            self.extended_val = torch.cat(
                (self.value.unsqueeze(-1), (1 - self.value).unsqueeze(-1)), -1
            )

        if isinstance(self.value, float_types) and self.extended_val is None:
            self.value.requires_grad_(True)
        elif isinstance(self.value, float_types):
            self.extended_val.requires_grad_(True)
            self.value = self.extended_val.transpose(-1, 0)[0].T

    def copy(self):
        """
        Makes a copy of self and returns it.

        :returns: copy of self.
        """
        return Variable(
            self.distribution,
            self.value,
            self.parent.copy(),
            self.children.copy(),
            self.log_prob,
            self.proposal_distribution,
            self.extended_val,
        )
