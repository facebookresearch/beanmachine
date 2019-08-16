# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass, fields
from typing import Set

import torch


@dataclass
class Variable(object):
    """
    represents each random variable instantiation in the World:

    parameters are:
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
     parent={(<function Model.foo at 0x7ff0bb6c0a60>, ())},
     children=set(),
     log_prob=tensor(-2.3026)
    )
    """

    distribution: torch.distributions.Distribution
    value: torch.Tensor
    parent: Set
    children: Set
    log_prob: torch.Tensor

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not isinstance(value, field.type):
                raise ValueError(
                    f"Expected {field.name} to be of {field.type}"
                    f", but got {repr(value)}"
                )
