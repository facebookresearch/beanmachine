# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.model.utils import RVIdentifier, float_types
from beanmachine.ppl.world.utils import get_transforms, is_discrete
from torch import Tensor
from torch.distributions import Distribution


@dataclass(eq=True, frozen=True)
class ProposalDistribution:
    proposal_distribution: Optional[Any]
    requires_transform: bool
    requires_reshape: bool
    arguments: Dict


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
    proposal_distribution: ProposalDistribution
    extended_val: Tensor
    is_discrete: bool
    transforms: List
    unconstrained_value: Tensor
    jacobian: Tensor

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                value is not None
                and field.name not in ("parent", "children", "proposal_distribution")
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
        return str(self.value) + " from " + str(self.distribution)

    def initialize_value(self, obs: Optional[Tensor]) -> Tensor:
        """
        Initialized the Variable value

        :param is_obs: the boolean representing whether the node is an
        observation or not
        :returns: the value to the set the Variable value to
        """
        distribution = self.distribution
        # pyre-fixme
        sample_val = distribution.sample()
        # pyre-fixme
        support = distribution.support
        if obs is not None:
            return obs
        elif isinstance(support, dist.constraints._Real):
            return torch.zeros(sample_val.shape, dtype=sample_val.dtype)
        elif isinstance(support, dist.constraints._Simplex):
            value = torch.ones(sample_val.shape, dtype=sample_val.dtype)
            return value / sample_val.shape[-1]
        elif isinstance(support, dist.constraints._GreaterThan):
            return (
                torch.ones(sample_val.shape, dtype=sample_val.dtype)
                + support.lower_bound
            )
        elif isinstance(support, dist.constraints._Boolean):
            return dist.Bernoulli(torch.ones(sample_val.shape) / 2).sample()
        elif isinstance(support, dist.constraints._Interval):
            lower_bound = torch.ones(sample_val.shape) * support.lower_bound
            upper_bound = torch.ones(sample_val.shape) * support.upper_bound
            return dist.Uniform(lower_bound, upper_bound).sample()
        elif isinstance(support, dist.constraints._IntegerInterval):
            integer_interval = support.upper_bound - support.lower_bound
            return dist.Categorical(
                (torch.ones(integer_interval)).expand(
                    sample_val.shape + (integer_interval,)
                )
            ).sample()
        elif isinstance(support, dist.constraints._IntegerGreaterThan):
            return (
                torch.ones(sample_val.shape, dtype=sample_val.dtype)
                + support.lower_bound
            )
        return sample_val

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
            self.is_discrete,
            self.transforms,
            self.unconstrained_value,
            self.jacobian,
        )

    def update_value(self, value: Tensor) -> None:
        """
        Update the value of the variable

        :param value: the value to update the variable to
        """
        if len(self.transforms) == 0:
            self.value = value
            self.jacobian = tensor(0.0)
            if isinstance(self.distribution, dist.Beta):
                self.extended_val = torch.cat(
                    (self.value.unsqueeze(-1), (1 - self.value).unsqueeze(-1)), -1
                )
            if isinstance(self.value, float_types) and self.extended_val is None:
                self.value.requires_grad_(True)
            elif isinstance(self.value, float_types):
                self.extended_val.requires_grad_(True)
                self.value = self.extended_val.transpose(-1, 0)[0]
            self.unconstrained_value = value
        else:
            transform = self.transforms[0]
            unconstrained_sample = self.transform_from_constrained_to_unconstrained(
                value
            )
            if isinstance(value, float_types):
                unconstrained_sample.requires_grad_(True)
            constrained_sample = self.transform_from_unconstrained_to_constrained(
                unconstrained_sample
            )
            self.value = constrained_sample
            self.unconstrained_value = unconstrained_sample
            if isinstance(self.distribution, dist.Beta):
                sample = torch.cat(
                    (
                        constrained_sample.unsqueeze(-1),
                        (1 - constrained_sample).unsqueeze(-1),
                    ),
                    -1,
                )
            else:
                sample = constrained_sample
            self.jacobian = transform.log_abs_det_jacobian(
                unconstrained_sample, sample
            ).sum()

    def update_fields(
        self,
        value: Optional[Tensor],
        obs_value: Optional[Tensor],
        should_transform: bool = False,
    ):
        """
        Updates log probability, transforms and is_discrete, value,
        unconstrained_value and jacobian parameters in the Variable.

        :param value: the value of the tensor if available, otherwise, a new
        initialized value will be set.
        :params obs_value: the observation value if observed else None.
        :params should_transform: a boolean to identify whether to set
        transforms and transformed values.
        """
        if self.transforms is None:
            self.transforms = (
                get_transforms(self.distribution)
                if obs_value is None and should_transform
                else []
            )

        if value is None:
            value = self.initialize_value(obs_value)
        self.update_value(value)
        self.update_log_prob()
        if self.is_discrete is None:
            self.is_discrete = is_discrete(self.distribution)

    def update_log_prob(self) -> None:
        """
        Computes the log probability of the value of the random varialble
        """
        try:
            # pyre-fixme
            self.log_prob = self.distribution.log_prob(self.value).sum()
        except (RuntimeError, ValueError) as e:
            # pyre-fixme
            if not self.distribution.support.check(self.value):
                self.log_prob = tensor(float("-Inf"))
            else:
                raise e

    def set_transform(self, transform: List):
        """
        Sets the variable transform to the transform passed in.

        :param transform: the transform value to the set the variable transform
        to
        """
        self.transforms = transform

    def transform_from_unconstrained_to_constrained(
        self, unconstrained_sample: Tensor
    ) -> Tensor:
        """
        Transforms the sample from unconstrained space to constrained space.

        :param unconstrained_sample: unconstrained sample
        :returns: constrained sample
        """
        if len(self.transforms) == 0:
            return unconstrained_sample
        if isinstance(self.distribution, dist.Beta) and isinstance(
            self.transforms[0], dist.StickBreakingTransform
        ):
            val = self.transforms[0]._call(unconstrained_sample)
            return val.transpose(-1, 0)[0].T
        return self.transforms[0]._call(unconstrained_sample)

    def transform_from_constrained_to_unconstrained(
        self, constrained_sample: Tensor
    ) -> Tensor:
        """
        Transforms the sample from constrained space to unconstrained space.

        :param unconstrained_sample: unconstrained sample
        :returns: constrained sample
        """
        if len(self.transforms) == 0:
            return constrained_sample

        if isinstance(self.distribution, dist.Beta) and isinstance(
            self.transforms[0], dist.StickBreakingTransform
        ):
            val = torch.cat(
                (
                    constrained_sample.unsqueeze(-1),
                    (1 - constrained_sample).unsqueeze(-1),
                ),
                -1,
            )
            return self.transforms[0]._inverse(val)

        return self.transforms[0]._inverse(constrained_sample)
