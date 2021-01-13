# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.utils import safe_log_prob_sum
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.utils import (
    BetaDimensionTransform,
    get_default_transforms,
    is_discrete,
)
from torch import Tensor
from torch.distributions import Distribution


@dataclass(eq=True, frozen=True)
class ProposalDistribution:
    proposal_distribution: Optional[Any]
    requires_transform: bool
    requires_reshape: bool
    arguments: Dict

    def __str__(self) -> str:
        try:
            str_ouput = str(self.proposal_distribution)
        except NotImplementedError:
            str_ouput = str(type(self.proposal_distribution))
        return str_ouput


class TransformType(Enum):
    NONE = 0
    DEFAULT = 1
    CUSTOM = 2


@dataclass
class TransformData:
    transform_type: TransformType
    transforms: Optional[List]


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

    @bm.random_variable
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
    is_discrete: bool
    transforms: List
    transformed_value: Tensor
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
        try:
            str_ouput = str(self.value) + " from " + str(self.distribution)
        except NotImplementedError:
            str_ouput = str(self.value) + " from " + str(type(self.distribution))
        return str_ouput

    def initialize_value(
        self, obs: Optional[Tensor], initialize_from_prior: bool = False
    ) -> Tensor:
        """
        Initialized the Variable value

        :param is_obs: the boolean representing whether the node is an
        observation or not
        :param initialize_from_prior: if true, returns sample from prior
        :returns: the value to the set the Variable value to
        """
        distribution = self.distribution
        if obs is not None:
            return obs

        # pyre-fixme
        sample_val = distribution.sample()
        # pyre-fixme
        support = distribution.support
        if initialize_from_prior:
            return sample_val
        elif isinstance(support, dist.constraints._Real):
            return torch.zeros(
                sample_val.shape, dtype=sample_val.dtype, device=sample_val.device
            )
        elif isinstance(support, dist.constraints._Simplex):
            value = torch.ones(
                sample_val.shape, dtype=sample_val.dtype, device=sample_val.device
            )
            return value / sample_val.shape[-1]
        elif isinstance(support, dist.constraints._GreaterThan):
            return (
                torch.ones(
                    sample_val.shape, dtype=sample_val.dtype, device=sample_val.device
                )
                + support.lower_bound
            )
        elif isinstance(support, dist.constraints._Boolean):
            return dist.Bernoulli(
                torch.ones(sample_val.shape, device=sample_val.device) / 2
            ).sample()
        elif isinstance(support, dist.constraints._Interval):
            lower_bound = (
                torch.ones(sample_val.shape, device=sample_val.device)
                * support.lower_bound
            )
            upper_bound = (
                torch.ones(sample_val.shape, device=sample_val.device)
                * support.upper_bound
            )
            return dist.Uniform(lower_bound, upper_bound).sample()
        elif isinstance(support, dist.constraints._IntegerInterval):
            integer_interval = support.upper_bound - support.lower_bound
            return dist.Categorical(
                (torch.ones(integer_interval, device=sample_val.device)).expand(
                    sample_val.shape + (integer_interval,)
                )
            ).sample()
        elif isinstance(support, dist.constraints._IntegerGreaterThan):
            return (
                torch.ones(
                    sample_val.shape, dtype=sample_val.dtype, device=sample_val.device
                )
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
            self.is_discrete,
            self.transforms,
            self.transformed_value,
            self.jacobian,
        )

    def update_value(self, value: Tensor) -> None:
        """
        Update the value of the variable

        :param value: the value to update the variable to
        """
        self.transformed_value = self.transform_value(value)
        if not self.is_discrete:
            self.transformed_value.requires_grad_(True)
        self.value = self.inverse_transform_value(self.transformed_value)
        self.update_jacobian()

    def update_fields(
        self,
        value: Optional[Tensor],
        obs_value: Optional[Tensor],
        transform_data: TransformData,
        proposer,
        initialize_from_prior: bool = False,
    ):
        """
        Updates log probability, transforms and is_discrete, value,
        transformed_value and jacobian parameters in the Variable.

        :param value: the value of the tensor if available, otherwise, a new
        initialized value will be set.
        :params obs_value: the observation value if observed else None.
        :params should_transform: a boolean to identify whether to set
        :param initialize_from_prior: if true, returns sample from prior
        transforms and transformed values.
        """
        if obs_value is not None:
            self.transforms = []
        else:
            self.set_transform(transform_data, proposer)

        if self.is_discrete is None:
            self.is_discrete = is_discrete(self.distribution)
        if value is None:
            value = self.initialize_value(obs_value, initialize_from_prior)
        self.update_value(value)
        self.update_log_prob()

    def update_log_prob(self) -> None:
        """
        Computes the log probability of the value of the random varialble
        """
        self.log_prob = safe_log_prob_sum(self.distribution, self.value)

    def set_transform(self, transform_data: TransformData, proposer):
        """
        Sets the variable transform to the transform passed in.

        :param transform: the transform value to the set the variable transform
        to
        """
        if transform_data.transform_type == TransformType.DEFAULT:
            self.transforms = get_default_transforms(self.distribution)
        elif transform_data.transform_type == TransformType.NONE:
            if (
                isinstance(self.distribution, dist.Beta)
                and proposer is not None
                and hasattr(proposer, "reshape_untransformed_beta_to_dirichlet")
                and proposer.reshape_untransformed_beta_to_dirichlet
            ):
                self.transforms = [BetaDimensionTransform()]
            else:
                self.transforms = []
        else:
            if transform_data.transforms is None:
                self.transforms = []
            else:
                # pyre-fixme
                self.transforms = transform_data.transforms

    def update_jacobian(self):
        temp = self.value
        self.jacobian = torch.zeros((), dtype=temp.dtype)
        for transform in self.transforms:
            transformed_value = transform(temp)
            self.jacobian -= transform.log_abs_det_jacobian(
                temp, transformed_value
            ).sum()
            temp = transformed_value

    def inverse_transform_value(self, transformed_value: Tensor) -> Tensor:
        """
        Transforms the value from transformed space to original space.

        :param transformed_value: transformed space sample
        :returns: original space sample
        """
        temp = transformed_value
        for transform in reversed(self.transforms):
            temp = transform.inv(temp)
        return temp

    def transform_value(self, value: Tensor) -> Tensor:
        """
        Transforms the value from original space to transformed space.

        :param value: value in original space
        :returns: value in transformed space
        """
        temp = value
        for transform in self.transforms:
            temp = transform(temp)
        return temp
