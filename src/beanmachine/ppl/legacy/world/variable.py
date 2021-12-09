# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import torch.distributions as dist
from beanmachine.ppl.inference.utils import safe_log_prob_sum
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.utils import (
    BetaDimensionTransform,
    get_default_transforms,
    initialize_value,
)
from torch import Tensor
from torch.distributions import Distribution


@dataclasses.dataclass(eq=True, frozen=True)
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


@dataclasses.dataclass
class TransformData:
    transform_type: TransformType
    transforms: Optional[List]


@dataclasses.dataclass
class Variable:
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
    log_prob: Tensor
    transformed_value: Tensor
    jacobian: Tensor
    proposal_distribution: Optional[ProposalDistribution] = None
    parent: Set[Optional[RVIdentifier]] = dataclasses.field(default_factory=set)
    children: Set[Optional[RVIdentifier]] = dataclasses.field(default_factory=set)
    transform: dist.Transform = dist.transforms.identity_transform
    cardinality: int = -1

    @property
    def is_discrete(self) -> bool:
        # pyre-fixme
        return self.distribution.support.is_discrete

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
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

    def copy(self):
        """
        Makes a copy of self and returns it.

        :returns: copy of self.
        """
        return dataclasses.replace(
            self,
            parent=self.parent.copy(),
            children=self.children.copy(),
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
            self.transform = dist.transforms.identity_transform
        else:
            self.set_transform(transform_data, proposer)

        if value is None:
            if obs_value is None:
                value = initialize_value(self.distribution, initialize_from_prior)
            else:
                value = obs_value
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
            self.transform = get_default_transforms(self.distribution)
        elif transform_data.transform_type == TransformType.NONE:
            if isinstance(self.distribution, dist.Beta) and getattr(
                proposer, "reshape_untransformed_beta_to_dirichlet", False
            ):
                self.transform = BetaDimensionTransform()
            else:
                self.transform = dist.transforms.identity_transform
        else:
            if transform_data.transforms is None:
                self.transform = dist.transforms.identity_transform
            else:
                self.transform = dist.ComposeTransform(transform_data.transforms)

    def update_jacobian(self):
        transformed_value = self.transform(self.value)
        self.jacobian = -self.transform.log_abs_det_jacobian(
            self.value, transformed_value
        ).sum()

    def inverse_transform_value(self, transformed_value: Tensor) -> Tensor:
        """
        Transforms the value from transformed space to original space.

        :param transformed_value: transformed space sample
        :returns: original space sample
        """
        return self.transform.inv(transformed_value)

    def transform_value(self, value: Tensor) -> Tensor:
        """
        Transforms the value from original space to transformed space.

        :param value: value in original space
        :returns: value in transformed space
        """
        return self.transform(value)
