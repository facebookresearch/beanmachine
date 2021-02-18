# Copyright (c) Facebook, Inc. and its affiliates
from collections.abc import Iterable
from typing import Iterable as IterableType, Type, Union, overload

import torch
import torch.distributions as dist
import torch.distributions.constraints as constraints
from torch.distributions import Distribution
from torch.distributions.transforms import Transform


ConstraintType = Union[constraints.Constraint, Type]


class BetaDimensionTransform(Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real_vector

    def __eq__(self, other):
        return isinstance(other, BetaDimensionTransform)

    def _call(self, x):
        """
        Abstract method to compute forward transformation.
        """
        return torch.cat((x.unsqueeze(-1), (1 - x).unsqueeze(-1)), -1)

    def _inverse(self, y):
        """
        Abstract method to compute inverse transformation.
        """
        return y.transpose(-1, 0)[0]

    def forward_shape(self, shape):
        return shape + (2,)

    def inverse_shape(self, shape):
        return shape[:-1]

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        return torch.zeros_like(x)


def _unwrap(constraint: ConstraintType):
    if isinstance(constraint, constraints.independent):
        return _unwrap(constraint.base_constraint)
    return constraint if isinstance(constraint, type) else constraint.__class__


def _is_constraint_eq(constraint1: ConstraintType, constraint2: ConstraintType):
    return _unwrap(constraint1) == _unwrap(constraint2)


@overload
def is_constraint_eq(
    constraint: ConstraintType, check_constraints: ConstraintType
) -> bool:
    ...


@overload
def is_constraint_eq(
    constraint: ConstraintType, check_constraints: IterableType[ConstraintType]
) -> IterableType[bool]:
    ...


def is_constraint_eq(
    constraint: ConstraintType,
    check_constraints: Union[ConstraintType, IterableType[ConstraintType]],
) -> Union[bool, IterableType[bool]]:
    """
    This provides an equality check that works for different constraints
    specified in :mod:`torch.distributions.constraints`. If `constraint` is
    `constraints.Independent`, then the `base_constraint` is checked. If
    `check_constraints` is a single `Constraint` type or instance this
    returns a `True` if the given `constraint` matches `check_constraints`.
    Otherwise, if `check_constraints` is an iterable, this returns a `bool`
    list that represents an element-wise check.

    :param constraint: A constraint class or instance.
    :param check_constraints: A constraint class or instance or an iterable
        containing constraint classes or instances to check against.
    :returns: bool (or a list of bool) values indicating if the given constraint
        equals the constraint in `check_constraints`.
    """
    if isinstance(check_constraints, Iterable):
        return [_is_constraint_eq(constraint, c) for c in check_constraints]
    return _is_constraint_eq(constraint, check_constraints)


def get_default_transforms(distribution: Distribution) -> dist.Transform:
    """
    Get transforms of a distribution to transform it from constrained space
    into unconstrained space.

    :param distribution: the distribution to check
    :returns: a Transform that need to be applied to the distribution
    to transform it from constrained space into unconstrained space
    """
    # pyre-fixme
    if distribution.support.is_discrete:
        return dist.transforms.identity_transform
    else:
        return dist.biject_to(distribution.support).inv
