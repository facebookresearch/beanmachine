# Copyright (c) Facebook, Inc. and its affiliates
from collections.abc import Iterable
from typing import Iterable as IterableType, List, Type, Union, overload

import torch
import torch.distributions as dist
import torch.distributions.constraints as constraints
from torch import Tensor, tensor
from torch.distributions import Distribution
from torch.distributions.transforms import Transform


ConstraintType = Union[constraints.Constraint, Type]


class BetaDimensionTransform(Transform):
    bijective = True

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

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        return tensor(0.0)


def is_discrete(distribution: Distribution) -> bool:
    """
    Checks whether a distribution is discrete or not.

    :param distribution: the distribution to check
    :returns: a boolean that is true if the distribution is discrete and false
    otherwise
    """
    return any(
        is_constraint_eq(
            # pyre-fixme
            distribution.support,
            (
                constraints.boolean,
                constraints.integer_interval,
                constraints._IntegerGreaterThan,
                constraints._IntegerLessThan,
            ),
        )
    )


def _unwrap(constraint: ConstraintType):
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
    specified in :mod:`torch.distributions.constraints`. If `check_constraints`
    is a single `Constraint` type or instance this returns a `True` if the
    given `constraint` matches  `check_constraints`. Otherwise, if
    `check_constraints` is an iterable, this returns a `bool` list that
    represents an element-wise check.

    :param constraint: A constraint class or instance.
    :param check_constraints: A constraint class or instance or an iterable
        containing constraint classes or instances to check against.
    :returns: bool (or a list of bool) values indicating if the given constraint
        equals the constraint in `check_constraints`.
    """
    if isinstance(check_constraints, Iterable):
        return [_is_constraint_eq(constraint, c) for c in check_constraints]
    return _is_constraint_eq(constraint, check_constraints)


def get_default_transforms(distribution: Distribution) -> List:
    """
    Get transforms of a distribution to transform it from constrained space
    into unconstrained space.

    :param distribution: the distribution to check
    :returns: the list of transforms that need to be applied to the distribution
    to transform it from constrained space into unconstrained space
    """
    # pyre-fixme
    support = distribution.support
    # pyre-fixme
    sample = distribution.sample()
    if is_discrete(distribution):
        return []
    elif is_constraint_eq(support, constraints.real):
        return []

    elif is_constraint_eq(support, constraints.interval):
        lower_bound = support.lower_bound
        if not isinstance(lower_bound, Tensor):
            lower_bound = tensor(lower_bound, dtype=sample.dtype)
        upper_bound = support.upper_bound
        if not isinstance(upper_bound, Tensor):
            upper_bound = tensor(upper_bound, dtype=sample.dtype)

        lower_bound_zero = dist.AffineTransform(-lower_bound, 1.0)
        upper_bound_one = dist.AffineTransform(0, 1.0 / (upper_bound - lower_bound))
        beta_dimension = BetaDimensionTransform()
        stick_breaking = dist.StickBreakingTransform().inv

        return [lower_bound_zero, upper_bound_one, beta_dimension, stick_breaking]

    elif is_constraint_eq(support, constraints.greater_than) or isinstance(
        support, constraints.greater_than_eq
    ):
        lower_bound = support.lower_bound
        if not isinstance(lower_bound, Tensor):
            lower_bound = tensor(lower_bound, dtype=sample.dtype)
        lower_bound_zero = dist.AffineTransform(-lower_bound, 1.0)
        log_transform = dist.ExpTransform().inv

        return [lower_bound_zero, log_transform]

    elif is_constraint_eq(support, constraints.less_than):
        upper_bound = support.upper_bound
        if not isinstance(upper_bound, Tensor):
            upper_bound = tensor(upper_bound, dtype=sample.dtype)

        upper_bound_zero = dist.AffineTransform(-upper_bound, 1.0)
        flip_to_greater = dist.AffineTransform(0, -1.0)
        log_transform = dist.ExpTransform().inv

        return [upper_bound_zero, flip_to_greater, log_transform]

    elif is_constraint_eq(support, constraints.simplex):
        return [dist.StickBreakingTransform().inv]

    return []
