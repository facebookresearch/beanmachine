# Copyright (c) Facebook, Inc. and its affiliates
from typing import List

import torch.distributions as dist
import torch.distributions.constraints as constraints
import torch.tensor as tensor
from torch.distributions import Distribution


def is_discrete(distribution: Distribution) -> bool:
    """
    Checks whether a distribution is discrete or not.

    :param distribution: the distribution to check
    :returns: a boolean that is true if the distribution is discrete and false
    otherwise
    """
    # pyre-fixme
    support = distribution.support
    if (
        isinstance(support, constraints._Boolean)
        or isinstance(support, constraints._IntegerGreaterThan)
        or isinstance(support, constraints._IntegerInterval)
        or isinstance(support, constraints._IntegerLessThan)
    ):
        return True
    return False


def get_transforms(distribution: Distribution) -> List:
    """
    Get transforms of a distribution to transform it from constrained space
    into unconstrained space.

    :param distribution: the distribution to check
    :returns: the list of transforms that need to be applied to the distribution
    to transform it from constrained space into unconstrained space
    """
    # pyre-fixme
    support = distribution.support
    if is_discrete(distribution):
        return []

    if isinstance(support, constraints._Real):
        return []

    if isinstance(support, constraints._Interval):
        lower_bound = tensor(support.lower_bound)
        upper_bound = tensor(support.upper_bound)
        if lower_bound.mean() != 0.0 or upper_bound.mean() != 1.0:
            raise ValueError(
                "Only distributions with 0 as lower bound and 1 as upper bound is supported"
            )

        return [dist.StickBreakingTransform()]

    if isinstance(support, constraints._GreaterThan) or isinstance(
        support, constraints._GreaterThanEq
    ):
        lower_bound = tensor(support.lower_bound)
        if lower_bound.sum() > 0.0:
            raise ValueError("Only distributions with 0 as lower bound is supported")

        return [dist.ExpTransform()]

    if isinstance(support, constraints._Simplex):
        return [dist.StickBreakingTransform()]
    return []
