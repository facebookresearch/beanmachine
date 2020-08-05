# Copyright (c) Facebook, Inc. and its affiliates
from typing import List

import torch
import torch.distributions as dist
import torch.distributions.constraints as constraints
from torch import Tensor, tensor
from torch.distributions import Distribution
from torch.distributions.transforms import Transform


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
    elif isinstance(support, constraints._Real):
        return []

    elif isinstance(support, constraints._Interval):
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

    elif isinstance(support, constraints._GreaterThan) or isinstance(
        support, constraints._GreaterThanEq
    ):
        lower_bound = support.lower_bound
        if not isinstance(lower_bound, Tensor):
            lower_bound = tensor(lower_bound, dtype=sample.dtype)
        lower_bound_zero = dist.AffineTransform(-lower_bound, 1.0)
        log_transform = dist.ExpTransform().inv

        return [lower_bound_zero, log_transform]

    elif isinstance(support, constraints._LessThan):
        upper_bound = support.upper_bound
        if not isinstance(upper_bound, Tensor):
            upper_bound = tensor(upper_bound, dtype=sample.dtype)

        upper_bound_zero = dist.AffineTransform(-upper_bound, 1.0)
        flip_to_greater = dist.AffineTransform(0, -1.0)
        log_transform = dist.ExpTransform().inv

        return [upper_bound_zero, flip_to_greater, log_transform]

    elif isinstance(support, constraints._Simplex):
        return [dist.StickBreakingTransform().inv]

    return []
