# @lint-ignore-every LICENSELINT
# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numbers

import torch
from torch.distributions import constraints, Distribution

# Helper function
def sum_rightmost(value, dim):
    if isinstance(value, numbers.Number):
        return value
    if dim < 0:
        dim += value.dim()
    if dim == 0:
        return value
    if dim >= value.dim():
        return value.sum()
    return value.reshape(value.shape[:-dim] + (-1,)).sum(-1)


# For MAP estimation
class Delta(Distribution):
    has_rsample = True
    arg_constraints = {"v": constraints.real, "log_density": constraints.real}
    support = constraints.real

    def __init__(self, v, log_density=0.0, event_dim=0, validate_args=None):
        if event_dim > v.dim():
            raise ValueError(
                "Expected event_dim <= v.dim(), actual {} vs {}".format(
                    event_dim, v.dim()
                )
            )
        batch_dim = v.dim() - event_dim
        batch_shape = v.shape[:batch_dim]
        event_shape = v.shape[batch_dim:]
        if isinstance(log_density, numbers.Number):
            log_density = v.new_empty(batch_shape).fill_(log_density)
        elif validate_args and log_density.shape != batch_shape:
            raise ValueError(
                "Expected log_density.shape = {}, actual {}".format(
                    log_density.shape, batch_shape
                )
            )
        self.v = v
        self.log_density = log_density
        super(Delta, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(batch_shape + self.event_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def shape(self, sample_shape=torch.Size()):  # noqa: B008
        return sample_shape + self.batch_shape + self.event_shape

    @property
    def event_dim(self):
        return len(self.event_shape)

    def rsample(self, sample_shape=torch.Size()):  # noqa: B008
        shape = sample_shape + self.v.shape
        return self.v.expand(shape)

    def log_prob(self, x):
        v = self.v.expand(self.shape())
        log_prob = ((x == v).type(x.dtype)).log()
        log_prob = sum_rightmost(log_prob, self.event_dim)
        return log_prob + self.log_density

    @property
    def mean(self):
        return self.v

    @property
    def variance(self):
        return torch.zeros_like(self.v)
