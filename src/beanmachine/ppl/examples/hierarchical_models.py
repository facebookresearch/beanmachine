# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from torch import Tensor, tensor


class UniformNormalModel:
    def __init__(self, lo: Tensor, hi: Tensor, std: Tensor) -> None:
        self.lo = lo
        self.hi = hi
        self.std = std

    @bm.random_variable
    def mean(self):
        return dist.Uniform(self.lo, self.hi)

    @bm.random_variable
    def obs(self):
        return dist.Normal(self.mean(), self.std)


class UniformBernoulliModel:
    def __init__(self, lo: Tensor, hi: Tensor) -> None:
        self.lo = lo
        self.hi = hi

    @bm.random_variable
    def prior(self):
        return dist.Uniform(self.lo, self.hi)

    @bm.random_variable
    def likelihood(self):
        return dist.Bernoulli(self.prior())

    @bm.random_variable
    def likelihood_i(self, i):
        return dist.Bernoulli(self.prior())

    @bm.random_variable
    def likelihood_dynamic(self, i):
        assert self.lo.ndim == self.hi.ndim == 0
        if self.likelihood_i(i).item() > 0:
            return dist.Normal(torch.zeros(1), torch.ones(1))
        else:
            return dist.Normal(5.0 * torch.ones(1), torch.ones(1))

    @bm.random_variable
    def likelihood_reg(self, x):
        assert self.lo.ndim == self.hi.ndim == 0
        return dist.Normal(self.prior() * x, torch.tensor(1.0))


class LogisticRegressionModel:
    @bm.random_variable
    def theta_0(self):
        return dist.Normal(tensor(0.0), tensor(1.0))

    @bm.random_variable
    def theta_1(self):
        return dist.Normal(tensor(0.0), tensor(1.0))

    @bm.random_variable
    def x(self, i):
        return dist.Normal(tensor(0.0), tensor(1.0))

    @bm.random_variable
    def y(self, i):
        y = self.theta_1() * self.x(i) + self.theta_0()
        probs = 1 / (1 + (y * -1).exp())
        return dist.Bernoulli(probs)
