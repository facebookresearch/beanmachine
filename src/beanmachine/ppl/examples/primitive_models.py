# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class PrimitiveModel(ABC):
    @abstractmethod
    def x(self) -> dist.Distribution:
        pass


class NormalModel(PrimitiveModel):
    def __init__(self, mu: Tensor, sigma: Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Normal(self.mu, self.sigma)


class GammaModel(PrimitiveModel):
    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.alpha = alpha
        self.beta = beta

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Gamma(self.alpha, self.beta)


class PoissonModel(PrimitiveModel):
    def __init__(self, rate: Tensor) -> None:
        self.rate = rate

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Poisson(self.rate)


class DirichletModel(PrimitiveModel):
    def __init__(self, alpha: Tensor) -> None:
        self.alpha = alpha

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Dirichlet(self.alpha)
