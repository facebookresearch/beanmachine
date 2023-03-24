# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from torch import Tensor


class ConjugateModel(ABC):
    """
    The Bean Machine models in this module are examples of conjugacy. Conjugacy
    means the posterior will also be in the same family as the prior. The random
    variable names theta and x follow the typical presentation of the conjugate
    prior relation in the form of p(theta|x) = p(x|theta) * p(theta)/p(x).

    See:
    https://en.wikipedia.org/wiki/Conjugate_prior
    """

    x_dim = 0
    """
    Number of indices in likelihood.
    """

    @abstractmethod
    def theta(self) -> dist.Distribution:
        """
        Prior of a conjugate model.
        """
        pass

    @abstractmethod
    def x(self, *args) -> dist.Distribution:
        """
        Likelihood of a conjugate model.
        """
        pass


class BetaBernoulliModel(ConjugateModel):
    x_dim = 1

    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.alpha_ = alpha
        self.beta_ = beta

    @bm.random_variable
    def theta(self) -> dist.Distribution:
        return dist.Beta(self.alpha_, self.beta_)

    @bm.random_variable
    def x(self, i: int) -> dist.Distribution:
        return dist.Bernoulli(self.theta())


class BetaBinomialModel(ConjugateModel):
    def __init__(self, alpha: Tensor, beta: Tensor, n: Tensor) -> None:
        self.alpha_ = alpha
        self.beta_ = beta
        self.n_ = n

    @bm.random_variable
    def theta(self) -> dist.Distribution:
        return dist.Beta(self.alpha_, self.beta_)

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Binomial(self.n_, self.theta())


class CategoricalDirichletModel(ConjugateModel):
    def __init__(self, alpha: Tensor) -> None:
        self.alpha_ = alpha

    @bm.random_variable
    def theta(self) -> dist.Distribution:
        return dist.Dirichlet(self.alpha_)

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Categorical(self.theta())


class GammaGammaModel(ConjugateModel):
    def __init__(self, shape: Tensor, rate: Tensor, alpha: Tensor) -> None:
        self.shape_ = shape
        self.rate_ = rate
        self.alpha_ = alpha

    @bm.random_variable
    def theta(self) -> dist.Distribution:
        return dist.Gamma(self.shape_, self.rate_)

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Gamma(self.alpha_, self.theta())


class GammaNormalModel(ConjugateModel):
    def __init__(self, shape: Tensor, rate: Tensor, mu: Tensor) -> None:
        self.shape_ = shape
        self.rate_ = rate
        self.mu_ = mu

    @bm.random_variable
    def theta(self) -> dist.Distribution:
        return dist.Gamma(self.shape_, self.rate_)

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Normal(self.mu_, torch.tensor(1) / torch.sqrt(self.theta()))


class NormalNormalModel(ConjugateModel):
    def __init__(self, mu: Tensor, sigma: Tensor, std: Tensor) -> None:
        self.mu = mu
        self.sigma = sigma
        self.std = std

    @bm.random_variable
    def theta(self) -> dist.Distribution:
        return dist.Normal(self.mu, self.sigma)

    @bm.random_variable
    def x(self) -> dist.Distribution:
        return dist.Normal(self.theta(), self.std)


class MV_NormalNormalModel(ConjugateModel):
    def __init__(self, mu, sigma, std) -> None:
        self.mu = mu
        self.sigma = sigma
        self.std = std

    @bm.random_variable
    def theta(self):
        return dist.MultivariateNormal(self.mu, self.sigma)

    @bm.random_variable
    def x(self):
        return dist.MultivariateNormal(self.theta(), self.std)
