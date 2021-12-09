# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class BetaBinomialModel(object):
    """This Bean Machine model is an example of conjugacy, where
    the prior and the likelihood are the Beta and the Binomial
    distributions respectively. Conjugacy means the posterior
    will also be in the same family as the prior, Beta.
    The random variable names theta and x follow the
    typical presentation of the conjugate prior relation in the
    form of p(theta|x) = p(x|theta) * p(theta)/p(x).
    Note: Variable names here follow those used on:
    https://en.wikipedia.org/wiki/Conjugate_prior
    """

    def __init__(self, alpha: Tensor, beta: Tensor, n: Tensor):
        self.alpha_ = alpha
        self.beta_ = beta
        self.n_ = n

    @bm.random_variable
    def theta(self):
        return dist.Beta(self.alpha_, self.beta_)

    @bm.random_variable
    def x(self):
        return dist.Binomial(self.n_, self.theta())
