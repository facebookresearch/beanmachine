# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import beanmachine.ppl as bm
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class BetaBernoulliBasicModel(object):
    def __init__(self, alpha: Tensor, beta: Tensor):
        self.alpha_ = alpha
        self.beta_ = beta

    @bm.random_variable
    def theta(self):
        return dist.Beta(self.alpha_, self.beta_)

    @bm.random_variable
    def y(self, i):
        return dist.Bernoulli(self.theta())

    def gen_obs(self, num_obs: int) -> Dict[RVIdentifier, Tensor]:
        true_theta = 0.75
        obs = {}
        for i in range(0, num_obs):
            obs[self.y(i)] = dist.Bernoulli(true_theta).sample()
        return obs


class BetaBernoulliOpsModel(BetaBernoulliBasicModel):
    @bm.functional
    def sum_y(self):
        sum = 0.0
        for i in range(0, 5):
            sum = sum + self.y(i)
        return sum


class BetaBernoulliScaleHyperParameters(BetaBernoulliBasicModel):
    def scale_alpha(self):
        factor = 2.0
        for i in range(0, 3):
            factor = factor * i
        return factor

    @bm.random_variable
    def theta(self):
        return dist.Beta(self.alpha_ + self.scale_alpha(), self.beta_ + 2.0)
