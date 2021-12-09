# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class NormalNormalModel(object):
    def __init__(self, mu: Tensor, std: Tensor, sigma: Tensor):
        self.mu_ = mu
        self.std_ = std
        self.sigma_ = sigma

    @bm.random_variable
    def normal_p(self):
        return dist.Normal(self.mu_, self.std_)

    @bm.random_variable
    def normal(self):
        return dist.Normal(self.normal_p(), self.sigma_)
