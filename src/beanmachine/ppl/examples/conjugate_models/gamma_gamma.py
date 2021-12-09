# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class GammaGammaModel(object):
    def __init__(self, shape: Tensor, rate: Tensor, alpha: Tensor):
        self.shape_ = shape
        self.rate_ = rate
        self.alpha_ = alpha

    @bm.random_variable
    def gamma_p(self):
        return dist.Gamma(self.shape_, self.rate_)

    @bm.random_variable
    def gamma(self):
        return dist.Gamma(self.alpha_, self.gamma_p())
