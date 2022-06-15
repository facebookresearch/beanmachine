# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from torch import Tensor


class GammaNormalModel:
    def __init__(self, shape: Tensor, rate: Tensor, mu: Tensor) -> None:
        self.shape_ = shape
        self.rate_ = rate
        self.mu_ = mu

    @bm.random_variable
    def gamma(self) -> dist.Distribution:
        return dist.Gamma(self.shape_, self.rate_)

    @bm.random_variable
    def normal(self) -> dist.Distribution:
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        return dist.Normal(self.mu_, 1 / torch.sqrt(self.gamma()))
