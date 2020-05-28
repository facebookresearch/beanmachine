# Copyright (c) Facebook, Inc. and its affiliates
import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from torch import Tensor


class GammaNormalModel(object):
    def __init__(self, shape: Tensor, rate: Tensor, mu: Tensor):
        self.shape_ = shape
        self.rate_ = rate
        self.mu_ = mu

    @bm.random_variable
    def gamma(self):
        return dist.Gamma(self.shape_, self.rate_)

    @bm.random_variable
    def normal(self):
        return dist.Normal(self.mu_, 1 / torch.sqrt(self.gamma()))
