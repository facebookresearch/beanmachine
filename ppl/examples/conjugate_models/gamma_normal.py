# Copyright (c) Facebook, Inc. and its affiliates
import torch
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import sample


class GammaNormalModel(object):
    def __init__(self, shape, rate, mu):
        self.shape_ = shape
        self.rate_ = rate
        self.mu_ = mu

    @sample
    def gamma(self):
        return dist.Gamma(self.shape_, self.rate_)

    @sample
    def normal(self):
        return dist.Normal(self.mu_, 1 / torch.sqrt(self.gamma()))
