# Copyright (c) Facebook, Inc. and its affiliates
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import sample
from torch import Tensor


class NormalNormalModel(object):
    def __init__(self, mu: Tensor, std: Tensor, sigma: Tensor):
        self.mu_ = mu
        self.std_ = std
        self.sigma_ = sigma

    @sample
    def normal_p(self):
        return dist.Normal(self.mu_, self.std_)

    @sample
    def normal(self):
        return dist.Normal(self.normal_p(), self.sigma_)
