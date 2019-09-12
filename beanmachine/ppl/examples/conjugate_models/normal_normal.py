# Copyright (c) Facebook, Inc. and its affiliates
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import sample


class NormalNormalModel(object):
    def __init__(self, mu, std, sigma):
        self.mu_ = mu
        self.std_ = std
        self.sigma_ = sigma

    @sample
    def normal_p(self):
        return dist.Normal(self.mu_, self.std_)

    @sample
    def normal(self):
        return dist.Normal(self.normal_p(), self.sigma_)
