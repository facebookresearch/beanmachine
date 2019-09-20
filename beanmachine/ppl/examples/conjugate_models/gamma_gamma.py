# Copyright (c) Facebook, Inc. and its affiliates
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import sample
from torch import Tensor


class GammaGammaModel(object):
    def __init__(self, shape: Tensor, rate: Tensor, alpha: Tensor):
        self.shape_ = shape
        self.rate_ = rate
        self.alpha_ = alpha

    @sample
    def gamma_p(self):
        return dist.Gamma(self.shape_, self.rate_)

    @sample
    def gamma(self):
        return dist.Gamma(self.alpha_, self.gamma_p())
