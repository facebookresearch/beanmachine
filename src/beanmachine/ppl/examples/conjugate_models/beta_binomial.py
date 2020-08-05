# Copyright (c) Facebook, Inc. and its affiliates
import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class BetaBinomialModel(object):
    def __init__(self, alpha: Tensor, beta: Tensor, trials: Tensor):
        self.alpha_ = alpha
        self.beta_ = beta
        self.trials_ = trials

    @bm.random_variable
    def beta(self):
        return dist.Beta(self.alpha_, self.beta_)

    @bm.random_variable
    def binomial(self):
        return dist.Binomial(self.trials_, self.beta())
