# Copyright (c) Facebook, Inc. and its affiliates
import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class BetaBernoulliModel(object):
    def __init__(self, alpha: Tensor, beta: Tensor):
        self.alpha_ = alpha
        self.beta_ = beta

    @bm.random_variable
    def theta(self):
        return dist.Beta(self.alpha_, self.beta_)

    @bm.random_variable
    def y(self, i):
        return dist.Bernoulli(self.theta())
