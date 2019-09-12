# Copyright (c) Facebook, Inc. and its affiliates
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import sample


class BetaBinomialModel(object):
    def __init__(self, alpha, beta, trials):
        self.alpha_ = alpha
        self.beta_ = beta
        self.trials_ = trials

    @sample
    def beta(self):
        return dist.Beta(self.alpha_, self.beta_)

    @sample
    def binomial(self):
        return dist.Binomial(self.trials_, self.beta())
