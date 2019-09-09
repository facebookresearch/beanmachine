# Copyright (c) Facebook, Inc. and its affiliates
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import sample


class CategoricalDirichletModel(object):
    def __init__(self, alpha):
        self.alpha_ = alpha

    @sample
    def dirichlet(self):
        return dist.Dirichlet(self.alpha_)

    @sample
    def categorical(self):
        return dist.Categorical(self.dirichlet())
