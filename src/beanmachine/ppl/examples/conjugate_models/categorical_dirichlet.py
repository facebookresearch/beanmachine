# Copyright (c) Facebook, Inc. and its affiliates
import beanmachine.ppl as bm
import torch.distributions as dist
from torch import Tensor


class CategoricalDirichletModel(object):
    def __init__(self, alpha: Tensor):
        self.alpha_ = alpha

    @bm.random_variable
    def dirichlet(self):
        return dist.Dirichlet(self.alpha_)

    @bm.random_variable
    def categorical(self):
        return dist.Categorical(self.dirichlet())
