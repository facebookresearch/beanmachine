# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import gpytorch
import torch
from beanmachine.ppl.experimental.gp import likelihoods
from beanmachine.ppl.model.utils import RVIdentifier


class LikelihoodTest(unittest.TestCase):
    def test_smoke(self):
        for k in gpytorch.likelihoods.__dict__.keys():
            if "Likelihood" not in k:
                continue
            assert k in likelihoods.__dict__.keys()
            assert issubclass(likelihoods.__dict__[k], likelihoods.GpytorchMixin)

    def test_forward_smoke(self):
        n = gpytorch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        assert isinstance(likelihoods.GaussianLikelihood().marginal(n), RVIdentifier)
        assert isinstance(likelihoods.GaussianLikelihood().forward(n), RVIdentifier)
        assert isinstance(likelihoods.GaussianLikelihood()(n), RVIdentifier)
