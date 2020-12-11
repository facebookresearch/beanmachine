# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import gpytorch
import torch
from beanmachine.ppl.experimental.gp import likelihoods
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class LikelihoodTest(unittest.TestCase):
    def test_smoke(self):
        for k in gpytorch.likelihoods.__dict__.keys():
            if "Likelihood" not in k:
                continue
            assert k in likelihoods.__dict__.keys()
            assert issubclass(likelihoods.__dict__[k], likelihoods.GpytorchMixin)

    def test_forward_smoke(self):
        n = gpytorch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        l = likelihoods.GaussianLikelihood()  # noqa: E741
        assert isinstance(l(n), RVIdentifier)
        l.eval()
        assert isinstance(l(torch.zeros(2)), torch.distributions.Normal)

    def test_prior(self):
        n = torch.distributions.HalfNormal(torch.ones(2))
        n = bm.random_variable(lambda: n)
        l = likelihoods.BetaLikelihood(scale_prior=n)  # noqa: E741
        assert isinstance(l.scale, RVIdentifier)
        l.eval()
        l.scale = torch.ones(4, 2)
        assert isinstance(l.scale, torch.Tensor)
        l.train()
        assert isinstance(l.scale, RVIdentifier)
