# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import gpytorch
import gpytorch.distributions as dist
import torch
from beanmachine.ppl.experimental.gp import kernels
from beanmachine.ppl.model.utils import RVIdentifier


class KernelTests(unittest.TestCase):
    def test_smoke(self):
        for k, v in gpytorch.kernels.__dict__.items():
            if "Kernel" in k:
                assert k in kernels.__dict__.keys()
                assert issubclass(kernels.__dict__[k], v)

    def test_prior(self):
        @bm.random_variable
        def normal():
            return dist.base_distributions.Normal(0.0, 1.0)

        kernel = kernels.CosineKernel()
        assert not kernel.has_lengthscale
        kernel = kernels.RBFKernel()
        assert kernel.has_lengthscale
        assert isinstance(kernel.lengthscale, torch.Tensor), type(kernel.lengthscale)
        kernel = kernels.RBFKernel(lengthscale_prior=normal)
        assert isinstance(kernel.lengthscale, RVIdentifier), type(kernel.lengthscale)

    def test_covar(self):
        @bm.random_variable
        def normal():
            return dist.base_distributions.Normal(0.0, 1.0)

        kernel = kernels.MaternKernel(lengthscale_prior=normal)
        lazy_covar = kernel(torch.zeros(1))
        # kernel.evaluate() can only be called within inference
        self.assertRaises(AttributeError, lazy_covar.evaluate)
