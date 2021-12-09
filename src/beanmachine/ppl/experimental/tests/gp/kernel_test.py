# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import gpytorch
import gpytorch.distributions as dist
import torch
from beanmachine.ppl.experimental.gp import kernels
from beanmachine.ppl.model.rv_identifier import RVIdentifier


class KernelTests(unittest.TestCase):
    def setUp(self):
        @bm.random_variable
        def prior():
            return dist.base_distributions.HalfNormal(1.0)

        self.prior = prior

    def test_smoke(self):
        for k, v in gpytorch.kernels.__dict__.items():
            if "Kernel" in k:
                assert k in kernels.__dict__.keys()
                assert issubclass(kernels.__dict__[k], v)

    def test_prior(self):
        kernel = kernels.CosineKernel()
        assert not kernel.has_lengthscale
        kernel = kernels.RBFKernel()
        assert kernel.has_lengthscale
        assert isinstance(kernel.lengthscale, torch.Tensor)
        kernel = kernels.RBFKernel(lengthscale_prior=self.prior)
        assert isinstance(kernel.lengthscale, RVIdentifier)

    def test_covar(self):
        kernel = kernels.MaternKernel(lengthscale_prior=self.prior)
        assert kernel.training
        # In train mode, can only run be called within inference
        self.assertRaises(AttributeError, kernel, torch.zeros(1))

    def test_setter(self):
        kernel = kernels.MaternKernel(lengthscale_prior=self.prior)
        assert isinstance(kernel.lengthscale, RVIdentifier)
        # Convert to gpytorch kernel
        kernel.eval()
        kernel.lengthscale = torch.ones(1)
        assert isinstance(kernel.lengthscale, torch.Tensor)
        kernel.train()
        assert isinstance(kernel.lengthscale, RVIdentifier)

        kernel = kernels.CylindricalKernel(5, kernel, alpha_prior=self.prior)
        assert isinstance(kernel.alpha, RVIdentifier)
        # Convert to gpytorch kernel
        kernel.eval()
        kernel.alpha = torch.ones(1)
        assert isinstance(kernel.alpha, torch.Tensor)
