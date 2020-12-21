# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.distribution.flat import Flat
from beanmachine.ppl.experimental.neutra.iafmcmc_infer import IAFMCMCinference
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from torch import nn, tensor as tensor


class TestIAFInfer(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.zeros(2), torch.ones(2))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.ones(2))

    class Target(dist.Distribution):
        has_enumerate_support = False
        support = dist.constraints.real
        has_rsample = True
        arg_constraints = {}

        def __init__(self, z_foo):
            super().__init__()
            if z_foo.size() == torch.Size([1, 2]):
                self.z_foo = z_foo.view(2, 1)
            else:
                self.z_foo = z_foo

        def rsample(self, sample_shape):
            return torch.zeros(sample_shape)

        def sample(self):
            return torch.tensor(0.0)

        def log_prob(self, value):
            return dist.Normal(0.0, 3.0).log_prob(self.z_foo[1]) + dist.Normal(
                0, (self.z_foo[1] / 2).exp()
            ).log_prob(self.z_foo[0])

    class NealFunnel(object):
        @bm.random_variable
        def foo(self):
            return Flat(2)

        @bm.random_variable
        def bar(self):
            return TestIAFInfer.Target(self.foo())

    class TargetRosenBrock(dist.Distribution):
        has_enumerate_support = False
        support = dist.constraints.real
        has_rsample = True

        def __init__(self, x_foo, *args):
            super().__init__()
            if x_foo.size() == torch.Size([1, 2]):
                self.x_foo = x_foo.view(2, 1)
            else:
                self.x_foo = x_foo
            self.mu, self.a, self.b, self.c = args

        def rsample(self, sample_shape):
            return torch.zeros(sample_shape)

        def sample(self):
            return torch.tensor(0.0)

        def log_prob(self, value):
            part1 = (
                -self.a * (-(self.x_foo[0]) + self.mu) * (-(self.x_foo[0]) + self.mu)
            )
            part2 = (
                -self.b
                * (self.x_foo[1] - self.c * self.x_foo[0] * self.x_foo[0])
                * (self.x_foo[1] - self.c * self.x_foo[0] * self.x_foo[0])
            )
            return part1 + part2

    class RosenBrock(object):
        @bm.random_variable
        def foo(self):
            return Flat(2)

        @bm.random_variable
        def bar(self):
            return TestIAFInfer.TargetRosenBrock(self.foo(), 1.0, 1.0, 100.0, 1.0)

    def setUp(self):
        torch.manual_seed(11)

    def optimizer_func(self, lr, weight_decay):
        return lambda parameters: torch.optim.Adam(
            parameters, lr=1e-4, weight_decay=1e-5
        )

    def test_normal_normal(self):
        model = self.SampleModel()
        training_sample_size = 100
        length = 2
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        optimizer_func = self.optimizer_func(1e-4, 1e-5)

        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            {},
        )
        samples_iaf = iaf.infer(
            queries=[model.foo()],
            observations={model.bar(): tensor([0.0, 0.0])},
            num_samples=100,
            num_chains=1,
            num_adaptive_samples=1000,
        )

        f_z = samples_iaf[model.foo()]
        obs_mu_1 = f_z[:, 1].mean().data
        truth_1 = 5000 * obs_mu_1 / (5000 + 1)

        obs_mu_2 = f_z[:, 0].mean().data
        truth_2 = 5000 * obs_mu_2 / (5000 + 1)

        self.assertAlmostEqual(obs_mu_1, truth_1, delta=0.0001)
        self.assertAlmostEqual(obs_mu_2, truth_2, delta=0.0001)

    def test_neal_funnel(self):
        model = self.NealFunnel()
        training_sample_size = 100
        length = 2
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        optimizer_func = self.optimizer_func(1e-4, 1e-5)

        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            {},
        )
        samples_iaf = iaf.infer(
            queries=[model.foo()],
            observations={model.bar(): tensor(0.0)},
            num_samples=200,
            num_chains=1,
            num_adaptive_samples=1000,
        )
        f_z = samples_iaf[model.foo()]
        print("f_z", f_z)
        self.assertAlmostEqual(f_z[0][:, 1].mean(), 0.828, delta=0.8)
        self.assertAlmostEqual(f_z[0][:, 0].mean(), 0.0390, delta=0.8)

    def test_rosen_brock(self):
        model = self.RosenBrock()
        training_sample_size = 100
        length = 2
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        optimizer_func = self.optimizer_func(1e-4, 1e-5)

        iaf = IAFMCMCinference(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            {},
        )
        samples_iaf = iaf.infer(
            queries=[model.foo()],
            observations={model.bar(): tensor([0.1, 0.1])},
            num_samples=100,
            num_chains=1,
            num_adaptive_samples=1000,
        )
        f_z = samples_iaf[model.foo()]
        self.assertAlmostEqual(f_z[0][:, 1].mean(), 0.3419, delta=0.05)
        self.assertAlmostEqual(f_z[0][:, 0].mean(), 0.5380, delta=0.05)
