import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.distribution.flat import Flat
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from beanmachine.ppl.experimental.neutra.train import IAFMap
from beanmachine.ppl.world import Variable, World
from torch import nn


class TraininfTest(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.zeros(1, 2), torch.ones(1, 2))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.ones(1, 2))

    class Target(dist.Distribution):
        has_enumerate_support = False
        support = dist.constraints.real
        has_rsample = True

        def __init__(self, x_foo, z_foo):
            super().__init__()
            self.x_foo = x_foo
            self.z_foo = z_foo

        def rsample(self, sample_shape):
            return torch.zeros(sample_shape)

        def sample(self):
            return torch.tensor(0.0)

        def log_prob(self, value):
            return dist.Normal(0.0, 3.0).log_prob(self.z_foo) + dist.Normal(
                0, (self.z_foo / 2).exp()
            ).log_prob(self.x_foo)

    class NealFunnel(object):
        @bm.random_variable
        def foo(self):
            return Flat(2)

        @bm.random_variable
        def bar(self):
            return TraininfTest.Target(self.foo()[0][0], self.foo()[0][1])

    def setUp(self):
        torch.manual_seed(11)

    def test_normal_normal(self):
        # set up the world in Bean machine
        world = World()
        model = self.SampleModel()
        foo_key = model.foo()
        bar_key = model.bar()

        world.set_observations({bar_key: tensor([0.1, 0.1])})
        # set up the node_var in Bean machine
        world_vars = world.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=dist.Normal(torch.zeros(1, 2), torch.ones(1, 2)),
            value=tensor([0.5, 0.5]),
            log_prob=dist.Normal(torch.zeros(1, 2), torch.ones(1, 2))
            .log_prob(tensor([0.5, 0.5]))
            .sum(),
            parent=set(),
            children=set({bar_key}),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor([0.5, 0.5]),
            jacobian=tensor(0.0),
        )

        world_vars[bar_key] = Variable(
            distribution=dist.Normal(tensor([0.5, 0.5]), torch.ones(1, 2)),
            value=tensor([0.1, 0.1]),
            log_prob=dist.Normal(tensor([0.5, 0.5]), torch.ones(1, 2))
            .log_prob(tensor([0.1, 0.1]))
            .sum(),
            parent=set({foo_key}),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor([0.1, 0.1]),
            jacobian=tensor(0.0),
        )

        # set the parameters needed in IAF
        # build masked autoencoder net
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        length = 2
        lr = 1e-4
        weight_decay = 1e-5
        num_sample = 100

        # pass the parameters in to construct IAF
        model = IAFMap(
            foo_key, world, length, in_layer, network_architecture, True, num_sample
        )
        optimizer = torch.optim.Adam(
            model.model_.parameters(), lr=lr, weight_decay=weight_decay
        )

        model.model_.train()
        for _ in range(1000):
            model.train_iaf(world, optimizer)

        model.model_.eval()
        x = dist.Normal(torch.zeros([5000, 2]), torch.ones([5000, 2])).sample()
        q_z, elbo, log_jacobian = model.model_(x)
        obs_mu = q_z[-1].mean().data
        truth = 5000 * obs_mu / (5000 + 1)
        self.assertAlmostEqual(obs_mu.numpy(), truth.numpy(), delta=0.00001)

    def test_neal_funnel(self):
        # set up the world in Bean machine
        model = self.NealFunnel()
        world = World()
        foo_key = model.foo()
        bar_key = model.bar()

        world.set_observations({bar_key: tensor([0.1, 0.1])})
        # set up the node_var in Bean machine
        world_vars = world.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=Flat(2),
            value=tensor([0.0, 0.0]),
            log_prob=Flat(2).log_prob(tensor([0.0, 0.0])),
            parent=set(),
            children=set({bar_key}),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor([0.0, 0.0]),
            jacobian=tensor(0.0),
        )

        world_vars[bar_key] = Variable(
            distribution=TraininfTest.Target(
                tensor([0.1, 0.1])[0], tensor([0.1, 0.1])[1]
            ),
            value=tensor([0.1, 0.1]),
            log_prob=TraininfTest.Target(
                tensor([0.1, 0.1])[0], tensor([0.1, 0.1])[1]
            ).log_prob(tensor([0.1, 0.1])),
            parent=set({foo_key}),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor([0.1, 0.1]),
            jacobian=tensor(0.0),
        )

        # set the parameters needed in IAF
        # build masked autoencoder net
        in_layer = 2
        out_layer = 4
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )

        length = 2
        lr = 1e-4
        weight_decay = 1e-5
        num_sample = 100

        # pass the parameters in to construct IAF
        model = IAFMap(
            foo_key, world, length, in_layer, network_architecture, True, num_sample
        )

        optimizer = torch.optim.Adam(
            model.model_.parameters(), lr=lr, weight_decay=weight_decay
        )
        model.model_.train()
        for _ in range(1000):
            model.train_iaf(world, optimizer)

        model.model_.eval()
        x = dist.Normal(torch.zeros([10000, 2]), torch.ones([10000, 2])).sample()
        q_z, elbo, log_jacobian = model.model_(x)
        z_mean = q_z[-1][:, 1].mean()
        x_mean = q_z[-1][:, 0].mean()
        # here we consider result from the Pytorch version as ground truth.
        # Thus, for ground truth value, mean of z is 0.032, and the mean
        # of x is -0.0474. because in test.
        self.assertAlmostEqual(z_mean, 0.032, delta=0.05)
        self.assertAlmostEqual(x_mean, -0.0474, delta=0.05)
