import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.distribution.flat import Flat
from beanmachine.ppl.experimental.neutra.iafflow import InverseAutoregressiveFlow
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import Mode
from beanmachine.ppl.world import Variable
from torch import nn


class IAFTest(unittest.TestCase):
    def tearDown(self):
        StatisticalModel.reset()

    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), tensor(1.0))

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
            return IAFTest.Target(self.foo()[0][0], self.foo()[0][1])

    def setUp(self):
        torch.manual_seed(11)

    def test_elbo_change(self):
        # set up the world in Bean machine
        model = self.SampleModel()
        world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        StatisticalModel.set_mode(Mode.INFERENCE)
        world.set_observations({bar_key: tensor(0.1)})
        world_vars = world.variables_.vars()
        world_vars[foo_key] = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.5),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.5)),
            parent=set(),
            children=set({bar_key}),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor(0.5),
            jacobian=tensor(0.0),
        )

        world_vars[bar_key] = Variable(
            distribution=dist.Normal(tensor(0.5), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor(0.1),
            jacobian=tensor(0.0),
        )
        # build the masked autoencoder network

        based_distribution = dist.Normal(tensor(0.0), tensor(1.0))

        x = dist.Normal(torch.zeros(1, 2), torch.ones(1, 2)).sample()

        # pass the parameters to IAF
        in_layer = 2
        out_layer = in_layer * 2
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        network_architecture = MaskedAutoencoder(
            in_layer, out_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        model = InverseAutoregressiveFlow(
            based_distribution, network_architecture, 2, in_layer, foo_key, world, True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        model.train()
        f_z, elbo, log_jacobian = model(x)
        tmp = elbo

        # maximaze elbo for 10 times and check if the elbo goes to larger
        for _ in range(10):
            f_z, elbo, log_jacobian = model(x)
            self.assertTrue(elbo >= tmp)
            tmp = elbo
            loss = -elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_normal_normal(self):
        # set up the world in Bean machine
        world = StatisticalModel.reset()
        model = self.SampleModel()
        foo_key = model.foo()
        bar_key = model.bar()

        StatisticalModel.set_mode(Mode.INFERENCE)
        world.set_observations({bar_key: tensor(0.1)})
        world_vars = world.variables_.vars()

        world_vars[foo_key] = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.5),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.5)),
            parent=set(),
            children=set({bar_key}),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor(0.5),
            jacobian=tensor(0.0),
        )

        world_vars[bar_key] = Variable(
            distribution=dist.Normal(tensor(0.5), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=tensor(0.1),
            jacobian=tensor(0.0),
        )

        # set the parameters needed in IAF
        based_distribution = dist.Normal(torch.zeros(1), torch.ones(1))
        input_layer = 2
        output_layer = input_layer * 2
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        # build the masked autoencoder network
        network_architecture = MaskedAutoencoder(
            input_layer, output_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        # pass all the parameters to IAF
        model = InverseAutoregressiveFlow(
            based_distribution,
            network_architecture,
            2,
            input_layer,
            foo_key,
            world,
            True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        model.train()
        # train for 1000 steps and check the learnt mapping.
        for _ in range(1000):
            x = dist.Normal(torch.zeros(100, 2), torch.ones(100, 2)).sample()
            z_f, elbo, log_jacobian = model(x)
            loss = -(torch.sum(elbo)) / 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        x = dist.Normal(torch.zeros([5000, 2]), torch.ones([5000, 2])).sample()
        q_z, elbo, log_jacobian = model(x)
        obs_mu = q_z[-1].mean().data
        truth = 5000 * obs_mu / (5000 + 1)
        self.assertAlmostEqual(obs_mu.numpy(), truth.numpy(), delta=0.00001)

    def test_neal_funnel_change(self):
        # set up the world in Bean machine
        model = self.NealFunnel()
        world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        StatisticalModel.set_mode(Mode.INFERENCE)

        world.set_observations({bar_key: tensor([0.1, 0.1])})
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
            distribution=IAFTest.Target(tensor([0.1, 0.1])[0], tensor([0.1, 0.1])[1]),
            value=tensor([0.1, 0.1]),
            log_prob=IAFTest.Target(
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
        based_distribution = dist.Normal(torch.zeros(1), torch.ones(1))
        input_layer = 2
        output_layer = input_layer * 2
        hidden_layer = 30
        n_block = 4
        seed_num = 11
        # build the masked autoencoder network
        network_architecture = MaskedAutoencoder(
            input_layer, output_layer, nn.ELU(), hidden_layer, n_block, seed_num
        )
        # pass all the parameters to IAF
        model = InverseAutoregressiveFlow(
            based_distribution,
            network_architecture,
            2,
            input_layer,
            foo_key,
            world,
            True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        model.train()
        # train for 1000 steps and check the learnt mapping.
        for _ in range(1000):
            x = dist.Normal(torch.zeros(100, 2), torch.ones(100, 2)).sample()
            z_f, elbo, log_jacobian = model(x)
            loss = -(torch.sum(elbo)) / 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        x = dist.Normal(torch.zeros([10000, 2]), torch.ones([10000, 2])).sample()
        q_z, elbo, log_jacobian = model(x)
        z_mean = q_z[-1][:, 1].mean()
        x_mean = q_z[-1][:, 0].mean()
        # here we consider result from the Pytorch version as ground truth.
        # Thus, for ground truth value, mean of z is 0.032, and the mean
        # of x is -0.0474.
        self.assertAlmostEqual(z_mean, 0.032, delta=0.05)
        self.assertAlmostEqual(x_mean, -0.0474, delta=0.05)
