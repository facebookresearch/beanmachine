import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.distribution.flat import Flat
from beanmachine.ppl.experimental.neutra.iafmcmc_proposer import IAFMCMCProposer
from beanmachine.ppl.experimental.neutra.maskedautoencoder import MaskedAutoencoder
from beanmachine.ppl.world import Variable, World
from torch import nn, tensor as tensor


class IAFMCMCProposerDoAdaptationTest(unittest.TestCase):
    class Target(dist.Distribution):
        has_enumerate_support = False
        support = dist.constraints.real
        has_rsample = True
        arg_constraints = {}

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
            return IAFMCMCProposerDoAdaptationTest.Target(
                self.foo()[0][0], self.foo()[0][1]
            )

    def setUp(self):
        torch.manual_seed(11)

    def optimizer_func(self, lr, weight_decay):
        return lambda parameters: torch.optim.Adam(
            parameters, lr=1e-4, weight_decay=1e-5
        )

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
            distribution=IAFMCMCProposerDoAdaptationTest.Target(
                tensor([0.1, 0.1])[0], tensor([0.1, 0.1])[1]
            ),
            value=tensor([0.1, 0.1]),
            log_prob=IAFMCMCProposerDoAdaptationTest.Target(
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
        # pass the parameters in to construct IAF

        optimizer_func = self.optimizer_func(1e-4, 1e-5)

        training_sample_size = 100

        iaf_proposer = IAFMCMCProposer(
            training_sample_size,
            length,
            in_layer,
            network_architecture,
            optimizer_func,
            True,
            [],
        )

        for i in range(1, 1000):
            iaf_proposer.do_adaptation(foo_key, world, tensor(1.0), 1000, True, i)
        x = dist.Normal(torch.zeros([5000, 2]), torch.ones([5000, 2])).sample()

        q_z, elbo, log_jacobian = iaf_proposer.mapping(x)
        # make sure there is no crash for propose() function
        proposed_value_list = []
        for _ in range(1000):
            proposed_value, proposed_log_prob, ax = iaf_proposer.propose(foo_key, world)
            proposed_value_list.append(proposed_value.unsqueeze(1))

        proposed = torch.stack(proposed_value_list, dim=0)

        z_mean = q_z[-1][:, 1].mean()
        x_mean = q_z[-1][:, 0].mean()

        p_z_mean = proposed[:, 1].mean()
        p_x_mean = proposed[:, 0].mean()

        # here we consider result from the Pytorch version as ground truth.
        # Thus, for ground truth value, mean of z is 0.032, and the mean
        # of x is -0.0474.
        self.assertAlmostEqual(z_mean, 0.032, delta=0.05)
        self.assertAlmostEqual(x_mean, -0.0474, delta=0.05)
        self.assertAlmostEqual(p_z_mean, 0.032, delta=0.08)
        self.assertAlmostEqual(p_x_mean, -0.0474, delta=0.08)
