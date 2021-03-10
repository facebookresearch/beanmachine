# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.nn as nn
from beanmachine.ppl.experimental.vi.variational_infer import (
    VariationalInference,
)
from beanmachine.ppl.world import World
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


class NormalNormal:
    @bm.random_variable
    def mu(self):
        return dist.Normal(torch.zeros(1), 10 * torch.ones(1))

    @bm.random_variable
    def x(self, i):
        return dist.Normal(self.mu(), torch.ones(1))


class StochasticVariationalInferTest(unittest.TestCase):
    def setUp(self) -> None:
        VariationalInference.set_seed(1)

    def test_normal_normal_guide(self):
        model = NormalNormal()

        @bm.param
        def phi():
            return torch.zeros(2)  # mean, log std

        @bm.random_variable
        def q_mu():
            params = phi()
            return dist.Normal(params[0], params[1].exp())

        opt_params = VariationalInference().infer(
            {model.mu(): q_mu()},
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e0,
        )
        q_mu_id = q_mu()
        mu_approx = None
        with World() as w:
            w.set_params(opt_params)
            w.call(q_mu_id)
            mu_approx = w.get_node_in_world_raise_error(q_mu_id).distribution

        sample_mean = mu_approx.sample((100, 1)).mean()
        self.assertGreater(sample_mean, 5.0)

        sample_var = mu_approx.sample((100, 1)).var()
        self.assertGreater(sample_var, 0.1)

    def test_conditional_guide(self):
        @bm.random_variable
        def mu():
            return dist.Normal(torch.zeros(1), torch.ones(1))

        @bm.random_variable
        def alpha():
            return dist.Normal(torch.zeros(1), torch.ones(1))

        @bm.random_variable
        def x(i):
            return dist.Normal(mu() + alpha(), torch.ones(1))

        @bm.param
        def phi_mu():
            return torch.zeros(2)  # mean, log std

        @bm.random_variable
        def q_mu():
            params = phi_mu()
            return dist.Normal(params[0] - alpha(), params[1].exp())

        @bm.param
        def phi_alpha():
            return torch.zeros(2)  # mean, log std

        @bm.random_variable
        def q_alpha():
            params = phi_alpha()
            return dist.Normal(params[0], params[1].exp())

        opt_params = VariationalInference().infer(
            {
                mu(): q_mu(),
                alpha(): q_alpha(),
            },
            observations={
                x(1): torch.tensor(9.0),
                x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e0,
        )
        q_mu_id = q_mu()
        alpha_id = alpha()

        mu_approx = None
        with World() as w:
            w.set_params(opt_params)
            w.set_observations({alpha_id: torch.tensor(10.0)})
            w.call(q_mu_id)
            mu_approx = w.get_node_in_world_raise_error(q_mu_id).distribution
        sample_mean_alpha_10 = mu_approx.sample((100, 1)).mean()

        mu_approx = None
        with World() as w:
            w.set_params(opt_params)
            w.set_observations({alpha_id: torch.tensor(-10.0)})
            w.call(q_mu_id)
            mu_approx = w.get_node_in_world_raise_error(q_mu_id).distribution
        sample_mean_alpha_neg_10 = mu_approx.sample((100, 1)).mean()

        self.assertGreater(sample_mean_alpha_neg_10, sample_mean_alpha_10)
