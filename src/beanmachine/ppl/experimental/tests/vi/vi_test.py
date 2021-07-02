# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import scipy.stats
import torch
import torch.distributions as dist
import torch.nn as nn
from beanmachine.ppl.distribution.flat import Flat
from beanmachine.ppl.experimental.vi.variational_infer import (
    MeanFieldVariationalInference,
    VariationalInference,
)
from beanmachine.ppl.world import World
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal


class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.

    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    support = constraints.real

    def __init__(self, validate_args=None):
        d = 2
        batch_shape, event_shape = (1,), (d,)
        super(NealsFunnel, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size()
        eps = _standard_normal(
            (sample_shape[0], 2), dtype=torch.float, device=torch.device("cpu")
        )
        z = torch.zeros(eps.shape)
        z[..., 1] = torch.tensor(3.0) * eps[..., 1]
        z[..., 0] = torch.exp(z[..., 1] / 2.0) * eps[..., 0]
        return z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value[..., 0]
        y = value[..., 1]

        log_prob = dist.Normal(0, 3).log_prob(y)
        log_prob += dist.Normal(0, torch.exp(y / 2)).log_prob(x)

        return log_prob


class BayesianRobustLinearRegression:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.X_train = torch.randn(n, d)
        self.beta_truth = torch.randn(d + 1, 1)
        noise = dist.StudentT(df=4.0).sample((n, 1))
        self.y_train = (
            torch.cat((self.X_train, torch.ones(n, 1)), -1).mm(self.beta_truth) + noise
        )

    @bm.random_variable
    def beta(self):
        return dist.Independent(
            dist.StudentT(df=4.0 * torch.ones(self.d + 1)),
            1,
        )

    @bm.random_variable
    def X(self):
        return dist.Normal(0, 1)  # dummy

    @bm.random_variable
    def y(self):
        X_with_ones = torch.cat((self.X(), torch.ones(self.X().shape[0], 1)), -1)
        b = self.beta().squeeze()
        if b.dim() == 1:
            b = b.unsqueeze(0)
        mu = X_with_ones.mm(b.T)
        return dist.Independent(
            dist.StudentT(df=4.0, loc=mu, scale=1),
            1,
        )


class NormalNormal:
    @bm.random_variable
    def mu(self):
        return dist.Normal(torch.zeros(1), 10 * torch.ones(1))

    @bm.random_variable
    def x(self, i):
        return dist.Normal(self.mu(), torch.ones(1))


class MeanFieldVariationalInferTest(unittest.TestCase):
    def setUp(self) -> None:
        MeanFieldVariationalInference.set_seed(1)

    def test_neals_funnel(self):
        nf = bm.random_variable(NealsFunnel)

        vi = MeanFieldVariationalInference().infer(
            queries=[nf()],
            observations={},
            num_iter=400,
            base_dist=dist.Normal,
            base_args={"loc": torch.zeros([1]), "scale": torch.ones([1])},
            num_elbo_mc_samples=100,
            lr=1e-3,
        )

        # compare 1D marginals of empirical distributions using 2-sample K-S test
        nf_samples = NealsFunnel().sample((50,)).squeeze().numpy()
        vi_samples = vi(nf()).sample((50,)).detach().numpy()

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue, 0.05
        )
        self.assertGreaterEqual(
            scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue, 0.05
        )

    def test_normal_normal(self):
        model = NormalNormal()
        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            base_dist=dist.Normal,
            base_args={
                "loc": nn.Parameter(torch.tensor([0.0])),
                "scale": nn.Parameter(torch.tensor([1.0])),
            },
        )

        mu_approx = vi_dicts(model.mu())
        sample_mean = mu_approx.sample((100,)).mean()
        self.assertGreater(sample_mean, 5.0)

        sample_var = mu_approx.sample((100,)).var()
        self.assertGreater(sample_var, 0.1)

    def test_normal_normal_studentt_base_dist(self):
        model = NormalNormal()
        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            base_dist=dist.StudentT,
            base_args={"df": nn.Parameter(torch.tensor([10.0]))},
        )

        mu_approx = vi_dicts(model.mu())
        sample_mean = mu_approx.sample((100,)).mean()
        self.assertGreater(sample_mean, 5.0)

        sample_var = mu_approx.sample((100,)).var()
        self.assertGreater(sample_var, 0.1)

    def test_brlr(self):
        brlr = BayesianRobustLinearRegression(n=83, d=7)
        vi_dicts = MeanFieldVariationalInference().infer(
            queries=[brlr.beta()],
            observations={
                brlr.X(): brlr.X_train,
                brlr.y(): brlr.y_train,
            },
            lr=1e-2,
        )
        beta_samples = vi_dicts(brlr.beta()).sample((100,))
        for i in range(beta_samples.shape[1]):
            self.assertLess(
                torch.norm(beta_samples[:, i].mean() - brlr.beta_truth[i]),
                0.5,
            )

    def test_constrained_positive_reals(self):
        exp = dist.Exponential(torch.tensor([1.0]))
        positive_rv = bm.random_variable(lambda: exp)
        vi_dicts = MeanFieldVariationalInference().infer(
            queries=[positive_rv()],
            observations={},
        )
        self.assertAlmostEqual(
            vi_dicts(positive_rv()).sample((100,)).mean().item(),
            exp.mean,
            delta=0.2,
        )

    def test_constrained_interval(self):
        beta = dist.Beta(torch.tensor([1.0]), torch.tensor([1.0]))
        interval_rv = bm.random_variable(lambda: beta)
        vi_dicts = MeanFieldVariationalInference().infer(
            queries=[interval_rv()],
            observations={},
        )
        self.assertAlmostEqual(
            vi_dicts(interval_rv()).sample((100,)).mean().item(),
            beta.mean,
            delta=0.2,
        )


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

        # 100 iterations in single call
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

    def test_normal_normal_guide_step(self):
        model = NormalNormal()

        @bm.param
        def phi():
            return torch.zeros(2)  # mean, log std

        @bm.random_variable
        def q_mu():
            params = phi()
            return dist.Normal(params[0], params[1].exp())

        # 100 steps, each 1 iteration
        opt_params = None
        for _ in range(100):
            opt_params = VariationalInference().infer(
                {model.mu(): q_mu()},
                observations={
                    model.x(1): torch.tensor(9.0),
                    model.x(2): torch.tensor(10.0),
                },
                num_iter=1,
                lr=1e0,
                params=opt_params,
                progress_bar=False,
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

    def test_logistic_regression(self):
        n, d = 1000, 10
        W = torch.randn(d)
        X = torch.randn((n, d))
        Y = torch.bernoulli(torch.sigmoid(X @ W))

        @bm.random_variable
        def x():
            return Flat(shape=X.shape)

        @bm.random_variable
        def y():
            return dist.Bernoulli(probs=Y.clone().detach().float())

        @bm.param
        def w():
            return torch.randn(d)

        @bm.random_variable
        def q_y():
            weights = w()
            data = x()
            p = torch.sigmoid(data @ weights.T)
            return dist.Bernoulli(p)

        opt_params = VariationalInference().infer(
            {y(): q_y()},
            observations={
                x(): X.float(),
            },
            num_iter=1000,
            lr=1e-2,
        )
        l2_error = (opt_params[w()] - W).norm()
        self.assertLess(l2_error, 0.5)
