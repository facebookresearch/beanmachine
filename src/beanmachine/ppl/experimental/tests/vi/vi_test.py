# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Optional

import beanmachine.ppl as bm
import numpy
import pytest
import scipy.stats
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.vi import ADVI, MAP, VariationalInfer
from beanmachine.ppl.experimental.vi.gradient_estimator import (
    monte_carlo_approximate_sf,
)
from beanmachine.ppl.experimental.vi.variational_world import VariationalWorld
from beanmachine.ppl.world import init_from_prior, RVDict
from torch import optim
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal

cpu_device = torch.device("cpu")


class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.

    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

    support = constraints.real

    def __init__(self, validate_args=None):
        d = 2
        batch_shape, event_shape = torch.Size([]), (d,)
        super(NealsFunnel, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def rsample(self, sample_shape=None):
        if not sample_shape:
            sample_shape = torch.Size((1,))
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
    def __init__(
        self,
        mean_0: float = 0.0,
        variance_0: float = 1.0,
        variance_x: float = 1.0,
        device: Optional[torch.device] = cpu_device,
    ):
        self.device = device
        self.mean_0 = mean_0
        self.variance_0 = variance_0
        self.variance_x = variance_x

    @bm.random_variable
    def mu(self):
        return dist.Normal(
            torch.zeros(1).to(self.device), 10 * torch.ones(1).to(self.device)
        )

    @bm.random_variable
    def x(self, i):
        return dist.Normal(self.mu(), torch.ones(1).to(self.device))

    def conjugate_posterior(self, observations: RVDict) -> torch.dist:
        # Normal-Normal conjugate prior formula (https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution)
        expected_variance = 1 / (
            (1 / self.variance_0) + (sum(observations.values()) / self.variance_x)
        )
        expected_std = numpy.sqrt(expected_variance)
        expected_mean = expected_variance * (
            (self.mean_0 / self.variance_0)
            + (sum(observations.values()) / self.variance_x)
        )
        return dist.Normal(expected_mean, expected_std)


class LogScaleNormal:
    @bm.param
    def phi(self):
        return torch.zeros(2)  # mean, log std

    @bm.random_variable
    def q_mu(self):
        params = self.phi()
        return dist.Normal(params[0], params[1].exp())


class BinaryGaussianMixture:
    @bm.random_variable
    def h(self, i):
        return dist.Bernoulli(0.5)

    @bm.random_variable
    def x(self, i):
        return dist.Normal(self.h(i).float(), 0.1)


class TestAutoGuide:
    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_can_use_functionals(self, auto_guide_inference):
        test_rv = bm.random_variable(lambda: dist.Normal(0, 1))
        test_functional = bm.functional(lambda: test_rv() ** 2)
        auto_guide = auto_guide_inference(
            queries=[test_rv(), test_functional()],
            observations={},
        )
        world = auto_guide.infer(num_steps=10)
        assert world.call(test_functional()) is not None

    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_neals_funnel(self, auto_guide_inference):
        nf = bm.random_variable(NealsFunnel)

        auto_guide = auto_guide_inference(
            queries=[nf()],
            observations={},
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-1),
        )
        world = auto_guide.infer(
            num_steps=100,
        )

        if auto_guide_inference == ADVI:
            # compare 1D marginals of empirical distributions using 2-sample K-S test
            nf_samples = NealsFunnel().sample((20,)).squeeze().numpy()
            vi_samples = (
                world.get_guide_distribution(nf())
                .sample((20,))
                .detach()
                .squeeze()
                .numpy()
            )
            assert (
                scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue >= 0.05
            )
            assert (
                scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue >= 0.05
            )
        else:
            vi_samples = world.get_guide_distribution(nf()).v.detach().squeeze().numpy()
            map_truth = [0, -4.5]

            assert numpy.isclose(map_truth, vi_samples, atol=0.05).all().item()

    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_normal_normal(self, auto_guide_inference):
        model = NormalNormal()
        auto_guide = auto_guide_inference(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            optimizer=lambda params: torch.optim.Adam(params, lr=1e0),
        )
        world = auto_guide.infer(
            num_steps=100,
        )

        mu_approx = world.get_guide_distribution(model.mu())
        sample_mean = mu_approx.sample((100,)).mean()
        assert sample_mean > 5.0

        if auto_guide_inference == ADVI:
            sample_var = mu_approx.sample((100,)).var()
            assert sample_var > 0.1

    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_brlr(self, auto_guide_inference):
        brlr = BayesianRobustLinearRegression(n=100, d=7)
        auto_guide = auto_guide_inference(
            queries=[brlr.beta()],
            observations={
                brlr.X(): brlr.X_train,
                brlr.y(): brlr.y_train,
            },
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-1),
        )
        world = auto_guide.infer(
            num_steps=100,
        )
        beta_samples = world.get_guide_distribution(brlr.beta()).sample((100,))
        for i in range(beta_samples.shape[1]):
            assert torch.norm(beta_samples[:, i].mean() - brlr.beta_truth[i]) < 0.2

    @pytest.mark.parametrize(
        "auto_guide_inference, expected", [(ADVI, 1.0), (MAP, 0.0)]
    )
    def test_constrained_positive_reals(self, auto_guide_inference, expected):
        exp = dist.Exponential(torch.tensor([1.0]))
        positive_rv = bm.random_variable(lambda: exp)
        auto_guide = auto_guide_inference(queries=[positive_rv()], observations={})
        world = auto_guide.infer(num_steps=100)
        assert (
            abs(
                world.get_guide_distribution(positive_rv()).sample((100,)).mean().item()
                - expected
            )
            <= 0.2
        )

    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_constrained_interval(self, auto_guide_inference):
        beta = dist.Beta(torch.tensor([1.0]), torch.tensor([1.0]))
        interval_rv = bm.random_variable(lambda: beta)
        auto_guide = auto_guide_inference(
            queries=[interval_rv()],
            observations={},
        )
        world = auto_guide.infer(num_steps=100)
        assert (
            abs(
                world.get_guide_distribution(interval_rv()).sample((100,)).mean().item()
                - beta.mean
            )
            <= 0.2
        )

    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_dirichlet(self, auto_guide_inference):
        dirichlet = dist.Dirichlet(2 * torch.ones(2))
        alpha = bm.random_variable(lambda: dirichlet)
        auto_guide = auto_guide_inference([alpha()], {})
        world = auto_guide.infer(num_steps=100)
        map_truth = torch.tensor([0.5, 0.5])
        vi_estimate = world.get_guide_distribution(alpha()).sample((100,)).mean(dim=0)
        assert vi_estimate.isclose(map_truth, atol=0.1).all().item()

    @pytest.mark.parametrize("auto_guide_inference", [ADVI, MAP])
    def test_ppca(self, auto_guide_inference):
        d, D = 2, 4
        n = 150

        @bm.random_variable
        def A():
            return dist.Normal(torch.zeros((D, d)), 2.0 * torch.ones((D, d)))

        @bm.random_variable
        def mu():
            return dist.Normal(torch.zeros(D), 2.0 * torch.ones(D))

        @bm.random_variable
        def z():
            return dist.Normal(torch.zeros(n, d), 1.0)

        @bm.random_variable
        def x():
            return dist.Normal(z() @ A().T + mu(), 1.0)

        vi = auto_guide_inference(
            queries=[A(), mu()],
            observations={x(): torch.random.randn(n, d) * torch.random.randn(d, D)},
        )
        losses = []
        for _ in range(30):
            loss, _ = vi.step()
            losses.append(loss)
        assert losses[-1].item() < losses[0].item()


class TestStochasticVariationalInfer:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        bm.seed(41)

    def test_normal_normal_guide(self):
        normal_normal_model = NormalNormal()
        log_scale_normal_model = LogScaleNormal()

        world = VariationalInfer(
            queries_to_guides={normal_normal_model.mu(): log_scale_normal_model.q_mu()},
            observations={
                normal_normal_model.x(1): torch.tensor(9.0),
                normal_normal_model.x(2): torch.tensor(10.0),
            },
            optimizer=lambda params: torch.optim.Adam(params, lr=1e0),
        ).infer(
            num_steps=100,
        )
        mu_approx = world.get_variable(log_scale_normal_model.q_mu()).distribution

        sample_mean = mu_approx.sample((100, 1)).mean()
        assert sample_mean > 5.0

        sample_var = mu_approx.sample((100, 1)).var()
        assert sample_var > 0.1

    def test_normal_normal_guide_step(self):
        normal_normal_model = NormalNormal()
        log_scale_normal_model = LogScaleNormal()

        # 100 steps, each 1 iteration
        world = VariationalInfer(
            queries_to_guides={
                normal_normal_model.mu(): log_scale_normal_model.q_mu(),
            },
            observations={
                normal_normal_model.x(1): torch.tensor(9.0),
                normal_normal_model.x(2): torch.tensor(10.0),
            },
            optimizer=lambda params: torch.optim.Adam(params, lr=1e0),
        ).infer(num_steps=100)
        mu_approx = world.get_variable(log_scale_normal_model.q_mu()).distribution

        sample_mean = mu_approx.sample((100, 1)).mean()
        assert sample_mean > 5.0

        sample_var = mu_approx.sample((100, 1)).var()
        assert sample_var > 0.1

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

        world = VariationalInfer(
            queries_to_guides={
                mu(): q_mu(),
                alpha(): q_alpha(),
            },
            observations={
                x(1): torch.tensor(9.0),
                x(2): torch.tensor(10.0),
            },
            optimizer=lambda params: torch.optim.Adam(params, lr=1e0),
        ).infer(
            num_steps=100,
        )

        vi = VariationalInfer(
            queries_to_guides={
                mu(): q_mu(),
                alpha(): q_alpha(),
            },
            observations={
                x(1): torch.tensor(9.0),
                x(2): torch.tensor(10.0),
            },
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-1),
        )
        vi.infer(num_steps=100)

        world = VariationalWorld(
            params=vi.params,
            observations={
                **{alpha(): torch.tensor(10.0)},
                **vi.observations,
            },
        )
        mu_approx, _ = world._run_node(q_mu())
        sample_mean_alpha_10 = mu_approx.sample((100, 1)).mean()

        world = VariationalWorld(
            params=vi.params,
            observations={
                **{alpha(): torch.tensor(-10.0)},
                **vi.observations,
            },
        )
        mu_approx, _ = world._run_node(q_mu())
        sample_mean_alpha_neg_10 = mu_approx.sample((100, 1)).mean()

        assert sample_mean_alpha_neg_10 > sample_mean_alpha_10

    def test_discrete_mixture(self):
        model = BinaryGaussianMixture()

        N = 25
        with bm.world.World.initialize_world(
            itertools.chain.from_iterable([model.x(i), model.h(i)] for i in range(N)),
            initialize_fn=init_from_prior,
        ):
            data = torch.tensor([[model.x(i), model.h(i)] for i in range(N)])

        @bm.param
        def phi(i):
            return torch.tensor(0.5, requires_grad=True)

        @bm.random_variable
        def q_h(i):
            return dist.Bernoulli(logits=phi(i))

        vi = VariationalInfer(
            queries_to_guides={model.h(i): q_h(i) for i in range(N)},
            observations={model.x(i): data[i, 0] for i in range(N)},
            optimizer=lambda p: optim.Adam(p, lr=5e-1),
        )
        world = vi.infer(
            num_steps=30, num_samples=50, mc_approx=monte_carlo_approximate_sf
        )

        accuracy = (
            (
                (
                    torch.stack(
                        [
                            world.get_guide_distribution(model.h(i)).probs
                            for i in range(N)
                        ]
                    )
                    > 0.5
                )
                == data[:, 1]
            )
            .float()
            .mean()
        )
        assert accuracy.float().item() > 0.80

    def test_logistic_regression(self):
        n, d = 5_000, 2
        X = torch.randn(n, d)
        W = torch.randn(d)

        @bm.random_variable
        def y():
            return dist.Independent(
                dist.Bernoulli(probs=torch.sigmoid(X @ W)),
                1,
            )

        @bm.param
        def w():
            return torch.randn(d)

        @bm.random_variable
        def q_y():
            weights = w()
            data = X
            p = torch.sigmoid(data @ weights)
            return dist.Independent(
                dist.Bernoulli(probs=p),
                1,
            )

        world = VariationalInfer(
            queries_to_guides={y(): q_y()},
            observations={},
            optimizer=lambda params: torch.optim.Adam(params, lr=3e-2),
        ).infer(
            num_steps=5000,
            num_samples=1,
            # NOTE: since y/q_y are discrete and not reparameterizable, we must
            # use the score function estimator
            mc_approx=monte_carlo_approximate_sf,
        )
        l2_error = (world.get_param(w()) - W).norm()
        assert l2_error < 0.5

    def test_subsample(self):
        # mu ~ N(0, 10) and x | mu ~ N(mu, 1)
        num_total = 3
        normal_normal_model = NormalNormal(mean_0=1, variance_0=100, variance_x=1)
        log_scale_normal_model = LogScaleNormal()

        total_observations = {
            normal_normal_model.x(i): torch.tensor(1.0) for i in range(num_total)
        }

        expected_mean = normal_normal_model.conjugate_posterior(total_observations).mean
        expected_stddev = normal_normal_model.conjugate_posterior(
            total_observations
        ).stddev

        for num_samples in range(1, num_total):
            world = VariationalInfer(
                queries_to_guides={
                    normal_normal_model.mu(): log_scale_normal_model.q_mu(),
                },
                observations={
                    normal_normal_model.x(i): torch.tensor(1.0)
                    for i in range(num_samples)
                },
                optimizer=lambda params: torch.optim.Adam(params, lr=3e-2),
            ).infer(
                num_steps=50, subsample_factor=num_samples / num_total, num_samples=100
            )
            mu_approx = world.get_guide_distribution(normal_normal_model.mu())
            assert (mu_approx.mean - expected_mean).norm() < 0.05
            assert (mu_approx.stddev - expected_stddev).norm() < 0.05

    def test_subsample_fail(self):
        # mu ~ N(0, 10) and x | mu ~ N(mu, 1)
        num_total = 3
        normal_normal_model = NormalNormal(mean_0=1, variance_0=100, variance_x=1)
        log_scale_normal_model = LogScaleNormal()

        total_observations = {
            normal_normal_model.x(i): torch.tensor(1.0) for i in range(num_total)
        }

        expected_mean = normal_normal_model.conjugate_posterior(total_observations).mean
        expected_stddev = normal_normal_model.conjugate_posterior(
            total_observations
        ).stddev

        for num_samples in range(1, num_total):
            world = VariationalInfer(
                queries_to_guides={
                    normal_normal_model.mu(): log_scale_normal_model.q_mu(),
                },
                observations={
                    normal_normal_model.x(i): torch.tensor(1.0)
                    for i in range(num_samples)
                },
                optimizer=lambda params: torch.optim.Adam(params, lr=3e-2),
            ).infer(num_steps=50, subsample_factor=1.0, num_samples=100)
            mu_approx = world.get_guide_distribution(normal_normal_model.mu())
            assert (mu_approx.mean - expected_mean).norm() > 0.05 or (
                mu_approx.stddev - expected_stddev
            ).norm() > 0.05
