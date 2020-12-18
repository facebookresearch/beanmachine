# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import scipy.stats
import torch
import torch.distributions as dist
import torch.nn as nn
from beanmachine.ppl.experimental.vi.variational_approximation import (
    VariationalApproximation,
)
from beanmachine.ppl.experimental.vi.variational_infer import (
    MeanFieldVariationalInference,
)
from torch.distributions.utils import _standard_normal


class NealsFunnel(dist.Distribution):
    """
    Neal's funnel.

    p(x,y) = N(y|0,3) N(x|0,exp(y/2))
    """

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


class VariationalInferTest(unittest.TestCase):
    def setUp(self) -> None:
        MeanFieldVariationalInference.set_seed(1)

    class NormalNormal:
        @bm.random_variable
        def mu(self):
            return dist.Normal(0, 10)

        @bm.random_variable
        def x(self, i):
            return dist.Normal(self.mu(), 1)

    def test_neals_funnel(self):
        nf = NealsFunnel()

        vi = VariationalApproximation(
            base_dist=dist.MultivariateNormal,
            base_args={"loc": torch.zeros(2), "covariance_matrix": torch.eye(2)},
        )
        vi.train(target_log_prob=nf.log_prob, epochs=300)

        # compare 1D marginals of empirical distributions using 2-sample K-S test
        nf_samples = nf.sample((20,)).squeeze().numpy()
        vi_samples = vi.sample((20,)).detach().numpy()

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(nf_samples[:, 0], vi_samples[:, 0]).pvalue, 0.05
        )
        self.assertGreaterEqual(
            scipy.stats.ks_2samp(nf_samples[:, 1], vi_samples[:, 1]).pvalue, 0.05
        )

    def test_normal_normal(self):
        model = self.NormalNormal()
        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e-0,
            base_dist=dist.Normal,
            base_args={
                "loc": nn.Parameter(torch.tensor([0.0])),
                "scale": nn.Parameter(torch.tensor([1.0])),
            },
        )

        mu_approx = vi_dicts[model.mu()]
        sample_mean = mu_approx.sample((100, 1)).mean()
        self.assertGreater(sample_mean, 5.0)

        sample_var = mu_approx.sample((100, 1)).var()
        self.assertGreater(sample_var, 0.1)

    def test_normal_normal_studentt_base_dist(self):
        model = self.NormalNormal()
        vi = MeanFieldVariationalInference()
        vi_dicts = vi.infer(
            queries=[model.mu()],
            observations={
                model.x(1): torch.tensor(9.0),
                model.x(2): torch.tensor(10.0),
            },
            num_iter=100,
            lr=1e-0,
            base_dist=dist.StudentT,
            base_args={"df": nn.Parameter(torch.tensor(1.0))},
        )

        mu_approx = vi_dicts[model.mu()]
        sample_mean = mu_approx.sample((100, 1)).mean()
        self.assertGreater(sample_mean, 5.0)

        sample_var = mu_approx.sample((100, 1)).var()
        self.assertGreater(sample_var, 0.1)
