import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist


class IntegrationTest(unittest.TestCase):
    class LogisticRegressionModel(object):
        @bm.random_variable
        def theta_0(self):
            return dist.Normal(0.0, 1.0)

        @bm.random_variable
        def theta_1(self):
            return dist.Normal(0.0, torch.ones(3))

        @bm.random_variable
        def y(self, X):
            logits = (X * self.theta_1() + self.theta_0()).sum(-1)
            return dist.Bernoulli(logits=logits)

    def test_logistic_regression(self):
        torch.manual_seed(1)
        true_coefs = torch.tensor([1.0, 2.0, 3.0])
        true_intercept = torch.tensor(1.0)
        X = torch.randn(3000, 3)
        Y = dist.Bernoulli(logits=(X * true_coefs + true_intercept).sum(-1)).sample()
        model = self.LogisticRegressionModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        samples_nw = nw.infer(
            queries=[model.theta_1(), model.theta_0()],
            observations={model.y(X): Y},
            num_samples=1000,
            num_chains=1,
        )
        coefs_mean = samples_nw[model.theta_1()].view(-1, 3).mean(0)
        intercept_mean = samples_nw[model.theta_0()].view(-1).mean(0)
        self.assertTrue(torch.isclose(coefs_mean, true_coefs, atol=0.15).all())
        self.assertTrue(torch.isclose(intercept_mean, true_intercept, atol=0.15).all())
