# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for tutorial on Robust Linear Regression"""

# This file is a manual replica of the Bento tutorial with the same name
### TODO: The disabled test produces the following error:
# E       TypeError: Distribution 'HalfNormal' is not supported by Bean Machine Graph.
# This error should be removed for OSS readiness.

### TODO: This tutorial has a couple of different calls to inference, and currently only the
### first call is being considered. It would be good to go through the other parts as well

import logging
import unittest

# TODO: Check imports for conistency

import beanmachine.ppl as bm
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.inference.bmg_inference import BMGInference
from sklearn import model_selection
from torch import tensor

# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(12)

# Model


@bm.random_variable
def beta():
    """
    Regression Coefficient
    """
    return dist.Normal(0, 1000)


@bm.random_variable
def alpha():
    """
    Regression Bias/Offset
    """
    return dist.Normal(0, 1000)


@bm.random_variable
def sigma_regressor():
    """
    Deviation parameter for Student's T
    Controls the magnitude of the errors.
    """
    return dist.HalfNormal(1000)


@bm.random_variable
def df_nu():
    """
    Degrees of Freedom of a Student's T
    Check https://en.wikipedia.org/wiki/Student%27s_t-distribution for effect
    """
    return dist.Gamma(2, 0.1)


@bm.random_variable
def y_robust(X):
    """
    Heavy-Tailed Noise model for regression utilizing StudentT
    Student's T : https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    return dist.StudentT(df=df_nu(), loc=beta() * X + alpha(), scale=sigma_regressor())


# Creating sample data

sigma_data = torch.tensor([20, 40])
rho = -0.95
N = 200

cov = torch.tensor(
    [
        [torch.pow(sigma_data[0], 2), sigma_data[0] * sigma_data[1] * rho],
        [sigma_data[0] * sigma_data[1] * rho, torch.pow(sigma_data[1], 2)],
    ]
)

dist_clean = dist.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=cov)
points = tensor([dist_clean.sample().tolist() for i in range(N)]).view(N, 2)
X = X_clean = points[:, 0]
Y = Y_clean = points[:, 1]

true_beta_1 = 2.0
true_beta_0 = 5.0
true_epsilon = 1.0

points_noisy = points
points_noisy[0, :] = torch.tensor([-20, -80])
points_noisy[1, :] = torch.tensor([20, 100])
points_noisy[2, :] = torch.tensor([40, 40])
X_corr = points_noisy[:, 0]
Y_corr = points_noisy[:, 1]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)

# Inference parameters

num_samples = (
    2  ###000 - Sample size reduced since it should not affect compilation issues
)
num_chains = 4

observations = {y_robust(X_train): Y_train}

queries = [beta(), alpha(), sigma_regressor(), df_nu()]

### The following is old code


class tutorialRobustLinearRegresionTest(unittest.TestCase):
    def test_tutorial_Robust_Linear_Regression(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        # Note: No explicit seed here (in original tutorial model). Should we add one?
        amh = bm.SingleSiteAncestralMetropolisHastings()  # Added local binding
        _ = amh.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        self.assertTrue(True, msg="We just want to check this point is reached")

    def disabled_test_tutorial_Robust_Linear_Regression_to_dot_cpp_python(
        self,
    ) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())
