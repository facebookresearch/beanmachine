# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Module for Robust Regression Model

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Model Specification:
Robust regression model with independent student-t errors. Less sensitive to
outliers than regression models with normal errors.

y ~ Student-T(nu, mean, sigma)

nu = degrees of freedom (as nu->inf, distribution resembles normal)
nu ~ Gamma(shape=2, scale=10)

mean = alpha + beta * x; mean of student_t distribution
alpha ~ normal(0, some_var)
beta ~ normal(0, some_var)
x ~ normal(0, 10)

sigma = variance of student_t distribution
sigma ~ exponential(some_mean)

Model specific arguments:
Pass these arguments in following order -
[alpha_scale, beta_scale, beta_loc, sigma_loc]
"""

import numpy as np
from scipy import stats


def get_defaults():
    defaults = {
        "n": 2000,
        "k": 10,
        "runtime": 200,
        "train_test_ratio": 0.5,
        "trials": 10,
        "model_args": [10, 2.5, 0, 1.0],
    }
    return defaults


def generate_model(args_dict):
    return None


def generate_data(args_dict, model=None):
    """
    Generate data for robust regression model.
    Inputs:
    - N(int) = number of data points to generate
    - K(int) = number of covariates in the regression model

    Returns:
    - generated_data(dict) = x_train, y_train, x_test, y_test
    """
    print("Generating data")

    np.random.seed(args_dict["rng_seed"])

    N = int(args_dict["n"])
    K = int(args_dict["k"])
    train_test_ratio = float(args_dict["train_test_ratio"])

    # parameters for distributions to sample parameters from
    alpha_scale = (args_dict["model_args"])[0]
    beta_scale = (args_dict["model_args"])[1]
    beta_locs = (args_dict["model_args"])[2] * np.ones(K).reshape([-1, 1])
    sigma_loc = (args_dict["model_args"])[3]

    # sampling parameters from tails (10-11 sigma)
    alpha = stats.truncnorm.rvs(
        10, 11, loc=0, scale=alpha_scale, size=N
    ) * np.random.choice([1, -1], size=N)
    beta = stats.truncnorm.rvs(
        10, 11, loc=beta_locs, scale=beta_scale, size=(K, N)
    ) * np.random.choice([1, -1], size=(K, N))
    nu = np.random.gamma(shape=2, scale=10, size=N)
    sigma = stats.truncexpon.rvs(10, loc=sigma_loc, size=N)

    # model
    x = np.random.normal(0, 10, size=(K, N))

    y = stats.t.rvs(df=nu, loc=(alpha + np.sum(beta * x, axis=0)), scale=sigma)

    # Split in train/test
    split = int(N * train_test_ratio)
    x_train, x_test = x[:, :split], x[:, split:]
    y_train, y_test = y[:split], y[split:]

    return {"data_train": (x_train, y_train), "data_test": (x_test, y_test)}


def evaluate_posterior_predictive(samples, data_test, model=None):
    """
    Computes the likelihood of held-out testset wrt parameter samples

    input-
    samples(dict): parameter samples from model
    data_test: test data
    model: modelobject

    returns- pred_log_lik_array(np.ndarray): log-likelihoods of data
            wrt parameter samples
    """
    x_test, y_test = data_test
    pred_log_lik_array = []
    for sample in samples:
        loc = (sample["alpha"] + np.dot(sample["beta"], x_test)).reshape(-1)
        log_lik_test = stats.t.logpdf(
            y_test, df=sample["nu"], loc=loc, scale=sample["sigma"]
        )
        pred_log_lik_array.append(log_lik_test)
    # return as a numpy array of mean over test data
    return np.mean(pred_log_lik_array, axis=1)
