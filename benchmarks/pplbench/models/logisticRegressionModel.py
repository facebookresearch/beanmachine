# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Module for Logistic Regression

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Model Specification:
Binary Classification model.

Y ~ Bernoulli(p)

p = probability that x belongs to class 1
p = sigmoid(mu)

mu = alpha + beta*X

X ~ normal(0, 10)
alpha ~ Normal(0, alpha_scale)
beta ~ Normal(beta_loc, beta_scale)

Model specific arguments:
Pass these arguments in following order -
[alpha_scale, beta_scale, beta_loc]
"""

import numpy as np
import torch
from scipy import special, stats


def get_defaults():
    defaults = {
        "n": 2000,
        "k": 10,
        "runtime": 200,
        "train_test_ratio": 0.5,
        "iterations": 10,
        "model_args": [10, 2.5, 0],
    }
    return defaults


def generate_model(args_dict):
    return None


def generate_data(args_dict, model=None):
    """
    Generate data for logistic regression model.
    Inputs:
    - N(int) = number of data points to generate
    - K(int) = number of covariates in the logistic regression model

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

    # sampling parameters from tails (10-11 sigma)
    alpha = stats.truncnorm.rvs(
        10, 11, loc=0, scale=alpha_scale, size=N
    ) * np.random.choice([1, -1], size=N)
    beta = stats.truncnorm.rvs(
        10, 11, loc=beta_locs, scale=beta_scale, size=(K, N)
    ) * np.random.choice([1, -1], size=(K, N))

    # model
    x = np.random.normal(0, 10, size=(K, N))
    mu = alpha + np.sum(beta * x, axis=0)
    y = torch.distributions.Bernoulli(logits=torch.tensor(mu)).sample().numpy()
    y = y.astype(dtype=np.int32)

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
    y_test = torch.tensor(np.array(y_test, dtype=np.float32))
    pred_log_lik_array = []
    for sample in samples:
        mu = torch.tensor(
            (sample["alpha"] + np.dot(sample["beta"], x_test)).reshape(-1)
        )
        log_lik_test = torch.distributions.Bernoulli(logits=mu).log_prob(y_test)
        pred_log_lik_array.append(log_lik_test.numpy())
    # return as a numpy array of sum of log likelihood over test data
    return np.sum(pred_log_lik_array, axis=1)
