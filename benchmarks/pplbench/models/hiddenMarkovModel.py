# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import scipy.special
import torch
import torch.distributions as dist
from torch import tensor


"""
Module for Hidden Markov Model

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Specification:
This is an HMM with discrete hidden state space that evolves
according to some transition matrix,
and has observable emissions with Gaussian noise.

Length of sequence of observed timesteps: N
Size of state space of hidden state: K

Y_i | X_i ~ Normal(mus[X_i], sigmas[X_i])

mus[j] ~ Normal(mu_loc, mu_scale)
sigmas[j] ~ Gamma(sigma_shape, sigma_scale)

X_{i+1} | X_i ~ M[X_i]
M[j] ~ Dirichlet(torch.ones(K) * concentration / K)

X_0 = 0

Model specific arguments:
Pass these arguments in following order -
[concentration, mu_loc, mu_scale, sigma_shape, sigma_scale]
"""


def get_defaults():
    defaults = {
        "n": 60,
        "k": 12,
        "train_test_ratio": 0.5,
        "runtime": 80,
        "trials": 10,
        "model_args": [0.1, 1.0, 5.0, 3.0, 3.0],
    }
    defaults["rng_seed"] = int(np.random.random() * 1000.0)
    return defaults


# Generate Samples Helpers


def generate_hiddens(model, N):
    """
    Given the generated model, sample a sequence of values for X_0 ... X_{N-1}
    """
    theta = model["theta"]
    distbns = {j: dist.Categorical(v) for j, v in enumerate(theta)}

    current = torch.tensor(0)
    chain = [current.item()]
    for _ in range(N - 1):
        current = distbns[current.item()].sample()
        chain.append(current.item())
    return chain


def observe_hiddens(model, hiddens):
    """
    Given the generated model, and a sequence of values for X_0 ... X_{N-1},
    sample a sequence of values for Y_0 ... Y_{N-1},
    """
    mus = model["mus"]
    sigmas = model["sigmas"]

    def observe(val):
        return dist.Normal(mus[val], sigmas[val]).sample()

    return list(map(observe, hiddens))


# PPLBench API


def generate_model(args_dict):
    """
    Given model args, sample a valid instance of our model. Consists of:
    - theta : Transition matrix for hidden states
    - mu_j , sigma_j : Parameters of emission distribution for hidden state X_j
    """

    K = int(args_dict["k"])
    concentration, mu_loc, mu_scale, sigma_shape, sigma_scale = list(
        map(float, args_dict["model_args"])
    )

    alpha = torch.ones(K) * concentration / K
    theta = dist.Dirichlet(alpha).sample((K,))
    mus = dist.Normal(mu_loc, mu_scale).sample((K,))
    sigmas = dist.Gamma(sigma_shape, sigma_scale).sample((K,))
    return {"theta": theta, "mus": mus, "sigmas": sigmas, "K": K}


def generate_data(args_dict, model=None):

    """
    Generate data for hidden Markov model.
    Inputs:
    - N(int) = number of data points to generate
    - K(int) = length of sequence

    Returns:
    - generated_data(dict) = samples of the hidden state value trajectories
    """

    print("Generating data")
    N = int(args_dict["n"])
    train_test_ratio = float(args_dict["train_test_ratio"])
    theta = model["theta"]
    mus = model["mus"]
    sigmas = model["sigmas"]

    # Train data
    hiddens = generate_hiddens(model, N)
    observs = observe_hiddens(model, hiddens)
    train_data = observs

    # Test data
    xk = hiddens[-1]
    # want many samples of obs Y(K+1)
    dctxk = dist.Categorical(theta[xk])
    xk1samples = dctxk.sample((int(train_test_ratio * N),))
    yk1samples = [dist.Normal(mus[xk1], sigmas[xk1]).sample() for xk1 in xk1samples]
    test_data = yk1samples

    return {"data_train": train_data, "data_test": test_data}


def evaluate_posterior_predictive(samples, data_test, model=None):
    """
    Computes the likelihood of held-out testset wrt parameter samples
    The testset is values of Y(self.N) simulated from the true value of X(self.N - 1)
    The parameter samples are samples of X(self.N - 1)

    input-
    samples(dict): samples, inferred by PPL, of X(self.N - 1)
    data_test: simulated values of Y(self.N), based on the true value of X(self.N - 1)
    model: args_dict

    returns- pred_log_lik_array(np.ndarray): log-likelihoods of data
            wrt parameter samples
    """

    K = model["K"]

    # Log-prob of observing (Y(N+1) = yn1 | X(N+1) ~ hidden_transition_pmf)
    # hidden_transition_pmf is determined by X(N) (outside of this fnc.'s scope)
    # Compute by summing over possible values of X(N+1).
    def log_prob_obs(yn1, hidden_transition_pmf, mus, sigmas):

        ls = [
            dist.Normal(mus[xn1], sigmas[xn1]).log_prob(tensor(float(yn1)))
            + hidden_transition_pmf.log_prob(tensor(float(xn1)))
            for xn1 in range(K)
        ]
        return scipy.special.logsumexp(ls)

    pred_log_lik_array = []
    for sample in samples:
        # transition likelihood:
        theta = sample["theta"]
        mus = sample["mus"]
        sigmas = sample["sigmas"]
        xn = sample["x[N]"]

        hidden_transition_pmf = dist.Categorical(torch.from_numpy(theta[int(xn)]))
        pred_log_lik_array.append(
            np.sum(
                log_prob_obs(yn1, hidden_transition_pmf, mus, sigmas)
                for yn1 in data_test
            )
        )

    # return as a numpy array of sum of log likelihood over test data
    return pred_log_lik_array
