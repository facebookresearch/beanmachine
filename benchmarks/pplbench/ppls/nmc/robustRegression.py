# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
from typing import Any, Dict, List, Tuple

import torch
from torch import tensor
from torch.distributions import (
    Bernoulli,
    Exponential,
    Gamma,
    MultivariateNormal,
    Normal,
    StudentT,
)
from tqdm import tqdm

from .utils import gradients, halfspace_proposer, real_proposer


def obtain_posterior(
    data_train: Tuple[Any, Any], args_dict: Dict, model=None
) -> Tuple[List, Dict]:
    """
    Beanmachine impmementation of logisitc regression model.

    :param data_train: tuple of np.ndarray (x_train, y_train)
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    # shape of x_train: (num_features, num_samples)
    x_train, y_train = data_train
    y_train = tensor(y_train, dtype=torch.float32)
    x_train = tensor(x_train, dtype=torch.float32)
    N = int(x_train.shape[1])
    K = int(x_train.shape[0])
    assert y_train.shape == (N,)
    # reshape x_train to (K+1) x N with an extra row 1 at the top
    x_train = torch.cat((torch.ones((1, N)), x_train))

    alpha_scale, beta_scale, beta_loc, sigma_mean = args_dict["model_args"]
    num_samples = args_dict["num_samples_nmc"]

    prior_sigma = Exponential(tensor(1.0 / sigma_mean))
    prior_nu = Gamma(tensor(2.0), tensor(0.1))
    prior_beta = Normal(
        tensor([0.0] + [float(beta_loc)] * K),
        tensor([float(alpha_scale)] + [float(beta_scale)] * K),
    )

    class State:
        def __init__(self, nu=None, sigma=None, beta=None):
            # initialize reals to zero and real+ to 1
            self.nu = tensor(1.0, requires_grad=False) if nu is None else nu
            self.sigma = tensor(1.0, requires_grad=False) if sigma is None else sigma
            self.beta = (
                torch.zeros(K + 1, requires_grad=False) if beta is None else beta
            )

        def clone(self):
            return State(self.nu.clone(), self.sigma.clone(), self.beta.clone())

        def compute_score_(self):
            self.score = prior_sigma.log_prob(self.sigma)
            self.score += prior_nu.log_prob(self.nu)
            self.score += prior_beta.log_prob(self.beta).sum()
            # (K+1)x1 * (K+1)xN => (K+1)xN .sum(0) => N
            mu = (self.beta.unsqueeze(1) * x_train).sum(axis=0)
            self.score += StudentT(self.nu, mu, self.sigma).log_prob(y_train).sum()

        def compute_nu_gradients_(self):
            self.nu.requires_grad_(True)
            self.compute_score_()
            self.nu_grad, = torch.autograd.grad(self.score, self.nu, create_graph=True)
            self.nu_hess, = torch.autograd.grad(self.nu_grad, self.nu)
            self.nu_grad.detach_()
            self.nu_hess.detach_()
            self.nu.requires_grad_(False)

        def compute_nu_proposer_(self):
            # proposer for nu
            self.compute_nu_gradients_()
            self.nu_proposer = halfspace_proposer(self.nu, self.nu_grad, self.nu_hess)

        def compute_sigma_gradients_(self):
            self.sigma.requires_grad_(True)
            self.compute_score_()
            self.sigma_grad, = torch.autograd.grad(
                self.score, self.sigma, create_graph=True
            )
            self.sigma_hess, = torch.autograd.grad(self.sigma_grad, self.sigma)
            self.sigma_grad.detach_()
            self.sigma_hess.detach_()
            self.sigma.requires_grad_(False)

        def compute_sigma_proposer_(self):
            # proposer for sigma
            self.compute_sigma_gradients_()
            self.sigma_proposer = halfspace_proposer(
                self.sigma, self.sigma_grad, self.sigma_hess
            )

        def compute_beta_gradients_(self):
            self.beta.requires_grad_(True)
            self.compute_score_()
            self.beta_grad, self.beta_hess = gradients(self.score, self.beta)
            self.beta_grad.detach_()
            self.beta_hess.detach_()
            self.beta.requires_grad_(False)

        def compute_beta_proposer_(self):
            self.compute_beta_gradients_()
            self.beta_proposer = real_proposer(
                self.beta, self.beta_grad, self.beta_hess
            )

    samples = []
    t1 = time.time()
    theta = State()
    for _i in tqdm(range(num_samples), desc="inference", leave=False):
        # propose nu
        theta.compute_nu_proposer_()
        nu = theta.nu_proposer.sample()
        theta2 = State(nu.clone(), theta.sigma.clone(), theta.beta.clone())
        theta2.compute_nu_proposer_()
        logacc = (
            theta2.score
            - theta.score
            + theta2.nu_proposer.log_prob(theta.nu)
            - theta.nu_proposer.log_prob(theta2.nu)
        )
        if logacc > 0 or Bernoulli(probs=torch.exp(logacc)).sample().item():
            theta = theta2
        # propose sigma
        theta.compute_sigma_proposer_()
        sigma = theta.sigma_proposer.sample()
        theta2 = State(theta.nu.clone(), sigma.clone(), theta.beta.clone())
        theta2.compute_sigma_proposer_()
        logacc = (
            theta2.score
            - theta.score
            + theta2.sigma_proposer.log_prob(theta.sigma)
            - theta.sigma_proposer.log_prob(theta2.sigma)
        )
        if logacc > 0 or Bernoulli(probs=torch.exp(logacc)).sample().item():
            theta = theta2
        # propose beta
        theta.compute_beta_proposer_()
        beta = theta.beta_proposer.sample()
        theta2 = State(theta.nu.clone(), theta.sigma.clone(), beta.clone())
        theta2.compute_beta_proposer_()
        logacc = (
            theta2.score
            - theta.score
            + theta2.beta_proposer.log_prob(theta.beta)
            - theta.beta_proposer.log_prob(theta2.beta)
        )
        if logacc > 0 or Bernoulli(probs=torch.exp(logacc)).sample().item():
            theta = theta2
        samples.append(theta.clone())
    t2 = time.time()
    samples_formatted = []
    for i in tqdm(range(num_samples), desc="collect samples", leave=False):
        theta = samples[i]
        sample_dict = {
            "sigma": theta.sigma.item(),
            "nu": theta.nu.item(),
            "alpha": theta.beta[0].item(),
            "beta": theta.beta[1:].numpy(),
        }
        samples_formatted.append(sample_dict)

    timing_info = {"compile_time": 0, "inference_time": t2 - t1}
    return (samples_formatted, timing_info)
