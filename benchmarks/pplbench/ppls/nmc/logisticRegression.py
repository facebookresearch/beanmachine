# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import tensor
from torch.distributions import Bernoulli, MultivariateNormal, Normal

from .utils import gradients


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
    y_train = np.array(y_train, dtype=np.float32)
    x_train = np.array(x_train, dtype=np.float32)
    x_train = tensor(x_train)
    y_train = tensor([tensor(y) for y in y_train])
    N = int(x_train.shape[1])
    K = int(x_train.shape[0])
    assert y_train.shape == (N,)

    alpha_scale = float((args_dict["model_args"])[0])
    beta_scale = float((args_dict["model_args"])[1])
    beta_loc = float((args_dict["model_args"])[2])
    num_samples = args_dict["num_samples_nmc"]

    # x_train has shape (K+1) x N
    x_train = torch.cat((torch.ones((1, N)), x_train))
    # we construct the prior to have mean and std with shape (K+1)
    prior = Normal(
        tensor([0.0] + ([beta_loc] * K)), tensor([alpha_scale] + ([beta_scale] * K))
    )

    def log_joint(theta):
        score = prior.log_prob(theta).sum()
        mu = torch.mm(theta.unsqueeze(dim=0), x_train).squeeze(dim=0)
        score += Bernoulli(logits=mu).log_prob(y_train).sum()
        return score

    def get_score_and_proposer(theta):
        theta.requires_grad_(True)
        score = log_joint(theta)
        grad, hess = gradients(score, theta)
        grad.detach_()
        hess.detach_()
        theta.requires_grad_(False)
        neg_hess_inv = torch.inverse(-hess)
        mu = theta + torch.mm(neg_hess_inv, grad.unsqueeze(1)).squeeze(1)
        return score, MultivariateNormal(mu, neg_hess_inv)

    # initialize to zero
    samples = []
    theta = torch.zeros(K + 1, requires_grad=True)
    score, proposer = get_score_and_proposer(theta)
    t1 = time.time()
    for _i in range(num_samples):
        theta2 = proposer.sample()
        score2, proposer2 = get_score_and_proposer(theta2)
        logacc = (
            score2
            - score
            + proposer2.log_prob(theta).sum()
            - proposer.log_prob(theta2).sum()
        )
        if logacc > 0 or Bernoulli(probs=torch.exp(logacc)).sample().item():
            theta, score, proposer = theta2, score2, proposer2
        samples.append(theta.clone())
    t2 = time.time()
    samples_formatted = []
    for i in range(num_samples):
        sample_dict = {
            "alpha": samples[i][0].item(),
            "beta": samples[i][1:].detach().numpy(),
        }
        samples_formatted.append(sample_dict)

    timing_info = {"compile_time": 0, "inference_time": t2 - t1}
    return (samples_formatted, timing_info)
