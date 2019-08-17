# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC


# define a generic regression model
class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * weight + bias
        return self.linear(x)


# create a robust regression model
def robust_model(x_train, y_train, args_dict):
    K = int(x_train.shape[1])
    N = int(x_train.shape[0])
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    scale_beta, scale_alpha, loc_beta, sigma_prior = args_dict["model_args"]
    # Create priors over the parameters
    w_prior = dist.Normal(
        loc_beta * torch.ones(1, K), torch.ones(1, K) * scale_beta
    ).to_event(1)
    b_prior = dist.Normal(torch.zeros(1), torch.ones(1) * scale_alpha).to_event(1)
    priors = {"linear.weight": w_prior, "linear.bias": b_prior}
    # lift module parameters to random variables sampled from the priors
    regression_model = RegressionModel(p=K)
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    df = pyro.sample("nu", dist.Gamma(2 * torch.ones(1), 0.1 * torch.ones(1)))
    sigma = pyro.sample("sigma", dist.Exponential(sigma_prior * torch.ones(1)))
    with pyro.plate("map", N):
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_train).squeeze(-1)
        pyro.sample("obs", dist.StudentT(df, prediction_mean, sigma), obs=y_train)


def obtain_posterior(data_train, args_dict, model=None):
    """
    Pyro impmementation of robust regression model.

    Inputs:
    - data_train(tuple of np.ndarray): x_train, y_train
    - args_dict: a dict of model arguments
    Returns:
    - samples_jags(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    assert pyro.__version__.startswith("0.3.4")
    x_train, y_train = data_train
    if args_dict["inference_type"] == "mcmc":
        start_time = time.time()
        nuts_kernel = NUTS(model=robust_model, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=args_dict["num_samples_pyro"])
        mcmc.run(x_train.T, y_train, args_dict)
        samples_pyro = mcmc.get_samples()
    elif args_dict["inference_type"] == "vi":
        print("ImplementationError; exiting...")
        exit(1)

    elapsed_time_sample_pyro = time.time() - start_time
    samples_pyro["beta"] = samples_pyro["module$$$linear.weight"].numpy()
    samples_pyro["alpha"] = samples_pyro["module$$$linear.bias"].numpy()
    samples_pyro["nu"] = samples_pyro["nu"].numpy().T
    samples_pyro["sigma"] = samples_pyro["sigma"].numpy().T
    del samples_pyro["module$$$linear.weight"]
    del samples_pyro["module$$$linear.bias"]

    # repackage samples into shape required by PPLBench
    samples = []
    # move axes to facilitate iterating over samples
    # change [sample, chain, values] to [chain, sample, value]
    samples_pyro["beta"] = np.moveaxis(samples_pyro["beta"], [0, 1, 2], [1, 0, 2])
    for parameter in samples_pyro.keys():
        if samples_pyro[parameter].shape[0] == 1:
            samples_pyro[parameter] = samples_pyro[parameter].squeeze()
    for i in range(int(args_dict["num_samples_pyro"] / args_dict["thinning_pyro"])):
        sample_dict = {}
        for parameter in samples_pyro.keys():
            sample_dict[parameter] = samples_pyro[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": 0,  # no compiliation for pyro
        "inference_time": elapsed_time_sample_pyro,
    }
    return (samples, timing_info)
