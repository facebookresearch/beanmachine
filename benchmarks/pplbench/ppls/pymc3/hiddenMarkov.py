# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pymc3 as pm
import torch
from numpy import array


# from torch import tensor


def obtain_posterior(data_train, args_dict, model=None):
    """
    PyMC3 impmementation of HMM prediction.

    :param data_train:
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    # true_theta = model["theta"]
    # true_theta = torch.stack(true_theta).numpy()

    N = int(args_dict["n"])
    K = int(args_dict["k"])
    concentration, mu_loc, mu_scale, sigma_shape, sigma_rate, observe_model = list(
        map(float, args_dict["model_args"])
    )
    num_samples = int(args_dict["num_samples"])
    observations = data_train
    observations = torch.stack(observations).numpy()

    start_time = time.time()

    Xs = {}
    Ys = {}
    with pm.Model():
        if observe_model:
            theta = pm.Dirichlet(
                "theta",
                a=np.ones(K) * concentration / K,
                shape=(K, K),
                observed=model["theta"].numpy(),
            )
            mus = pm.Normal(
                "mus", mu_loc, mu_scale, shape=(K,), observed=model["mus"].numpy()
            )
            sigmas = pm.Gamma(
                "sigmas",
                sigma_shape,
                sigma_rate,
                shape=(K,),
                observed=model["sigmas"].numpy(),
            )
        else:
            theta = pm.Dirichlet(
                "theta", a=np.ones(K) * concentration / K, shape=(K, K)
            )
            mus = pm.Normal("mus", mu_loc, mu_scale, shape=(K,))
            sigmas = pm.Gamma("sigmas", sigma_shape, sigma_rate, shape=(K,))

        Xs[0] = pm.Categorical("X[0]", p=array([1] + [0] * (K - 1)), observed=0)
        Ys[0] = pm.Normal(
            "Y[0]", mu=mus[Xs[0]], sigma=sigmas[Xs[0]], observed=observations[0]
        )
        for i in range(1, N):
            Xs[i] = pm.Categorical("X[" + str(i) + "]", p=theta[Xs[i - 1]])
            Ys[i] = pm.Normal(
                "Y[" + str(i) + "]",
                mu=mus[Xs[i]],
                sigma=sigmas[Xs[i]],
                observed=observations[i],
            )

        elapsed_time_compile_pymc3 = time.time() - start_time
        start_time = time.time()
        samples_pymc3 = pm.sample(
            draws=0,
            chains=1,
            cores=1,
            tune=num_samples,
            discard_tuned_samples=False,
            compute_convergence_checks=False,
        )

    elapsed_time_sample_pymc3 = time.time() - start_time

    # repackage samples into shape required by PPLBench
    xn1str = "X[" + str(N - 1) + "]"
    samples = []
    for i in range(num_samples):
        result = {}
        if not observe_model:
            result["theta"] = samples_pymc3["theta"][i]
            result["mus"] = samples_pymc3["mus"][i]
            result["sigmas"] = samples_pymc3["sigmas"][i]
        result[xn1str] = samples_pymc3["X[" + str(N - 1) + "]"][i]
        samples.append(result)

    timing_info = {
        "compile_time": elapsed_time_compile_pymc3,
        "inference_time": elapsed_time_sample_pymc3,
    }

    return (samples, timing_info)
