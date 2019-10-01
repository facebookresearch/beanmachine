# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pymc3 as pm


def obtain_posterior(data_train, args_dict, model=None):
    """
    PyMC3 implementation of robust regression model.

    Inputs:
    - data_train(tuple of np.ndarray): x_train, y_train
    - args_dict: a dict of model arguments
    Returns:
    - samples(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    x_train, y_train = data_train
    N = int(x_train.shape[1])
    K = int(x_train.shape[0])
    thinning = args_dict["thinning_pymc3"]
    alpha_scale, beta_scale, beta_loc, sigma_loc = args_dict["model_args"]
    num_samples = int(args_dict["num_samples_pymc3"])

    # Define model and sample
    if args_dict["inference_type"] == "mcmc":
        start_time = time.time()
        with pm.Model() as robust_regression:
            alpha = pm.Normal("alpha", mu=0, sigma=alpha_scale)
            beta = pm.Normal("beta", mu=beta_loc, sigma=beta_scale, shape=K)
            sigma = pm.Exponential("sigma", lam=sigma_loc)
            nu = pm.Gamma("nu", alpha=2, beta=10)
            mean = (alpha + x_train.T * beta).T
            y_observed = pm.StudentT(
                "y_observed", nu=nu, mu=mean, sigma=sigma, observed=y_train
            )
            elapsed_time_compile_pymc3 = time.time() - start_time
            start_time = time.time()
            samples_pymc3 = pm.sample(
                num_samples, cores=1, chains=1, discard_tuned_samples=False
            )

    elif args_dict["inference_type"] == "vi":
        raise NotImplementedError

    elapsed_time_sample_pymc3 = time.time() - start_time
    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(0, int(args_dict["num_samples_pymc3"]), args_dict["thinning_pymc3"]):
        sample_dict = {}
        for parameter in ["alpha", "beta", "nu", "sigma"]:
            sample_dict[parameter] = samples_pymc3[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_pymc3,
        "inference_time": elapsed_time_sample_pymc3,
    }

    return (samples, timing_info)
