# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle
import time

import numpy as np
import pystan


CODE = """
data {
  // number of observations
  int N;
  // response
  int<lower = 0, upper=1> y[N];
  // number of columns in the design matrix X
  int K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real scale_alpha;
  vector[K] scale_beta;
  vector[K] beta_loc;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
}
transformed parameters {
  vector[N] mu;

  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0.0, scale_alpha);
  beta ~ normal(beta_loc, scale_beta);
  // likelihood
  y ~ bernoulli_logit(mu);
}
"""


def obtain_posterior(data_train, args_dict, model=None):
    """
    Stan impmementation of robust regression model.

    Inputs:
    - data_train(tuple of np.ndarray): x_train, y_train
    - args_dict: a dict of model arguments
    Returns:
    - samples_stan(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    global CODE
    x_train, y_train = data_train
    N = int(x_train.shape[1])
    K = int(x_train.shape[0])
    thinning = args_dict["thinning_stan"]
    alpha_scale, beta_scale, beta_loc, _ = args_dict["model_args"]

    data_stan = {
        "N": N,
        "K": K,
        "X": x_train.T,
        "y": y_train,
        "scale_alpha": alpha_scale,
        "scale_beta": beta_scale * np.ones(K),
        "beta_loc": beta_loc * np.ones(K),
    }

    code_loaded = None
    pkl_filename = os.path.join(args_dict["output_dir"], "stan_logisticRegression.pkl")
    if os.path.isfile(pkl_filename):
        model, code_loaded, elapsed_time_compile_stan = pickle.load(
            open(pkl_filename, "rb")
        )
    if code_loaded != CODE:
        # compile the model, time it
        start_time = time.time()
        model = pystan.StanModel(model_code=CODE, model_name="logistic_regression")
        elapsed_time_compile_stan = time.time() - start_time
        # save it to the file 'model.pkl' for later use
        with open(pkl_filename, "wb") as f:
            pickle.dump((model, CODE, elapsed_time_compile_stan), f)

    if args_dict["inference_type"] == "mcmc":
        # sample the parameter posteriors, time it
        start_time = time.time()
        fit = model.sampling(
            data=data_stan,
            iter=int(args_dict["num_samples_stan"]),
            chains=1,
            thin=thinning,
            check_hmc_diagnostics=False,
        )
        samples_stan = fit.extract(
            pars=["alpha", "beta"], permuted=False, inc_warmup=True
        )
        elapsed_time_sample_stan = time.time() - start_time

    elif args_dict["inference_type"] == "vi":
        # sample the parameter posteriors, time it
        start_time = time.time()
        fit = model.vb(data=data_stan, iter=args_dict["num_samples_stan"])
        samples_stan = fit.extract(
            pars=["alpha", "beta"], permuted=False, inc_warmup=True
        )
        elapsed_time_sample_stan = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(int(args_dict["num_samples_stan"] / args_dict["thinning_stan"])):
        sample_dict = {}
        for parameter in samples_stan.keys():
            sample_dict[parameter] = samples_stan[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_stan,
        "inference_time": elapsed_time_sample_stan,
    }

    return (samples, timing_info)
