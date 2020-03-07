# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle
import time

import pystan


CODE = """
// Linear Model with Student-t Errors
data {
  // number of observations
  int N;
  // response
  vector[N] y;
  // number of columns in the design matrix X
  int K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real scale_alpha;
  real loc_beta;
  real scale_beta;
  real rate_sigma;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
  real sigma;
  // degrees of freedom;
  // limit df = 2 so that there is a finite variance
  real nu;
}
transformed parameters {
  vector[N] mu;

  mu = alpha + X * beta;
}
model {
  // priors
  alpha ~ normal(0.0, scale_alpha);
  beta ~ normal(loc_beta, scale_beta);
  sigma ~ exponential(rate_sigma);
  // see Stan prior distribution suggestions
  nu ~ gamma(2, 0.1);
  // likelihood
  y ~ student_t(nu, mu, sigma);
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
    alpha_scale, beta_scale, beta_loc, sigma_mean = args_dict["model_args"]

    data_stan = {
        "N": N,
        "K": K,
        "X": x_train.T,
        "y": y_train,
        "scale_alpha": alpha_scale,
        "scale_beta": beta_scale,
        "loc_beta": beta_loc,
        "rate_sigma": 1.0 / sigma_mean,
    }

    code_loaded = None
    pkl_filename = os.path.join(args_dict["output_dir"], "stan_robustRegression.pkl")
    if os.path.isfile(pkl_filename):
        model, code_loaded, elapsed_time_compile_stan = pickle.load(
            open(pkl_filename, "rb")
        )
    if code_loaded != CODE:
        # compile the model, time it
        start_time = time.time()
        model = pystan.StanModel(model_code=CODE, model_name="robust_regression")
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
            check_hmc_diagnostics=False,
        )
        samples_stan = fit.extract(
            pars=["alpha", "beta", "sigma", "nu"], permuted=False, inc_warmup=True
        )
        elapsed_time_sample_stan = time.time() - start_time

    elif args_dict["inference_type"] == "vi":
        # sample the parameter posteriors, time it
        start_time = time.time()
        fit = model.vb(data=data_stan, iter=args_dict["num_samples_stan"])
        samples_stan = fit.extract(
            pars=["alpha", "beta", "sigma", "nu"], permuted=False, inc_warmup=True
        )
        elapsed_time_sample_stan = time.time() - start_time

    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(int(args_dict["num_samples_stan"])):
        sample_dict = {}
        for parameter in samples_stan.keys():
            sample_dict[parameter] = samples_stan[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_stan,
        "inference_time": elapsed_time_sample_stan,
    }

    return (samples, timing_info)
