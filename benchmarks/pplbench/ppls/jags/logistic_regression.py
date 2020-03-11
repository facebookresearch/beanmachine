# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pyjags
from ppls.pplbench_ppl import PPLBenchPPL


CODE = """
# Classification model
model {
  # priors
  alpha ~ dnorm(0.0, 1/(scale_alpha**2));
  for (k in 1:K) {
  beta[k] ~ dnorm(0.0, 1/(scale_beta[k]**2));
  }
  # likelihood
  for (n in 1:N) {
  y[n] ~ dbern(mean[n])
  logit(mean[n]) <- alpha + inprod(beta, X[,n])
  }
}
"""


class LogisticRegression(PPLBenchPPL):
    def obtain_posterior(self, data_train, args_dict, model=None):
        """
        Jags impmementation of logistic regression model.

        Inputs:
        - data_train(tuple of np.ndarray): x_train, y_train
        - args_dict: a dict of model arguments
        Returns:
        - samples_jags(dict): posterior samples of all parameters
        - timing_info(dict): compile_time, inference_time
        """
        global CODE
        x_train, y_train = data_train
        N = int(x_train.shape[1])
        K = int(x_train.shape[0])
        alpha_scale = (args_dict["model_args"])[0]
        beta_scale = (args_dict["model_args"])[1]

        data_jags = {
            "N": N,
            "K": K,
            "X": x_train,
            "y": y_train,
            "scale_alpha": alpha_scale,
            "scale_beta": beta_scale * np.ones(K),
        }

        # compile the model, time it
        start_time = time.time()
        brmodel = pyjags.Model(CODE, data=data_jags, chains=1, adapt=0)
        elapsed_time_compile_jags = time.time() - start_time

        if args_dict["inference_type"] == "mcmc":
            # sample the parameter posteriors, time it
            start_time = time.time()
            # Choose the parameters to watch and iterations:
            samples_jags = brmodel.sample(
                int(args_dict["num_samples_jags"]), vars=["alpha", "beta"]
            )
            elapsed_time_sample_jags = time.time() - start_time
        elif args_dict["inference_type"] == "vi":
            print("Jags does not support Variational Inference")
            exit()

        # repackage samples into shape required by PPLBench
        samples = []
        # move axes to facilitate iterating over samples
        # change (sample, chain, values) to (chain, sample, value)
        samples_jags["beta"] = np.moveaxis(samples_jags["beta"], [0, 1, 2], [1, 0, 2])
        for parameter in samples_jags.keys():
            if samples_jags[parameter].shape[0] == 1:
                samples_jags[parameter] = samples_jags[parameter].squeeze()
        for i in range(int(args_dict["num_samples_jags"])):
            sample_dict = {}
            for parameter in samples_jags.keys():
                sample_dict[parameter] = samples_jags[parameter][i].T
            samples.append(sample_dict)
        timing_info = {
            "compile_time": elapsed_time_compile_jags,
            "inference_time": elapsed_time_sample_jags,
        }

        return (samples, timing_info)
