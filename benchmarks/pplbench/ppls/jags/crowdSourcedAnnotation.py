# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pyjags


CODE = """
data {
  # priors
  for (k in 1:n_categories) {
    beta[k] <- 1.0/n_categories;
  }

  for (k1 in 1:n_categories) {
    for (k2 in 1:n_categories) {
      alpha[k1,k2] <- ifelse(k1==k2, 0.5, 0.5/(n_categories-1))
    }
  }
}
model {
  # sample from priors
  pi ~ ddirch(beta);
  for (i in 1:n_items) {
      z[i] ~ dcat(pi)
    }
  for (j in 1:n_labelers) {
    for (k in 1:n_categories) {
      theta[j,k,1:n_categories] ~ ddirch(alpha[k,1:n_categories]);
    }
  }
  # likelihoods
  for (i in 1:n_obs){
    labels[i] ~ dcat(theta[labeler[i],z[item[i]],1:n_categories]);
  }
}
"""


def obtain_posterior(data_train, args_dict, model=None):
    """
    Jags impmementation of robust regression model.

    Inputs:
    - data_train(tuple of np.ndarray): vector_y, vector_J_i, num_labels
    - args_dict: a dict of model arguments
    Returns:
    - samples_jags(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    global CODE
    vector_y, vector_J_i, num_labels = data_train
    J = int(args_dict["k"])
    n_items = len(num_labels)
    thinning = args_dict["thinning_jags"]
    K = (args_dict["model_args"])[0]
    # item array for jags; change from list-of-lengths format of num_labels
    # to list-of-items associated with each label and labeler in vector_y, vector_J_i
    item = []
    for i in range(len(num_labels)):
        item.extend(np.repeat(i + 1, num_labels[i]))
    data_jags = {
        "n_obs": len(vector_y),
        "n_items": n_items,
        "n_labelers": J,
        "n_categories": K,
        "labels": [i + 1 for i in vector_y],  # 0-indexed to 1-indexed
        "labeler": [i + 1 for i in vector_J_i],  # 0-indexed to 1-indexed
        "item": item,
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
            int(args_dict["num_samples_jags"]), vars=["theta", "pi"], thin=thinning
        )
        elapsed_time_sample_jags = time.time() - start_time
    elif args_dict["inference_type"] == "vi":
        print("Jags does not support Variational Inference")
        exit(1)
    # repackage samples into shape required by PPLBench
    samples = []
    # move axes to facilitate iterating over samples
    # change (sample, chain, values) to (chain, sample, value)
    samples_jags["pi"] = np.moveaxis(samples_jags["pi"], [0, 1, 2], [2, 1, 0])
    # change from (J, K, K, samples, chains) to (chains, samples, J, K, K)
    samples_jags["theta"] = np.moveaxis(samples_jags["theta"], [3, 4], [1, 0])
    for parameter in samples_jags.keys():
        if samples_jags[parameter].shape[0] == 1:
            samples_jags[parameter] = samples_jags[parameter].squeeze()
    for i in range(int(args_dict["num_samples_jags"] / float(thinning))):
        sample_dict = {}
        for parameter in samples_jags.keys():
            sample_dict[parameter] = samples_jags[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_jags,
        "inference_time": elapsed_time_sample_jags,
    }

    return (samples, timing_info)
