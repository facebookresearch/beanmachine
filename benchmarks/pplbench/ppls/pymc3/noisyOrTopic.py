# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pymc3 as pm


def obtain_posterior(data_train, args_dict, model=None):
    """
    PyMC3 implementation of noisy-or topic model

    Inputs:
    - data_train(tuple of np.ndarray): graph, words
    - args_dict: a dict of model arguments
    Returns:
    - samples_jags(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    graph = model
    words = data_train
    K = int(args_dict["k"])
    word_fraction = (args_dict["model_args"])[3]
    T = int(K * (1 - word_fraction))
    thinning = args_dict["thinning_pymc3"]
    assert len(words.shape) == 1
    num_samples = int(args_dict["num_samples_pymc3"])
    nodearray = np.concatenate([np.zeros(T), words])
    prob = np.zeros_like(nodearray)
    # sample the parameter posteriors, time it
    start_time = time.time()

    # Define model and sample
    if args_dict["inference_type"] == "mcmc":
        with pm.Model():
            for node in range(K):
                if node == 0:
                    nodearray[node] = 1
                else:
                    parent_accumulator = 0
                    for par, wt in enumerate(graph[:, node]):
                        if wt:
                            parent_accumulator += nodearray[par] * wt
                    prob[node] = 1 - np.exp(-np.sum(parent_accumulator))
                    if node < T:
                        pm.Bernoulli(f"node_{node}", p=prob[node])
                    else:
                        pm.Bernoulli(
                            f"node_{node}", p=prob[node], observed=nodearray[node]
                        )
            elapsed_time_compile_pymc3 = time.time() - start_time
            start_time = time.time()
            samples_pymc3 = pm.sample(
                num_samples, cores=1, chains=1, discard_tuned_samples=False
            )
    elif args_dict["inference_type"] == "vi":
        raise NotImplementedError
    parameters = [f"node_{i}" for i in range(1, T)]
    elapsed_time_sample_pymc3 = time.time() - start_time
    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(0, num_samples, thinning):
        sample_dict = {}
        sample_dict["node"] = np.zeros(T)
        sample_dict["node"][0] = 1
        for nodeid, parameter in enumerate(parameters):
            sample_dict["node"][nodeid + 1] = samples_pymc3[parameter][i]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_pymc3,
        "inference_time": elapsed_time_sample_pymc3,
    }

    return (samples, timing_info)
