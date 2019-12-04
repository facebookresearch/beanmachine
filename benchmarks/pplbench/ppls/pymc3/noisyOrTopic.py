# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import theano
import theano.tensor as t
import pymc3 as pm


@theano.compile.ops.as_op(itypes=[t.lscalar, np.array, int], otypes=[t.lscalar])
def get_extracted_nodeoutcome(nodeoutcome, nodearray, node):
    nodeoutcome = int(nodeoutcome)
    nodearray[node] = nodeoutcome
    return nodearray


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
            # traverse through nodes
            for node in range(K):
                # leak node
                if node == 0:
                    nodearray[node] = 1
                else:
                    # compute the weighted sum of parent activations
                    parent_accumulator = 0
                    for par, wt in enumerate(graph[:, node]):
                        if wt:
                            parent_accumulator += nodearray[par] * wt
                    prob[node] = 1 - np.exp(-np.sum(parent_accumulator))
                    # topics are not observed
                    if node < T:
                        nodeoutcome = pm.Bernoulli(f"node_{node}", p=prob[node])
                        # debug: create a wrapper to extract a sample out of the distribution
                        # store the generated sample in nodearray so it's children can read it later
                        nodearray = get_extracted_nodeoutcome(nodeoutcome, nodearray, node)
                    # words are observed
                    else:
                        pm.Bernoulli(
                            f"node_{node}", p=prob[node], observed=nodearray[node]
                        )
            elapsed_time_compile_pymc3 = time.time() - start_time
            # Start sampling process
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
