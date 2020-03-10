# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pyjags


def obtain_posterior(data_train, args_dict, model):
    """
    Jags impmementation of Noisy-Or Topic Model.

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
    assert len(words.shape) == 1
    nodearray = np.concatenate([np.zeros(T), words])
    mask = np.concatenate((np.ones(T), np.zeros_like(words)))
    # Construct the jags code for the graph
    code = "model {\n"
    for node in range(K):
        if node == 0:
            code += "    node[{}] <- 1\n".format(node + 1)
        else:
            code += "    prob[{}] <- 1 - exp(-(".format(node + 1)
            for par, wt in enumerate(graph[:, node]):
                if wt:
                    code += "node[{}]*{} + ".format(par + 1, wt)
            code += "0))\n"
            code += "    node[{}] ~ dbern(prob[{}])\n".format(node + 1, node + 1)
    code += "}"
    data_jags = {"node": np.ma.masked_array(data=nodearray, mask=mask)}
    # compile the model, time it
    start_time = time.time()
    model = pyjags.Model(code, data=data_jags, chains=1, adapt=0)
    elapsed_time_compile_jags = time.time() - start_time
    if args_dict["inference_type"] == "mcmc":
        # sample the parameter posteriors, time it
        start_time = time.time()
        samples_jags = model.sample(int(args_dict["num_samples_jags"]), vars=["node"])
        elapsed_time_sample_jags = time.time() - start_time
    elif args_dict["inference_type"] == "vi":
        print("Jags does not support Variational Inference")
        exit()
    # repackage samples into shape required by PPLBench
    samples = []
    # move axes to facilitate iterating over samples
    # change (sample, chain, values) to (chain, sample, value)
    samples_jags["node"] = np.moveaxis(samples_jags["node"], [0, 1, 2], [1, 0, 2])
    for parameter in samples_jags.keys():
        # samples ndarray may have an extra dimension for num_chains which is 1
        # in PPLbench, we flatten it
        if samples_jags[parameter].shape[0] == 1:
            samples_jags[parameter] = samples_jags[parameter].squeeze()
    for i in range(int(args_dict["num_samples_jags"])):
        sample_dict = {}
        for parameter in samples_jags.keys():
            # convert from [parameter][sample] dict to [sample][parameter]
            # keep only the topics
            sample_dict[parameter] = samples_jags[parameter][i].reshape(-1)[:T]
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_jags,
        "inference_time": elapsed_time_sample_jags,
    }
    return (samples, timing_info)
