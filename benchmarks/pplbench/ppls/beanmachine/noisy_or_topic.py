# Copyright (c) Facebook, Inc. and its affiliates
import time

import numpy as np
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.statistical_model import sample


class NoisyOrModel(object):
    def __init__(self, graph, words, T, iterations, inference_type):
        self.graph = graph
        self.words = words
        self.T = T
        self.iterations = iterations
        self.inference_type = inference_type

    # leak node is always on
    # i = 1 to T - 1
    @sample
    def node(self, i):
        parent_accumulator = torch.tensor(0.0)
        for par, wt in enumerate(self.graph[:, i]):
            if wt:
                parent_accumulator += self.node(par) * wt
        prob = 1 - torch.exp(-1 * parent_accumulator)
        return dist.Bernoulli(prob)

    # here j is the index of the words, so j starts from T
    @sample
    def y(self, j):
        parent_accumulator = torch.tensor(0.0)
        for par, wt in enumerate(self.graph[:, j]):
            if wt:
                parent_accumulator += self.node(par) * wt
        prob = 1 - torch.exp(-1 * parent_accumulator)
        return dist.Bernoulli(prob)

    def infer(self):
        dict_y = {self.y(k + self.T): self.words[k] for k in range(len(self.words))}
        dict_y[self.node(0)] = torch.tensor(1.0)
        if self.inference_type == "mcmc":
            nmc = SingleSiteNewtonianMonteCarlo()
            start_time = time.time()
            samples = nmc.infer(
                [self.node(i) for i in range(1, self.T)], dict_y, self.iterations, 1
            ).get_chain()
            elapsed_time_sample_beanmachine = time.time() - start_time
        elif self.inference_type == "vi":
            print("ImplementationError; exiting...")
            exit(1)
        return (samples, elapsed_time_sample_beanmachine)


def obtain_posterior(data_train, args_dict, model):
    """
    Beanmachine impmementation of Noisy-Or Topic Model.

    Inputs:
    - data_train(tuple of np.ndarray): graph, words
    - args_dict: a dict of model arguments
    Returns:
    - samples_bm(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    graph = model
    words = data_train
    words = torch.Tensor(words)

    K = int(args_dict["k"])
    word_fraction = (args_dict["model_args"])[3]
    T = int(K * (1 - word_fraction))
    iterations = int(args_dict["num_samples_beanmachine"])
    inference_type = args_dict["inference_type"]
    assert len(words.shape) == 1

    noisy_or_model = NoisyOrModel(graph, words, T, iterations, inference_type)
    samples, elapsed_time_bm = noisy_or_model.infer()
    # repackage samples into format required by PPLBench

    samples_formatted = []
    all_node_samples = np.zeros((iterations, T))
    for t in range(1, T):
        all_node_samples[:, t] = samples[noisy_or_model.node(t)].detach().numpy()
    all_node_samples[:, 0] = 1.0
    for i in range(iterations):
        sample_dict = {}
        sample_dict["node"] = all_node_samples[i]
        samples_formatted.append(sample_dict)
    timing_info = {"compile_time": 0, "inference_time": elapsed_time_bm}
    return (samples_formatted, timing_info)
