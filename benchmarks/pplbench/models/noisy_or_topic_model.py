# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Module for Noisy-Or Topic Model aka Latent Keyphrase Inference (LAKI)

Paper describing this: http://hanj.cs.illinois.edu/pdf/www16_jliu.pdf

Module contains following methods:
get_defaults()
generate_model()
generate_data()
evaluate_posterior_predictive()

Model Specification:
This is a Bayesian Network with boolean-valued random variables with Noisy-OR
dependencies.In this case we are inferring the latent topics given the set of
observed words.

Total number of nodes(Z): K(default=150)
One leak node(O)
Number of word nodes(W) : (word_fraction) * K
Number of topics(T): (1 - word_fraction) * K
Z is set of all nodes O,W,T

Word nodes W can only have topic nodes T as parents
All nodes Z have leak node O as parent
children per node ~poisson(rate_fanout)
Leak node weight W_oj  ~exp(avg_leak_weight)
All other weights W_ij ~exp(avg_weight)

for all nodes in Z
P(Z_j = 1| Pa(Z_j)) = 1 - exp(-W_oj - sum_over_parents_i(W_ij*Z_i))
Z_j ~ bernoulli(P(Z_j = 1| Pa(Z_j)))

Model specific arguments:
Pass these arguments in following order -
[rate_fanout, avg_leak_weight, avg_weight, word_fraction]
"""

import numpy as np
from scipy import stats


def get_defaults():
    """
    Return defaults for Noisy-Or Topic Model
    """
    defaults = {
        "n": 2,  # number of sentences
        "k": 150,  # number of nodes in the network
        "runtime": 200,
        "train_test_ratio": 0.5,
        "trials": 100,
        # [rate_fanout, avg_leak_weight, avg_weight, word_fraction]
        "model_args": [3, 0.1, 1, 0.66],
    }
    return defaults


def generate_model(args_dict):
    """
    Generate model structure for Noisy-Or Topic Model
    Inputs:
    Model arguments
    Returns:
    model graph (ndarray, KxK) : graph strucutre as a weighted adjecency matrix.
                                 graph[x,y] denotes weight of parent x to child y
    """
    np.random.seed(args_dict["rng_seed"])
    K = int(args_dict["k"])
    # parameters for distributions to sample parameters from
    rate_fanout, avg_leak_weight, avg_weight, word_fraction = args_dict["model_args"]
    T = int(K * (1 - word_fraction))
    assert T > 1
    assert K > T
    # create Bayesian network
    graph = np.zeros([K, K])  # adjacency weight matrix
    graph[0, 1:] = np.random.exponential(avg_leak_weight, K - 1)
    for node in range(1, T):
        num_children = np.random.poisson(rate_fanout)
        children = (
            node
            + 1
            + np.random.choice(int(K - 1 - node), size=num_children, replace=False)
        )
        graph[node, children] = np.random.exponential(avg_weight, num_children)
    return graph


def generate_data(args_dict, model):
    """
    Generate data for Noisy-Or Topic model.
    Inputs:
    - N(int) = number of data points to generate
    - K(int) = number of covariates in the model

    Returns:
    - generated_data(dict) = data_train (train sentence), data test (test sentence)
    """
    np.random.seed(args_dict["rng_seed"])
    print("Generating data")
    graph = model
    K = int(args_dict["k"])
    N = int(args_dict["n"])
    assert N == 2
    train_test_ratio = float(args_dict["train_test_ratio"])
    rate_fanout, avg_leak_weight, avg_weight, word_fraction = args_dict["model_args"]
    T = int(K * (1 - word_fraction))
    # sample!
    sample = np.zeros([N, K])
    sample[:, 0] = 1  # leaky node is always on
    # sample the topic nodes first
    for node in range(1, T):
        p_node = 1 - np.exp(-np.sum(graph[:, node] * sample[0]))
        # sample with this probability, boadcast to all N
        sample[:, node] = np.random.binomial(1, p_node)
    # sample the words with fixed topic nodes
    for node in range(T, K):
        # 'sample' acts as mask while graph[:, node] are the weights
        # N * K -> [N,K]; sum over axis 1 -> N
        p_node = 1 - np.exp(-np.sum(graph[:, node] * sample, axis=1))
        sample[:, node] = np.random.binomial(1, p_node, size=N)
    # Split in topic/sentence
    topics, words = sample[:, :T], sample[:, T:]
    split = int(N * train_test_ratio)
    assert np.array_equal(topics[0], topics[1])
    # reshape to flatten the words to 1d array; be sure to pass one sentence
    # for train and test each
    return {
        "data_train": (words[:split].reshape(-1)),
        "data_test": (words[split:].reshape(-1)),
    }


def evaluate_posterior_predictive(samples, data_test, model):
    """
    Computes the likelihood of held-out testset wrt parameter samples

    input-
    samples(dict): parameter samples from model
    data_test: test data
    model: modelobject

    returns- pred_log_lik_array(np.ndarray): log-likelihoods of data
            wrt parameter samples
    """
    words, graph = data_test, model
    K = graph.shape[0]
    T = K - len(words)
    pred_log_lik_array = []
    for sample in samples:
        log_lik_test = 0
        for node in range(T, K):
            p_node = 1 - np.exp(
                -np.sum(graph[:T, node] * np.array(sample["node"], dtype=float))
            )
            log_lik_test += stats.binom.logpmf(k=words[node - T], n=1, p=p_node)
        pred_log_lik_array.append(log_lik_test)
    # return as a numpy array
    return np.array(pred_log_lik_array)
