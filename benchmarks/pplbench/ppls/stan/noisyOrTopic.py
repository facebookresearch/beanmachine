# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pickle
import time

import numpy as np
import pystan


def obtain_posterior(data_train, args_dict, model):
    """
    Stan impmementation of Noisy-Or Topic Model.

    Inputs:
    - data_train(tuple of np.ndarray): graph, words
    - args_dict: a dict of model arguments
    - model: the model object
    Returns:
    - samples_stan(dict): posterior samples of all parameters
    - timing_info(dict): compile_time, inference_time
    """
    graph = model
    words = data_train
    K = int(args_dict["k"])
    word_fraction = (args_dict["model_args"])[3]
    thinning = args_dict["thinning_stan"]
    T = int(K * (1 - word_fraction))
    assert len(words.shape) == 1
    # construct the Stan code for the graph
    tokens = graph[T:]
    topics = graph[:T]
    code = """
data {{
    int num_tokens;
    int tokens[num_tokens];
    real tau;
}}

transformed data {{
    int NUM_TOKENS = {};  // this better match num_tokens in input!
    int NUM_TOPICS = {};
}}

parameters {{
    // these two gumbel parameters control the sampling probability
    // for each node in the graph
    vector[2] gumbel_topics[NUM_TOPICS];
}}

transformed parameters {{
    vector[NUM_TOKENS] token_prob;
    vector[NUM_TOPICS] topic_prob;

    matrix[NUM_TOPICS, 2] topic;  // the state of each topic [True, False]

    """.format(
        len(tokens), len(topics) - 1
    )
    for node in range(1, T):
        code += """
    topic_prob[{}] = 1 - exp(""".format(
            node
        )
        for par, wt in enumerate(graph[:, node]):
            if par == 0:
                code += "-{}".format(wt)
            elif wt:
                code += "- topic[{}] * [{}, 0]'".format(par, wt)
        code += """);
    topic[{}] = softmax((log([topic_prob[{}], 1-topic_prob[{}]]')
        + gumbel_topics[{}])/tau)';
        """.format(
            node, node, node, node
        )
    # add the token probabilities
    for node in range(T, K):
        code += """
    token_prob[{}] = 1 - exp(""".format(
            node + 1 - T
        )
        for par, wt in enumerate(graph[:, node]):
            if par == 0:
                code += "-{}".format(wt)
            elif wt:
                code += "- topic[{}] * [{}, 0]'".format(par, wt)
        code += """);
        """
    # add the rest of the model
    code += """
}

model {
    for  (n in 1:NUM_TOPICS) {
        gumbel_topics[n] ~ gumbel(0, 1);
    }
    tokens ~ bernoulli(token_prob);
}
"""
    data_stan = {
        "num_tokens": len(tokens),
        "tau": 0.1,
        "tokens": np.array(words, dtype=int),
    }
    code_loaded = None
    if os.path.isfile("./ppls/stan/noisyOrTopic.pkl"):
        model, code_loaded, elapsed_time_compile_stan = pickle.load(
            open("./ppls/stan/noisyOrTopic.pkl", "rb")
        )
    if code_loaded != code:
        # compile the model, time it
        start_time = time.time()
        model = pystan.StanModel(model_code=code, model_name="noisy_or_topic")
        elapsed_time_compile_stan = time.time() - start_time
        # save it to the file 'model.pkl' for later use
        with open("./ppls/stan/noisyOrTopic.pkl", "wb") as f:
            pickle.dump((model, code, elapsed_time_compile_stan), f)

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
        samples_stan = fit.extract(pars=["topic"], permuted=False, inc_warmup=True)
        elapsed_time_sample_stan = time.time() - start_time

    elif args_dict["inference_type"] == "vi":
        # sample the parameter posteriors, time it
        start_time = time.time()
        fit = model.vb(
            data=data_stan, iter=args_dict["num_samples_stan"], pars=["topic"]
        )
        samples_stan = fit.extract(pars=["topic"], permuted=False, inc_warmup=True)
        elapsed_time_sample_stan = time.time() - start_time
    # convert shape of samples from prob(1), prob(0) to 1s and 0s
    samples_stan["topic"] = 1 - samples_stan["topic"].argmax(axis=3)
    # append the leak node as 1 in each sample as it is not sampled in the model
    samples_stan["topic"] = np.insert(samples_stan["topic"], 0, 1, axis=2)
    # repackage samples into shape required by PPLBench
    samples = []
    for i in range(int(args_dict["num_samples_stan"] / args_dict["thinning_stan"])):
        sample_dict = {}
        for parameter in samples_stan.keys():
            sample_dict["node"] = samples_stan[parameter][i].reshape(-1)
        samples.append(sample_dict)
    timing_info = {
        "compile_time": elapsed_time_compile_stan,
        "inference_time": elapsed_time_sample_stan,
    }
    return (samples, timing_info)
