# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time

import numpy as np
import pymc3 as pm
from ppls.pplbench_ppl import PPLBenchPPL


class CrowdSourcedAnnotation(PPLBenchPPL):
    def obtain_posterior(self, data_train, args_dict, model=None):
        """
        PyMC3 implementation of crowdsourced annotation model

        Inputs:
        - data_train(tuple of np.ndarray): vector_y, vector_J_i, num_labels
        - args_dict: a dict of model arguments
        Returns:
        - samples(dict): posterior samples of all parameters
        - timing_info(dict): compile_time, inference_time
        """
        vector_y, vector_J_i, num_labels = data_train
        n_labelers = int(args_dict["k"])
        n_items = len(num_labels)
        K, labeler_rate, expected_correctness, concentration = args_dict["model_args"]
        num_samples = args_dict["num_samples_pymc3"]
        # item array for pymc3; change from list-of-lengths format
        # of num_labels to list-of-items associated with each label
        # and labeler in vector_y, vector_J_i
        item = []
        for i in range(len(num_labels)):
            item.extend(np.repeat(i, num_labels[i]))
        # set prior that each labeler on average has 50% chance of getting true label
        alpha = ((1 - expected_correctness) / (K - 1)) * np.ones([K, K]) + (
            expected_correctness - (1 - expected_correctness) / (K - 1)
        ) * np.eye(K)
        alpha *= concentration
        # set prior that each item is equally likely to be in any given label
        beta = 1 / K * np.ones(K)
        # sample the parameter posteriors, time it
        start_time = time.time()

        # Define model and sample
        if args_dict["inference_type"] == "mcmc":
            with pm.Model():
                # sample a true class z for each item
                pi = pm.Dirichlet("pi", beta)
                z = pm.Categorical("z", p=pi, shape=n_items)
                # sample confusion matrices theta for labelers from this dirichlet prior
                theta = pm.Dirichlet("theta", a=alpha, shape=(n_labelers, K, K))
                # likelihood
                pm.Categorical(
                    "labels",
                    p=theta[vector_J_i, z[item]],
                    observed=vector_y,
                    shape=n_items,
                )
                elapsed_time_compile_pymc3 = time.time() - start_time
                start_time = time.time()
                samples_pymc3 = pm.sample(
                    num_samples, cores=1, chains=1, discard_tuned_samples=False
                )

        elif args_dict["inference_type"] == "vi":
            raise NotImplementedError

        elapsed_time_sample_pymc3 = time.time() - start_time
        # repackage samples into shape required by PPLBench
        samples = []
        for i in range(0, int(args_dict["num_samples_pymc3"])):
            sample_dict = {}
            for parameter in ["theta", "pi"]:
                sample_dict[parameter] = samples_pymc3[parameter][i]
            samples.append(sample_dict)
        timing_info = {
            "compile_time": elapsed_time_compile_pymc3,
            "inference_time": elapsed_time_sample_pymc3,
        }

        return (samples, timing_info)
