# Copyright (c) Facebook, Inc. and its affiliates
import time
from random import choices

from ppls.pplbench_ppl import PPLBenchPPL
from sklearn.linear_model import LogisticRegression as LR


"""
For model definition, see models/logisticRegressionModel.py
"""


class LogisticRegression(PPLBenchPPL):
    def obtain_posterior(self, data_train, args_dict, model=None):
        """
        Bootstrap impmementation of logistic regression model.

        :param data_train: tuple of np.ndarray (x_train, y_train)
        :param args_dict: a dict of model arguments
        :returns: samples_bootstrap(dict): posterior samples of all parameters
        :returns: timing_info(dict): compile_time, inference_time
        """

        # shape of x_train: (num_features, num_samples)
        x_train, y_train = data_train
        N = int(x_train.shape[1])
        num_samples = args_dict["num_samples_bootstrapping"]

        # x_train is now (num_samples, num_features)
        x_train = x_train.T

        samples = []
        inference_start_time = time.time()
        for _i in range(num_samples):
            sample_dict = {}
            # randomnly pick with replacement N training observations
            indices = list(range(N))
            chosen_indices = choices(indices, k=N)
            x_train_sampled = x_train[chosen_indices, :]
            y_train_sampled = y_train[chosen_indices]

            clf = LR(random_state=0).fit(x_train_sampled, y_train_sampled)

            alpha = clf.intercept_

            # shape: (1, num_features)
            beta = clf.coef_
            sample_dict["alpha"] = alpha
            sample_dict["beta"] = beta
            samples.append(sample_dict)

        inference_time = time.time() - inference_start_time

        timing_info = {"compile_time": 0, "inference_time": inference_time}
        return (samples, timing_info)
