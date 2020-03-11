# Copyright (c) Facebook, Inc. and its affiliates
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.statistical_model import sample
from ppls.pplbench_ppl import PPLBenchPPL
from torch import Tensor


"""
For model definition, see models/logistic_regression_model.py
"""


class LogisticRegressionModel(object):
    def __init__(
        self,
        N: int,
        K: int,
        scale_alpha: float,
        scale_beta: List[float],
        loc_beta: float,
        num_samples: int,
        inference_type: str,
        X: Tensor,
        Y: Tensor,
    ):
        self.N = N
        self.K = K
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.loc_beta = loc_beta
        self.num_samples = num_samples
        self.inference_type = inference_type
        self.X = torch.cat((torch.ones((1, N)), X))
        self.Y = Y

    @sample
    def beta(self):
        return dist.Normal(
            tensor([0.0] + [self.loc_beta] * self.K),
            tensor([self.scale_alpha] + self.scale_beta),
        )

    @sample
    def y(self):
        # Compute X * Beta
        beta_ = self.beta().reshape((1, self.beta().shape[0]))
        mu = beta_.mm(self.X)
        return dist.Bernoulli(logits=mu)

    def infer(self):
        dict_y = {self.y(): self.Y}
        if self.inference_type == "mcmc":
            nmc = SingleSiteNewtonianMonteCarlo()
            start_time = time.time()
            samples = nmc.infer([self.beta()], dict_y, self.num_samples, 1).get_chain()
        elif self.inference_type == "vi":
            print("ImplementationError; exiting...")
            exit(1)
        elapsed_time_sample_beanmachine = time.time() - start_time
        return (samples, elapsed_time_sample_beanmachine)


class LogisticRegression(PPLBenchPPL):
    def obtain_posterior(
        self, data_train: Tuple[np.ndarray, np.ndarray], args_dict: Dict, model=None
    ) -> Tuple[List, Dict]:
        """
        Beanmachine impmementation of logisitc regression model.

        :param data_train: tuple of np.ndarray (x_train, y_train)
        :param args_dict: a dict of model arguments
        :returns: samples_beanmachine(list of dict): posterior samples of all parameters
        :returns: timing_info(dict): compile_time, inference_time
        """
        # shape of x_train: (num_features, num_samples)
        x_train, y_train = data_train
        y_train = np.array(y_train, dtype=np.float32)
        x_train = np.array(x_train, dtype=np.float32)
        x_train = tensor(x_train)
        y_train = tensor([tensor(y) for y in y_train])
        N = int(x_train.shape[1])
        K = int(x_train.shape[0])

        alpha_scale = float((args_dict["model_args"])[0])
        beta_scale = [float((args_dict["model_args"])[1])] * K
        beta_loc = float((args_dict["model_args"])[2])
        num_samples = args_dict["num_samples_beanmachine-vectorized"]
        inference_type = args_dict["inference_type"]

        start_time = time.time()
        logistic_regression_model = LogisticRegressionModel(
            N,
            K,
            alpha_scale,
            beta_scale,
            beta_loc,
            num_samples,
            inference_type,
            x_train,
            y_train,
        )
        elapsed_time_compile_beanmachine = time.time() - start_time
        samples, elapsed_time_sample_beanmachine = logistic_regression_model.infer()

        # repackage samples into format required by PPLBench
        # List of dict, where each dict has key = param (string), value = value of param
        param_keys = ["beta", "alpha"]
        samples_formatted = []
        for i in range(num_samples):
            sample_dict = {}
            for j, parameter in enumerate(samples.get_rv_names()):
                sample_dict[param_keys[j]] = (
                    samples.get_variable(parameter)[i][1:].detach().numpy()
                )
                sample_dict[param_keys[1]] = (
                    samples.get_variable(parameter)[i][0].detach().numpy()
                )
            samples_formatted.append(sample_dict)

        timing_info = {
            "compile_time": elapsed_time_compile_beanmachine,
            "inference_time": elapsed_time_sample_beanmachine,
        }
        return (samples_formatted, timing_info)
