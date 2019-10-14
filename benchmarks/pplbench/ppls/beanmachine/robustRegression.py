import time
from typing import Any, Dict, List, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.model.statistical_model import sample
from torch import Tensor


"""
For model definition, see models/robustRegressionModel.py
"""


class RobustRegressionModel(object):
    def __init__(
        self,
        N: int,
        K: int,
        scale_alpha: float,
        scale_beta: float,
        loc_beta: float,
        rate_sigma: float,
        num_samples: int,
        inference_type: str,
        X: Tensor,
        Y: Tensor,
    ):
        self.N = N
        self.K = K
        self.scale_alpha = scale_alpha
        self.loc_beta = loc_beta
        self.scale_beta = scale_beta
        self.rate_sigma = rate_sigma
        self.num_samples = num_samples
        self.inference_type = inference_type
        self.X = X
        self.Y = Y

    @sample
    def nu(self):
        return dist.Gamma(2.0, 0.1)

    @sample
    def sigma(self):
        return dist.Exponential(self.rate_sigma)

    @sample
    def alpha(self):
        return dist.Normal(0, self.scale_alpha)

    @sample
    def beta(self, k: int):
        return dist.Normal(self.loc_beta, self.scale_beta)

    @sample
    def y(self, i: int):
        # Compute X * Beta
        mean = self.alpha()
        for j in range(self.K):
            mean = mean + self.beta(j) * self.X[j][i]
        return dist.StudentT(self.nu(), mean, self.sigma())

    def infer(self):
        dict_y = {self.y(i): self.Y[i] for i in range(self.N)}
        if self.inference_type == "mcmc":
            samples_beta = torch.zeros([self.num_samples, self.K])
            mh = SingleSiteAncestralMetropolisHastings()
            start_time = time.time()
            samples = mh.infer(
                [self.nu(), self.sigma(), self.alpha()]
                + [self.beta(j) for j in range(self.K)],
                dict_y,
                self.num_samples,
            )
            for j in range(self.K):
                samples_beta[:, j] = [samples[self.beta(k)] for k in range(self.K)][j]
        elif self.inference_type == "vi":
            print("ImplementationError; exiting...")
            exit(1)
        elapsed_time_sample_beanmachine = time.time() - start_time
        return (samples_beta.detach().numpy(), samples, elapsed_time_sample_beanmachine)


def obtain_posterior(
    data_train: Tuple[Any, Any], args_dict: Dict, model: RobustRegressionModel
) -> Tuple[List, Dict]:
    """
    Beanmachine impmementation of robust regression model.

    :param data_train: tuple of np.ndarray (x_train, y_train)
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    # shape of x_train: (num_features, num_samples)
    x_train, y_train = data_train
    y_train = [tensor(y) for y in y_train]
    N = int(x_train.shape[1])
    K = int(x_train.shape[0])

    alpha_scale, beta_scale, beta_loc, sigma_mean = args_dict["model_args"]
    num_samples = args_dict["num_samples_beanmachine"]
    inference_type = args_dict["inference_type"]

    start_time = time.time()
    robust_regression_model = RobustRegressionModel(
        N,
        K,
        alpha_scale,
        beta_scale,
        beta_loc,
        1.0 / sigma_mean,
        num_samples,
        inference_type,
        Tensor(x_train),
        Tensor(y_train),
    )
    elapsed_time_compile_beanmachine = time.time() - start_time
    samples_beta, samples, elapsed_time_sample_beanmachine = (
        robust_regression_model.infer()
    )

    # repackage samples into format required by PPLBench
    # List of dict, where each dict has key = param (string), value = value of param
    param_keys = ["nu", "sigma", "alpha"]
    samples_formatted = []
    for i in range(num_samples):
        sample_dict = {}
        sample_dict["beta"] = samples_beta[i]
        for j, parameter in enumerate(samples.keys()):
            if j >= len(param_keys):
                break
            sample_dict[param_keys[j]] = samples[parameter][i].item()
        samples_formatted.append(sample_dict)

    timing_info = {
        "compile_time": elapsed_time_compile_beanmachine,
        "inference_time": elapsed_time_sample_beanmachine,
    }
    return (samples_formatted, timing_info)
