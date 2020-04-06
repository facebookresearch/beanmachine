# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.compositional_infer import CompositionalInference
from beanmachine.ppl.model.statistical_model import sample
from torch import Tensor, tensor


class HiddenMarkovModel(object):
    def __init__(
        self,
        N: int,
        K: int,
        observations: List,
        num_samples: int,
        concentration: float,
        mu_loc: float,
        mu_scale: float,
        sigma_shape: float,
        sigma_rate: float,
        model: Optional[Tensor],
    ) -> None:
        self.observations = observations
        self.N = N
        self.K = K
        self.num_samples = num_samples

        self.concentration: float = concentration
        self.mu_loc = mu_loc
        self.mu_scale = mu_scale
        self.sigma_shape = sigma_shape
        self.sigma_rate = sigma_rate
        self.model = model

    @sample
    def Theta(self, k):
        return dist.Dirichlet(torch.ones(self.K) * self.concentration / self.K)

    @sample
    def Mu(self, k):
        return dist.Normal(self.mu_loc, self.mu_scale)

    @sample
    def Sigma(self, k):
        return dist.Gamma(self.sigma_shape, self.sigma_rate)

    # Hidden states
    @sample
    def X(self, n: int):
        if n == 0:
            return dist.Categorical(tensor([1.0] + [0.0] * (self.K - 1)))
        else:
            return dist.Categorical(self.Theta(self.X(n - 1).item()))

    # Noisy observations/emissions
    @sample
    def Y(self, n: int):
        return dist.Normal(self.Mu(self.X(n).item()), self.Sigma(self.X(n).item()))

    def infer(self):
        mh = CompositionalInference()
        queries = [self.X(self.N - 1)]
        if not self.model:
            queries += (
                [self.Theta(k) for k in range(self.K)]
                + [self.Mu(k) for k in range(self.K)]
                + [self.Sigma(k) for k in range(self.K)]
            )

        observations_dict = {self.Y(n): self.observations[n] for n in range(self.N)}
        observations_dict.update({self.X(0): tensor(0.0)})
        if self.model:
            observations_dict.update(
                {self.Theta(k): self.model["theta"][k] for k in range(self.K)}
            )
            observations_dict.update(
                {self.Mu(k): self.model["mus"][k] for k in range(self.K)}
            )
            observations_dict.update(
                {self.Sigma(k): self.model["sigmas"][k] for k in range(self.K)}
            )
        observations_dict = dict(observations_dict)

        start_time = time.time()
        inferred = mh.infer(queries, observations_dict, self.num_samples, num_chains=1)
        elapsed_time_sample_beanmachine = time.time() - start_time

        if self.model:
            samples = inferred.get_chain()[self.X(self.N - 1)].detach().numpy()

        else:
            # Return as a tuple of: thetas, mus, sigmas, xNs
            samples = (
                [
                    inferred.get_chain()[self.Theta(k)].detach().numpy()
                    for k in range(self.K)
                ],
                [
                    inferred.get_chain()[self.Mu(k)].detach().numpy()
                    for k in range(self.K)
                ],
                [
                    inferred.get_chain()[self.Sigma(k)].detach().numpy()
                    for k in range(self.K)
                ],
                inferred.get_chain()[self.X(self.N - 1)].detach().numpy(),
            )
        return (samples, elapsed_time_sample_beanmachine)


def obtain_posterior(
    data_train: List, args_dict: Dict, model: Optional[Tensor]
) -> Tuple[List, Dict]:
    """
    Beanmachine impmementation of HMM prediction.

    :param data_train:
    :param args_dict: a dict of model arguments
    :returns: samples_beanmachine(dict): posterior samples of all parameters
    :returns: timing_info(dict): compile_time, inference_time
    """
    concentration, mu_loc, mu_scale, sigma_shape, sigma_rate, observe_model = list(
        map(float, args_dict["model_args"])
    )
    N = int(args_dict["n"])
    K = int(args_dict["k"])
    num_samples = int(args_dict["num_samples"])
    if not observe_model:
        model = None

    start_time = time.time()
    # hmm = HiddenMarkovModel(N, K, theta, sigma, data_train, num_samples)
    hmm = HiddenMarkovModel(
        N,
        K,
        data_train,
        num_samples,
        concentration,
        mu_loc,
        mu_scale,
        sigma_shape,
        sigma_rate,
        model,
    )
    elapsed_time_compile_beanmachine = time.time() - start_time

    samples, elapsed_time_sample_beanmachine = hmm.infer()

    # repackage samples into shape required by PPLBench
    xn1str = "X[" + str(N - 1) + "]"
    if observe_model:
        samples_formatted = [{xn1str: xn1} for xn1 in samples]

    else:
        thetas, mus, sigmas, xn1s = samples
        # Want to swap the way these are ordered, so we can iterate through.
        thetas = [[thetasK[i] for thetasK in thetas] for i in range(num_samples)]
        mus = [[musK[i] for musK in mus] for i in range(num_samples)]
        sigmas = [[sigmasK[i] for sigmasK in sigmas] for i in range(num_samples)]
        # Now, e.g., thetas[i] gives the 'i'th MCMC sample of theta[0 .. K] as a list

        samples_formatted = [
            {"theta": theta, "mus": mu, "sigmas": sigma, xn1str: xn1}
            for theta, mu, sigma, xn1 in zip(thetas, mus, sigmas, xn1s)
        ]

    timing_info = {
        "compile_time": elapsed_time_compile_beanmachine,
        "inference_time": elapsed_time_sample_beanmachine,
    }
    return (samples_formatted, timing_info)
