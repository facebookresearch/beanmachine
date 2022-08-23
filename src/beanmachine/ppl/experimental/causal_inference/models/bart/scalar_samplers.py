# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.node import LeafNode
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal


class NoiseStandardDeviation:
    """The NoiseStandardDeviation class encapsulates the noise standard deviation.
    The variance is parametrized by an inverse-gamma prior which is conjugate to a normal likelihood.

    Args:
        prior_concentration (float): Also called alpha. Must be greater than zero.
        prior_rate (float): Also called beta. Must be greater than 0.
        val (float): Current value of noise standard deviation.
    """

    def __init__(
        self, prior_concentration: float, prior_rate: float, val: Optional[float] = None
    ):
        if prior_concentration <= 0 or prior_rate <= 0:
            raise ValueError("Invalid prior hyperparameters")
        self.prior_concentration = prior_concentration
        self.prior_rate = prior_rate
        if val is None:
            self.sample(X=torch.Tensor([]), residual=torch.Tensor([]))  # prior init
        else:
            self._val = val

    @property
    def val(self) -> float:
        return self._val

    @val.setter
    def val(self, val: float):
        self._val = val

    def sample(self, X: torch.Tensor, residual: torch.Tensor) -> float:
        """Sample from the posterior distribution of sigma.
        If empty tensors are passed for X and residual, there will be no update so the sampling will be from the prior.

        Note:
            This sets the value of the `val` attribute to the drawn sample.

        Args:
            X: Covariate matrix / training data shape (num_observations, input_dimensions).
            residual: The current residual of the model shape (num_observations, 1).
        """
        self.val = self._get_sample(X, residual)
        return self.val

    def _get_sample(self, X: torch.Tensor, residual: torch.Tensor) -> float:
        """
        Draw a sample from the posterior.

        Args:
            X: Covariate matrix / training data of shape (num_observations, input_dimensions).
            residual: The current residual of the model of shape (num_observations, 1).

        """
        posterior_concentration = self.prior_concentration + (len(X) / 2.0)
        posterior_rate = self.prior_rate + (0.5 * (torch.sum(torch.square(residual))))
        draw = torch.pow(Gamma(posterior_concentration, posterior_rate).sample(), -0.5)
        return draw.item()


class LeafMean:
    """
    Class to sample form the prior and posterior distributions of the leaf nodes in BART.

    Reference:
        [1] Hugh A. Chipman, Edward I. George, Robert E. McCulloch (2010). "BART: Bayesian additive regression trees"
        https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full

    Args:
        prior_loc: Prior location parameter.
        prior_scale: Prior scale parameter.
    """

    def __init__(self, prior_loc: float, prior_scale: float):
        if prior_scale < 0:
            raise ValueError("Invalid prior hyperparameters")
        self._prior_loc = prior_loc
        self._prior_scale = prior_scale

    @property
    def prior_scale(self):
        return self._prior_scale

    def sample_prior(self):
        return Normal(loc=self._prior_loc, scale=self._prior_scale).sample().item()

    def sample_posterior(
        self,
        node: LeafNode,
        X: torch.Tensor,
        y: torch.Tensor,
        current_sigma_val: float,
    ):
        X_in_node, y_in_node = node.data_in_node(X, y)
        if len(X_in_node) == 0:
            return None  # no new data
        num_points_in_node = len(X_in_node)
        prior_variance = (self._prior_scale) ** 2
        likelihood_variance = (current_sigma_val**2) / num_points_in_node
        likelihood_mean = torch.sum(y_in_node) / num_points_in_node
        posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / likelihood_variance)
        posterior_mean = (
            likelihood_mean * prior_variance + self._prior_loc * likelihood_variance
        ) / (likelihood_variance + prior_variance)
        return (
            Normal(loc=posterior_mean, scale=math.sqrt(posterior_variance))
            .sample()
            .item()
        )
