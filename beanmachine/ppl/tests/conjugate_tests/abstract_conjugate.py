# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.tensor as tensor
from beanmachine.ppl.examples.conjugate_models.beta_binomial import BetaBinomialModel
from beanmachine.ppl.examples.conjugate_models.categorical_dirichlet import (
    CategoricalDirichletModel,
)
from beanmachine.ppl.examples.conjugate_models.gamma_gamma import GammaGammaModel
from beanmachine.ppl.examples.conjugate_models.gamma_normal import GammaNormalModel
from beanmachine.ppl.examples.conjugate_models.normal_normal import NormalNormalModel
from beanmachine.ppl.model.utils import RandomVariable
from torch import Tensor


class AbstractConjugateTests(metaclass=ABCMeta):
    """
     Computes the posterior mean and standard deviation of some of the conjugate
     distributions included below.
     https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions
    """

    def compute_statistics(self, predictions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes mean and standard deviation of a given tensor of samples.

        :param predictions: tensor of samples
        :returns: mean and standard deviation of the tensor of samples.
        """
        return (torch.mean(predictions, 0), torch.std(predictions, 0))

    def compute_beta_binomial_moments(
        self
    ) -> Tuple[Tensor, Tensor, List[RandomVariable], Dict[RandomVariable, Tensor]]:
        """
        Computes mean and standard deviation of a small beta binomial model.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        alpha = tensor([2.0, 2.0])
        beta = tensor([1.0, 1.0])
        trials = tensor([1.0, 1.0])
        obs = tensor([1.0, 0.0])
        model = BetaBinomialModel(alpha, beta, trials)
        queries = [model.beta()]
        observations = {model.binomial(): obs}
        alpha = alpha + obs
        beta = beta - obs + trials
        expected_mean = alpha / (alpha + beta)
        expected_std = (
            (alpha * beta) / (alpha + beta).pow(2.0) * (alpha + beta + 1.0)
        ).pow(0.5)

        return (expected_mean, expected_std, queries, observations)

    def compute_gamma_gamma_moments(
        self
    ) -> Tuple[Tensor, Tensor, List[RandomVariable], Dict[RandomVariable, Tensor]]:
        """
        Computes mean and standard deviation of a small gamma gamma model.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        shape = tensor([1.0, 1.0])
        rate = tensor([2.0, 2.0])
        alpha = tensor([0.5, 0.5])
        obs = tensor([2.0, 4.0])
        model = GammaGammaModel(shape, rate, alpha)
        queries = [model.gamma_p()]
        observations = {model.gamma(): obs}
        shape = shape + alpha
        rate = rate + obs
        expected_mean = shape / rate
        expected_std = (expected_mean / rate).pow(0.5)
        return (expected_mean, expected_std, queries, observations)

    def compute_gamma_normal_moments(
        self
    ) -> Tuple[Tensor, Tensor, List[RandomVariable], Dict[RandomVariable, Tensor]]:
        """
        Computes mean and standard deviation of a small gamma normal model.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        shape = tensor([1.0, 1.0])
        rate = tensor([2.0, 2.0])
        mu = tensor([1.0, 2.0])
        obs = tensor([1.5, 2.5])
        model = GammaNormalModel(shape, rate, mu)
        queries = [model.gamma()]
        observations = {model.normal(): obs}
        shape = shape + tensor([0.5, 0.5])
        deviations = (obs - mu).pow(2.0)
        deviations = deviations.sum(0)
        rate = rate + (deviations * (0.5))
        expected_mean = shape / rate
        expected_std = (expected_mean / rate).pow(0.5)
        return (expected_mean, expected_std, queries, observations)

    def compute_normal_normal_moments(
        self
    ) -> Tuple[Tensor, Tensor, List[RandomVariable], Dict[RandomVariable, Tensor]]:
        """
        Computes mean and standard deviation of a small normal normal model.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        mu = tensor([1.0, 1.0])
        std = tensor([1.0, 1.0])
        sigma = tensor([1.0, 1.0])
        obs = tensor([1.5, 2.5])
        model = NormalNormalModel(mu, std, sigma)
        queries = [model.normal_p()]
        observations = {model.normal(): obs}
        expected_mean = (1 / (1 / sigma.pow(2.0) + 1 / std.pow(2.0))) * (
            mu / std.pow(2.0) + obs / sigma.pow(2.0)
        )
        expected_std = (std.pow(2.0) + sigma.pow(2.0)).pow(-0.5)
        return (expected_mean, expected_std, queries, observations)

    def compute_distant_normal_normal_moments(self):
        """
        Computes mean and standard deviation of a small normal normal model
        where the prior and posterior are far from each other.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        mu = tensor([1.0, 1.0])
        std = tensor([1.0, 1.0])
        sigma = tensor([1.0, 1.0])
        obs = tensor([100.0, 100.0])
        model = NormalNormalModel(mu, std, sigma)
        queries = [model.normal_p()]
        observations = {model.normal(): obs}
        expected_mean = (1 / (1 / sigma.pow(2.0) + 1 / std.pow(2.0))) * (
            mu / std.pow(2.0) + obs / sigma.pow(2.0)
        )
        expected_std = (std.pow(2.0) + sigma.pow(2.0)).pow(-0.5)
        return (expected_mean, expected_std, queries, observations)

    def compute_dirichlet_categorical_moments(self):
        """
        Computes mean and standard deviation of a small dirichlet categorical
        model.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        alpha = tensor([0.5, 0.5])
        model = CategoricalDirichletModel(alpha)
        obs = tensor([1.0])
        queries = [model.dirichlet()]
        observations = {model.categorical(): obs}
        alpha = alpha + tensor([0.0, 1.0])
        expected_mean = alpha / alpha.sum()
        expected_std = expected_mean * (1 - expected_mean) / (alpha.sum() + 1)
        return (expected_mean, expected_std, queries, observations)

    @abstractmethod
    def test_beta_binomial_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_beta_binomial_conjugate_run."
        )

    @abstractmethod
    def test_gamma_gamma_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_gamma_gamma_conjugate_run."
        )

    @abstractmethod
    def test_gamma_normal_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_gamma_normal_conjugate_run."
        )

    @abstractmethod
    def test_normal_normal_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_normal_normal_conjugate_run."
        )

    @abstractmethod
    def test_distant_normal_normal_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_normal_normal_conjugate_run."
        )

    @abstractmethod
    def test_dirichlet_categorical_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_categorical_dirichlet_conjugate_run."
        )
