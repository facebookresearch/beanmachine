# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.tensor as tensor
from beanmachine.ppl.examples.conjugate_models.beta_binomial import BetaBinomialModel
from beanmachine.ppl.examples.conjugate_models.categorical_dirichlet import (
    CategoricalDirichletModel,
)
from beanmachine.ppl.examples.conjugate_models.gamma_gamma import GammaGammaModel
from beanmachine.ppl.examples.conjugate_models.gamma_normal import GammaNormalModel
from beanmachine.ppl.examples.conjugate_models.normal_normal import NormalNormalModel
from beanmachine.ppl.inference.abstract_infer import AbstractInference
from beanmachine.ppl.model.utils import RVIdentifier
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
        self,
    ) -> Tuple[Tensor, Tensor, List[RVIdentifier], Dict[RVIdentifier, Tensor]]:
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
            (alpha * beta) / ((alpha + beta).pow(2.0) * (alpha + beta + 1.0))
        ).pow(0.5)

        return (expected_mean, expected_std, queries, observations)

    def compute_gamma_gamma_moments(
        self,
    ) -> Tuple[Tensor, Tensor, List[RVIdentifier], Dict[RVIdentifier, Tensor]]:
        """
        Computes mean and standard deviation of a small gamma gamma model.

        :return: expected mean, expected standard deviation, conjugate model
        queries and observations
        """
        shape = tensor([2.0, 2.0])
        rate = tensor([2.0, 2.0])
        alpha = tensor([1.5, 1.5])
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
        self,
    ) -> Tuple[Tensor, Tensor, List[RVIdentifier], Dict[RVIdentifier, Tensor]]:
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
        rate = rate + (deviations * (0.5))
        expected_mean = shape / rate
        expected_std = (expected_mean / rate).pow(0.5)
        return (expected_mean, expected_std, queries, observations)

    def compute_normal_normal_moments(
        self,
    ) -> Tuple[Tensor, Tensor, List[RVIdentifier], Dict[RVIdentifier, Tensor]]:
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
        # pyre-fixme[7]: Expected `Tuple[Tensor, Tensor, List[RVIdentifier],
        #  Dict[RVIdentifier, Tensor]]` but got `Tuple[float, typing.Any,
        #  List[typing.Any], Dict[typing.Any, typing.Any]]`.
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

    def _compare_run(
        self,
        moments: Tuple[Tensor, Tensor, List[RVIdentifier], Dict[RVIdentifier, Tensor]],
        mh: AbstractInference,
        num_chains: int,
        num_samples: int,
        delta: float,
        random_seed: Optional[int],
        num_adaptive_samples: int = 0,
    ):
        expected_mean, expected_std, queries, observations = moments

        if random_seed is not None:
            torch.manual_seed(random_seed)

        predictions = mh.infer(
            queries,
            observations,
            num_samples,
            num_chains=num_chains,
            num_adaptive_samples=num_adaptive_samples,
        )
        for i in range(predictions.get_num_chains()):
            mean, _ = self.compute_statistics(predictions.get_chain(i)[queries[0]])
            # pyre-fixme[16]: `AbstractConjugateTests` has no attribute
            #  `assertAlmostEqual`.
            self.assertAlmostEqual(
                torch.abs(mean - expected_mean).sum().item(), 0, delta=delta
            )

    def beta_binomial_conjugate_run(
        self,
        mh: AbstractInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        delta: float = 0.05,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small beta binomial model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param delta: delta to check against expected results
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_beta_binomial_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            delta,
            random_seed,
            num_adaptive_samples,
        )

    def gamma_gamma_conjugate_run(
        self,
        mh: AbstractInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        delta: float = 0.05,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small gamma gamma model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param delta: delta to check against expected results
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_gamma_gamma_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            delta,
            random_seed,
            num_adaptive_samples,
        )

    def gamma_normal_conjugate_run(
        self,
        mh: AbstractInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        delta: float = 0.05,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small gamma normal model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param delta: delta to check against expected results
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_gamma_normal_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            delta,
            random_seed,
            num_adaptive_samples,
        )

    def normal_normal_conjugate_run(
        self,
        mh: AbstractInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        delta: float = 0.05,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small normal normal model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param delta: delta to check against expected results
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_normal_normal_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            delta,
            random_seed,
            num_adaptive_samples,
        )

    def distant_normal_normal_conjugate_run(
        self,
        mh: AbstractInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        delta: float = 0.05,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small normal normal model
        where the prior and posterior are far from each other.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param delta: delta to check against expected results
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_distant_normal_normal_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            delta,
            random_seed,
            num_adaptive_samples,
        )

    def dirichlet_categorical_conjugate_run(
        self,
        mh: AbstractInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        delta: float = 0.05,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small dirichlet categorical model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param delta: delta to check against expected results
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_dirichlet_categorical_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            delta,
            random_seed,
            num_adaptive_samples,
        )

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
