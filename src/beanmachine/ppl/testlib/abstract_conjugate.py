# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch
from beanmachine.ppl.diagnostics.common_statistics import effective_sample_size
from beanmachine.ppl.examples.conjugate_models.beta_binomial import BetaBinomialModel
from beanmachine.ppl.examples.conjugate_models.categorical_dirichlet import (
    CategoricalDirichletModel,
)
from beanmachine.ppl.examples.conjugate_models.gamma_gamma import GammaGammaModel
from beanmachine.ppl.examples.conjugate_models.gamma_normal import GammaNormalModel
from beanmachine.ppl.examples.conjugate_models.normal_normal import NormalNormalModel
from beanmachine.ppl.inference import utils
from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.testlib.hypothesis_testing import (
    mean_equality_hypothesis_confidence_interval,
    variance_equality_hypothesis_confidence_interval,
)
from torch import Tensor, tensor


class AbstractConjugateTests(metaclass=ABCMeta):
    """
    Computes the posterior mean and standard deviation of some of the conjugate
    distributions included below.
    https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions

    Note: Whenever possible, we will use same variable names as on that page.
    """

    def compute_statistics(self, predictions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes mean and standard deviation of a given tensor of samples.

        :param predictions: tensor of samples
        :returns: mean and standard deviation of the tensor of samples.
        """
        return (
            torch.mean(predictions, 0),
            torch.std(predictions, 0, unbiased=True, keepdim=True),
        )

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
        n = tensor([1.0, 1.0])
        obs = tensor([1.0, 0.0])
        model = BetaBinomialModel(alpha, beta, n)
        queries = [model.theta()]
        observations = {model.x(): obs}
        alpha_prime = alpha + obs
        beta_prime = beta - obs + n
        mean_prime = alpha_prime / (alpha_prime + beta_prime)
        std_prime = (
            (alpha_prime * beta_prime)
            / ((alpha_prime + beta_prime).pow(2.0) * (alpha_prime + beta_prime + 1.0))
        ).pow(0.5)

        return (mean_prime, std_prime, queries, observations)

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
        expected_std = (std.pow(-2.0) + sigma.pow(-2.0)).pow(-0.5)
        # pyre-fixme[7]: Expected `Tuple[Tensor, Tensor, List[RVIdentifier],
        #  Dict[RVIdentifier, Tensor]]` but got `Tuple[float, typing.Any,
        #  List[typing.Any], Dict[typing.Any, typing.Any]]`.
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
        expected_std = (expected_mean * (1 - expected_mean) / (alpha.sum() + 1)).pow(
            0.5
        )
        return (expected_mean, expected_std, queries, observations)

    def _compare_run(
        self,
        moments: Tuple[Tensor, Tensor, List[RVIdentifier], Dict[RVIdentifier, Tensor]],
        mh: BaseInference,
        num_chains: int,
        num_samples: int,
        random_seed: Optional[int],
        num_adaptive_samples: int = 0,
        alpha: float = 0.01,
    ):

        # Helper functions for hypothesis tests
        def chi2(alpha, df):
            return scipy.stats.chi2.ppf(alpha, df)

        def z(alpha):
            return scipy.stats.norm.ppf(alpha)

        expected_mean, expected_std, queries, observations = moments

        if random_seed is None:
            random_seed = 123
        utils.seed(random_seed)

        predictions = mh.infer(
            queries,
            observations,
            num_samples,
            num_chains=num_chains,
            num_adaptive_samples=num_adaptive_samples,
        )

        for i in range(predictions.num_chains):
            sample = predictions.get_chain(i)[queries[0]]
            mean, std = self.compute_statistics(sample)
            total_samples = tensor(sample.size())[0].item()
            n_eff = effective_sample_size(sample.unsqueeze(dim=0))

            # For out purposes, it seems more appropriate to use n_eff ONLY
            # to discount sample size. In particular, we should not allow
            # n_eff > total_samples

            n_eff = torch.min(n_eff, tensor(total_samples))

            # Hypothesis Testing
            # First, let's start by making sure that we can assume normalcy of means
            # pyre-fixme[16]: `AbstractConjugateTests` has no attribute
            #  `assertGreaterEqual`.
            self.assertGreaterEqual(
                total_samples, 30, msg="Sample size too small for normalcy assumption"
            )
            self.assertGreaterEqual(
                torch.min(n_eff).item(),
                30,
                msg="Effective sample size too small for normalcy assumption",
            )
            # Second, let us check the means using confidence intervals:
            lower_bound, upper_bound = mean_equality_hypothesis_confidence_interval(
                expected_mean, expected_std, n_eff, alpha
            )
            below_upper = torch.min(lower_bound <= mean).item()
            above_lower = torch.min(mean <= upper_bound).item()
            accept_interval = below_upper and above_lower
            message = "abs(mean - expected_mean) * sqr(n_eff) / expected_std = " + str(
                torch.abs(mean - expected_mean) / (expected_std / np.sqrt(n_eff))
            )
            message = (
                " alpha = "
                + str(alpha)
                + " z_alpha/2 = "
                + str(z(1 - alpha / 2))
                + " => "
                + message
            )
            message = (
                str(lower_bound)
                + " <= "
                + str(mean)
                + " <= "
                + str(upper_bound)
                + ". "
                + message
            )
            message = (
                "Mean outside confidence interval.\n"
                + "n_eff = "
                + str(n_eff)
                + ".\nExpected: "
                + message
            )
            # pyre-fixme[16]: `AbstractConjugateTests` has no attribute
            #  `assertTrue`.
            self.assertTrue(accept_interval, msg=message)
            # Third, let us check the variance using confidence intervals:
            lower_bound, upper_bound = variance_equality_hypothesis_confidence_interval(
                expected_std, n_eff - 1, alpha
            )
            below_upper = torch.min(lower_bound <= std).item()
            above_lower = torch.min(std <= upper_bound).item()
            accept_interval = below_upper and above_lower
            message = "(n_eff - 1) * (std/ expected_std) ** 2 = " + str(
                (n_eff - 1) * (std / expected_std) ** 2
            )
            message = (
                " alpha = "
                + str(alpha)
                + " chi2_alpha/2 = "
                + str(chi2(alpha / 2, n_eff - 1))
                + " <= "
                + message
                + " <= "
                + " chi2_(1-alpha/2) = "
                + str(chi2(1 - alpha / 2, n_eff - 1))
            )
            message = (
                str(lower_bound)
                + " <= "
                + str(std)
                + " <= "
                + str(upper_bound)
                + ". "
                + message
            )
            message = (
                "Standard deviation outside confidence interval.\n"
                + "n_eff = "
                + str(n_eff)
                + ".\nExpected: "
                + message
            )
            self.assertTrue(accept_interval, msg=message)
            continue

    def beta_binomial_conjugate_run(
        self,
        mh: BaseInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small beta binomial model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_beta_binomial_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            random_seed,
            num_adaptive_samples,
        )

    def gamma_gamma_conjugate_run(
        self,
        mh: BaseInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small gamma gamma model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_gamma_gamma_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            random_seed,
            num_adaptive_samples,
        )

    def gamma_normal_conjugate_run(
        self,
        mh: BaseInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small gamma normal model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_gamma_normal_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            random_seed,
            num_adaptive_samples,
        )

    def normal_normal_conjugate_run(
        self,
        mh: BaseInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small normal normal model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_normal_normal_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
            random_seed,
            num_adaptive_samples,
        )

    def dirichlet_categorical_conjugate_run(
        self,
        mh: BaseInference,
        num_chains: int = 1,
        num_samples: int = 1000,
        random_seed: Optional[int] = 17,
        num_adaptive_samples: int = 0,
    ):
        """
        Tests the inference run for a small dirichlet categorical model.

        :param mh: inference algorithm
        :param num_samples: number of samples
        :param num_chains: number of chains
        :param random_seed: seed for pytorch random number generator
        """
        moments = self.compute_dirichlet_categorical_moments()
        self._compare_run(
            moments,
            mh,
            num_chains,
            num_samples,
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
    def test_dirichlet_categorical_conjugate_run(self):
        """
        To be implemented for all classes extending AbstractConjugateTests.
        """
        raise NotImplementedError(
            "Conjugate test must implement test_categorical_dirichlet_conjugate_run."
        )
