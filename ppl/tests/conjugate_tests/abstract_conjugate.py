# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod

import torch.tensor as tensor
from beanmachine.ppl.examples.conjugate_models.beta_binomial import BetaBinomialModel
from beanmachine.ppl.examples.conjugate_models.categorical_dirichlet import (
    CategoricalDirichletModel,
)
from beanmachine.ppl.examples.conjugate_models.gamma_gamma import GammaGammaModel
from beanmachine.ppl.examples.conjugate_models.gamma_normal import GammaNormalModel
from beanmachine.ppl.examples.conjugate_models.normal_normal import NormalNormalModel


class AbstractConjugateTests(metaclass=ABCMeta):
    """
     Computes the posterior mean and standard deviation of some of the conjugate
     distributions included below.
     https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions
    """

    def compute_statistics(self, predictions):
        """
        Computes mean and standard deviation of a given tensor of samples.

        :param predictions: tensor of samples
        :returns: mean and standard deviation of the tensor of samples.
        """
        return predictions.mean(0), predictions.var(0).pow(0.5)

    def compute_beta_binomial_moments(self, num_samples, inference):
        """
        Computes mean and standard deviation of a small beta binomial model.

        :param num_samples: number of samples to generate from the posterior.
        :param inference: inference algorithm class to run on the model.
        :return: mean, expected mean, standard deviation and expected standard
        deviation
        """
        alpha = tensor([2.0, 2.0])
        beta = tensor([1.0, 1.0])
        trials = tensor([1.0, 1.0])
        obs = tensor([1.0, 0.0])
        model = BetaBinomialModel(alpha, beta, trials)
        infer = inference([model.beta()], {model.binomial(): obs})
        predictions = infer.infer(num_samples)
        mean, std_dev = self.compute_statistics(predictions[model.beta()])
        alpha = alpha + obs
        beta = beta - obs + trials
        expected_mean = alpha / (alpha + beta)
        expected_std = (
            (alpha * beta) / (alpha + beta).pow(2.0) * (alpha + beta + 1.0)
        ).pow(0.5)

        return (mean, expected_mean, std_dev, expected_std)

    def compute_gamma_gamma_moments(self, num_samples, inference):
        """
        Computes mean and standard deviation of a small gamma gamma model.

        :param num_samples: number of samples to generate from the posterior.
        :param inference: inference algorithm class to run on the model.
        :return: mean, expected mean, standard deviation and expected standard
        deviation
        """
        shape = tensor([1.0, 1.0])
        rate = tensor([2.0, 2.0])
        alpha = tensor([0.5, 0.5])
        obs = tensor([2.0, 4.0])
        model = GammaGammaModel(shape, rate, alpha)
        infer = inference([model.gamma_p()], {model.gamma(): obs})
        predictions = infer.infer(num_samples)
        mean, std_dev = self.compute_statistics(predictions[model.gamma_p()])
        shape = shape + alpha
        rate = rate + obs
        expected_mean = shape / rate
        expected_std = (expected_mean / rate).pow(0.5)
        return (mean, expected_mean, std_dev, expected_std)

    def compute_gamma_normal_moments(self, num_samples, inference):
        """
        Computes mean and standard deviation of a small gamma normal model.

        :param num_samples: number of samples to generate from the posterior.
        :param inference: inference algorithm class to run on the model.
        :return: mean, expected mean, standard deviation and expected standard
        deviation
        """
        shape = tensor([1.0, 1.0])
        rate = tensor([2.0, 2.0])
        mu = tensor([1.0, 2.0])
        obs = tensor([1.5, 2.5])
        model = GammaNormalModel(shape, rate, mu)
        infer = inference([model.gamma()], {model.normal(): obs})
        predictions = infer.infer(num_samples)
        mean, std_dev = self.compute_statistics(predictions[model.gamma()])
        shape = shape + tensor([0.5, 0.5])
        deviations = (obs - mu).pow(2.0)
        deviations = deviations.sum(0)
        rate = rate + (deviations * (0.5))
        expected_mean = shape / rate
        expected_std = (expected_mean / rate).pow(0.5)
        return (mean, expected_mean, std_dev, expected_std)

    def compute_normal_normal_moments(self, num_samples, inference):
        """
        Computes mean and standard deviation of a small normal normal model.

        :param num_samples: number of samples to generate from the posterior.
        :param inference: inference algorithm class to run on the model.
        :return: mean, expected mean, standard deviation and expected standard
        deviation
        """
        mu = tensor([1.0, 1.0])
        std = tensor([1.0, 1.0])
        sigma = tensor([1.0, 1.0])
        obs = tensor([1.5, 2.5])
        model = NormalNormalModel(mu, std, sigma)
        infer = inference([model.normal_p()], {model.normal(): obs})
        predictions = infer.infer(num_samples)
        mean, std_dev = self.compute_statistics(predictions[model.normal_p()])
        expected_mean = (1 / (1 / sigma.pow(2.0) + 1 / std.pow(2.0))) * (
            mu / std.pow(2.0) + obs / sigma.pow(2.0)
        )
        expected_std = (std.pow(2.0) + sigma.pow(2.0)).pow(-0.5)
        return (mean, expected_mean, std_dev, expected_std)

    def compute_dirichlet_categorical_moments(self, num_samples, inference):
        """
        Computes mean and standard deviation of a small dirichlet categorical
        model.

        :param num_samples: number of samples to generate from the posterior.
        :param inference: inference algorithm class to run on the model.
        :return: mean, expected mean, standard deviation and expected standard
        deviation
        """
        alpha = tensor([0.5, 0.5])
        model = CategoricalDirichletModel(alpha)
        obs = tensor([1.0])
        infer = inference([model.dirichlet()], {model.categorical(): obs})
        predictions = infer.infer(num_samples)
        mean, std_dev = self.compute_statistics(predictions[model.dirichlet()])
        alpha = alpha + tensor([0.0, 1.0])
        expected_mean = alpha / alpha.sum()
        expected_std = expected_mean * (1 - expected_mean) / (alpha.sum() + 1)
        return mean, expected_mean, std_dev, expected_std

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
