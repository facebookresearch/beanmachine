# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Dict, List


@dataclass
class RegressionConfig:
    """
    A configuration class for regression model specifications.

    :param distribution: distribution family of outcome variable conditional on mixed effects, defaults to 'normal'
    :param outcome: column name of outcome, defaults to 'y'
    :param stderr: column name of standard deviation of afore-mentioned distribution family, defaults to None
    :param formula: statistical formula establishing the relationship between outcome and covariates, defaults to '1'
    :param link: link function linking the systematic component (mixed effects) and outcome y, defaults to 'identity'
    """

    distribution: str = "normal"
    outcome: str = "y"
    stderr: str = None
    formula: str = "1"
    link: str = "identity"

    def __post_init__(self):
        if self.distribution not in ["normal", "bernoulli"]:
            raise ValueError("distribution must be normal or bernoulli")
        if self.link not in ["identity", "logit"]:
            raise ValueError("link function must be identity or logit")


@dataclass
class MixtureConfig:
    """
    A configuration class for mixture components.

    :param use_null_mixture: flag of whether the outcome is a mixture of null (0 effect) and alternative (non-zero effect), defaults to True
    :param use_bimodal_alternative: flag of whether to consider H1 model as a mixture of positive (H1+) and negative (H1-) effects, defaults to False
    :param use_asymmetric_modes: flag of whether H1+ and H1- are asymmetric around zero, defaults to False
    :param use_partial_asymmetric_modes: flag of whether H1+ shares the same scale parameters of the random effects with H1-, defaults to False
    :param null_prob_regression: model configurations for null mixture probability parameter (also known as movability)
    :param sign_prob_regression: model configurations for bimodal sign probability parameter
    """

    use_null_mixture: bool = True
    use_bimodal_alternative: bool = False
    use_asymmetric_modes: bool = False
    use_partial_asymmetric_modes: bool = False
    null_prob_regression: RegressionConfig = field(default_factory=RegressionConfig)
    sign_prob_regression: RegressionConfig = field(default_factory=RegressionConfig)

    def __post_init__(self):
        self.use_bimodal_alternative = (
            self.use_bimodal_alternative and self.use_null_mixture
        )
        self.use_asymmetric_modes = (
            self.use_asymmetric_modes and self.use_bimodal_alternative
        )
        self.use_partial_asymmetric_modes = (
            self.use_partial_asymmetric_modes and self.use_asymmetric_modes
        )


@dataclass
class PriorConfig:
    """
    A configuration class for prior distributions.

    :param distribution: prior distribution family, defaults to flat prior
    :param parameters: parameter values of prior distribution family
    """

    distribution: str = "flat"
    parameters: Dict[str, Union[float, PriorConfig]] = field(default_factory=dict)


@dataclass
class StructuredPriorConfig:
    """
    A configuration class for structured prior distributions.

    :param specification: type (e.g., AR, RW, etc) of the structured prior
    :param category_order: list of categorical covariate levels, specifying its ordinal structure
    """

    specification: str
    category_order: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """
    A configuration class for integrated models. E.g.::

        ModelConfig(
            mean_regression = RegressionConfig(
                distribution="normal",
                outcome="y",
                stderr=None,
                formula="y ~ 1 + (1|x)",
                link="identity",
            ),
            mean_mixture = MixtureConfig(
                use_null_mixture=True,
                use_bimodal_alternative=True,
                use_asymmetric_modes=True,
                use_partial_asymmetric_modes=True,
            ),
            priors = {
                "fe": PriorConfig('normal', {"mean": 0.0, "scale": 1.0}),
                }
        )

    :param mean_regression: regression model configurations for fixed and random effects components
    :param mean_mixture:  configuration settings for mixture components
    :param priors: a dictionary mapping model parameters to their prior distribution configurations
    """

    mean_regression: RegressionConfig = field(default_factory=RegressionConfig)
    mean_mixture: MixtureConfig = field(default_factory=MixtureConfig)
    priors: Dict[str, PriorConfig] = field(default_factory=dict)


@dataclass
class InferConfig:
    """
    A configuration class for MCMC posterior inference.

    :param n_iter: total number of posterior inference iterations
    :param n_warmup: number of posterior inference iterations to discard as burn-in or warm-up period
    :param algorithm: MCMC posterior inference method, defaults to 'NMC'
    :param n_chains: number of independent Markov chains running in parallel, defaults to 2
    :param queries: a set of model parameters to store posterior MCMC to be returned to user
    :param keep_warmup: flag of whether to keep the first n_warmup posterior samples, defaults to False
    :param keep_logprob: flag of whether to keep track of log-likelihood along MCMC inference, defaults to False
    :param seed: an integer number that controls the pattern of randomness, defaults to 5123401
    """

    n_iter: int
    n_warmup: int
    algorithm: str = "NMC"
    n_chains: int = 2
    queries: List[str] = field(default_factory=list)
    keep_warmup: bool = False
    keep_logprob: bool = False
    seed: int = 5123401

    def __post_init__(self):
        if self.algorithm not in ["NMC"]:
            raise ValueError("algorithm must be NMC")
