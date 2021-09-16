# Copyright (c) Facebook, Inc. and its affiliates.
from torch.distributions import Distribution

from . import experimental
from .diagnostics import Diagnostics
from .diagnostics.common_statistics import effective_sample_size, r_hat, split_r_hat
from .inference import (
    CompositionalInference,
    GlobalHamiltonianMonteCarlo,
    GlobalNoUTurnSampler,
    Predictive,
    RejectionSampling,
    SingleSiteAncestralMetropolisHastings,
    SingleSiteHamiltonianMonteCarlo,
    SingleSiteNewtonianMonteCarlo,
    SingleSiteNoUTurnSampler,
    SingleSiteRandomWalk,
    SingleSiteUniformMetropolisHastings,
    empirical,
    simulate,
)
from .model import (
    functional,
    get_beanmachine_logger,
    param,
    random_variable,
    RVIdentifier,
)


LOGGER = get_beanmachine_logger()
# TODO(@neerajprad): Remove once T81756389 is fixed.
Distribution.set_default_validate_args(False)

__all__ = [
    "CompositionalInference",
    "Diagnostics",
    "GlobalHamiltonianMonteCarlo",
    "GlobalNoUTurnSampler",
    "Predictive",
    "RejectionSampling",
    "RVIdentifier",
    "SingleSiteAncestralMetropolisHastings",
    "SingleSiteHamiltonianMonteCarlo",
    "SingleSiteNewtonianMonteCarlo",
    "SingleSiteNoUTurnSampler",
    "SingleSiteRandomWalk",
    "SingleSiteUniformMetropolisHastings",
    "effective_sample_size",
    "empirical",
    "experimental",
    "functional",
    "param",
    "r_hat",
    "random_variable",
    "simulate",
    "split_r_hat",
]
