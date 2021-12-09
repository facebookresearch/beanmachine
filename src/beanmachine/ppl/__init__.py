# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributions import Distribution

from . import experimental
from .diagnostics import Diagnostics
from .diagnostics.common_statistics import effective_sample_size, r_hat, split_r_hat
from .inference import (
    CompositionalInference,
    GlobalHamiltonianMonteCarlo,
    GlobalNoUTurnSampler,
    RejectionSampling,
    SingleSiteAncestralMetropolisHastings,
    SingleSiteHamiltonianMonteCarlo,
    SingleSiteNewtonianMonteCarlo,
    SingleSiteNoUTurnSampler,
    SingleSiteRandomWalk,
    SingleSiteUniformMetropolisHastings,
    empirical,
    seed,
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
    "seed",
    "param",
    "r_hat",
    "random_variable",
    "simulate",
    "split_r_hat",
]
