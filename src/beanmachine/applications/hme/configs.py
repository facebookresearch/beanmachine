# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RegressionConfig:
    distribution: str = "normal"
    outcome: str = "y"
    stderr: str = None
    formula: str = "1"
    link: str = "identity"
    random_effect_distribution: str = "normal"

    def __post_init__(self):
        if self.distribution not in ["normal", "bernoulli"]:
            raise ValueError("distribution must be normal or bernoulli")
        if self.link not in ["identity", "logit"]:
            raise ValueError("link function must be identity or logit")
        if self.random_effect_distribution not in ["normal", "t"]:
            raise ValueError("random_effect_distribution must be normal or t")


@dataclass
class MixtureConfig:
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
    distribution: str = "flat"
    support: str = "real"
    parameters: List[Any] = field(default_factory=list)


@dataclass
class ModelConfig:
    mean_regression: RegressionConfig = field(default_factory=RegressionConfig)
    mean_mixture: MixtureConfig = field(default_factory=MixtureConfig)
    priors: Dict[str, PriorConfig] = field(default_factory=dict)


@dataclass
class InferConfig:
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
