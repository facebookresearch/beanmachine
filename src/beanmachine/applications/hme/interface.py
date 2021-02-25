# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Tuple

import pandas as pd

from .configs import InferConfig, ModelConfig
from .null_mixture_model import NullMixtureMixedEffectModel


class HME:
    """
    The Hierarchical Mixed Effect model interface.
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        self.model = NullMixtureMixedEffectModel(data, model_config)
        self.posterior_samples = None
        self.posterior_diagnostics = None

    def infer(self, infer_config: InferConfig) -> Tuple[pd.DataFrame]:
        self.posterior_samples, self.posterior_diagnostics = self.model.infer(
            infer_config
        )
        return self.posterior_samples, self.posterior_diagnostics
