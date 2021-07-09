# Copyright(C) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Tuple

import pandas as pd

from .configs import InferConfig, ModelConfig
from .null_mixture_model import NullMixtureMixedEffectModel


class HME:
    """The Hierarchical Mixed Effect model interface.

    :param data: observed train data
    :type data: class:`pd.DataFrame`
    :param model_config: HME model configuration parameters
    :type model_config: class:`ModelConfig`
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        self.model = NullMixtureMixedEffectModel(data, model_config)
        self.posterior_samples = None
        self.posterior_diagnostics = None

    def infer(self, infer_config: InferConfig) -> Tuple[pd.DataFrame]:
        """Performs MCMC posterior inference on HME model parameters and
        returns MCMC samples for those parameters registered in the query.

        :param infer_config: configuration settings of posterior inference
        :type infer_config: class:`InferConfig`
        :return: posterior samples and their diagnostic summary statistics
        :rtype: (class:`pd.DataFrame`, class:`pd.DataFrame`)
        """

        self.posterior_samples, self.posterior_diagnostics = self.model.infer(
            infer_config
        )
        return self.posterior_samples, self.posterior_diagnostics
