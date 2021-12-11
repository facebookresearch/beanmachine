# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pandas as pd

from .configs import InferConfig, ModelConfig
from .null_mixture_model import NullMixtureMixedEffectModel


class HME:
    """The Hierarchical Mixed Effect model interface.

    :param data: observed train data
    :param model_config: HME model configuration parameters
    """

    def __init__(self, data: pd.DataFrame, model_config: ModelConfig) -> None:
        self.model = NullMixtureMixedEffectModel(data, model_config)
        self.posterior_samples = None
        self.posterior_diagnostics = None

    def infer(self, infer_config: InferConfig) -> Tuple[pd.DataFrame]:
        """Performs MCMC posterior inference on HME model parameters and
        returns MCMC samples for those parameters registered in the query.

        :param infer_config: configuration settings of posterior inference
        :return: posterior samples and their diagnostic summary statistics
        """

        self.posterior_samples, self.posterior_diagnostics = self.model.infer(
            infer_config
        )
        return self.posterior_samples, self.posterior_diagnostics

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Computes predictive distributions on the new test data according to
        MCMC posterior samples.

        :param new_data: test data for prediction
        :return: predictive distributions on the new test data
        """

        return self.model.predict(new_data, self.posterior_samples)
