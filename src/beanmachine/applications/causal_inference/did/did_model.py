# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

from enum import auto, Enum
from typing import Dict, Sequence, Union

import matplotlib

import numpy as np
import pandas as pd
from beanmachine.applications.hme.configs import (
    InferConfig,
    MixtureConfig,
    ModelConfig,
    PriorConfig,
    RegressionConfig,
)
from beanmachine.applications.hme.interface import HME

from .did_data import DiDData
from .exceptions import ModelNotFitException, PriorConfigException


class PriorConfigSetting(Enum):
    EMPIRICAL_BAYES = auto()
    NORMAL_CENTERED_ZERO = auto()
    FLAT = auto()


class BayesianDiffInDiff:
    def __init__(
        self,
        data: pd.DataFrame,
        pre_period_keys: Sequence[Union[str, int, float, pd.Timestamp]],
        intervention_key: Union[str, int, float, pd.Timestamp],
        post_period_keys: Sequence[Union[str, int, float, pd.Timestamp]],
    ) -> None:
        """
        Parameters
        ----------
        data: pd.DataFrame
            Dataframe to be used by the rest of the class functions
        pre_period_keys: Sequence[str, int, float, OR pd.Timestamp]
            Keys that denote the timestamps that occurred before the intervention key that will be considered when fitting.
        intervention_key: str, int, float, OR pd.Timestamp
            Key that denotes the timestamp that the intervention occurred. This intervention timestamp should occur after the pre-period range.
        post_period_keys: Sequence[str, int, float, OR pd.Timestamp]
            Keys that denote the timestamps that occurred after the intervention that will be considered when fitting.
        """
        self._is_fit = False
        self._posterior_samples = pd.DataFrame()
        self._diagnostics = pd.DataFrame()
        self._num_features = 3  # Note: This may change in the future if/when there is added support for additional covariates.
        self._data = DiDData(data, pre_period_keys, intervention_key, post_period_keys)

    def plot_parallel_trends_assumption(self) -> matplotlib.figure.Axes:
        """
        Plots data to assist with checking for parallel trends assumption.
        Produces a graph that shows response plotted against timestamps with
        color-coded pre- and post- periods and intervention timestamps.
        """
        return self._data.plot_parallel_trends_assumption()

    def fit(
        self,
        priors: PriorConfigSetting = PriorConfigSetting.NORMAL_CENTERED_ZERO,
        n_warmup: int = 500,
        n_samples: int = 1000,
    ) -> BayesianDiffInDiff:
        """
        Fits the model.

        Parameters
        ----------
        priors: PriorConfigSetting
            Currently supported prior configs include PriorConfigSetting.EMPIRICAL_BAYES and PriorConfigSetting.NORMAL_CENTERED_ZERO.
            NORMAL_CENTERED_ZERO priors denote Normal distributions centered around 0 with a scale of 2
            FLAT priors denotes a uniform distribution over all Real numbers
            EMPIRICAL_BAYES priors denote Normal distributions centered around maximum-likelihood estimates with a scale of 2.
        n_warmup: int
            Number of warm-up or adaptive samples
        n_samples: int
            Number of samples to draw from the posterior after the warm-up samples
        """
        fit_df = pd.DataFrame(self._data.get_fit_df().astype(float))
        outcome = "response"
        formula = "~ 1 + is_after_intervention + is_treatment_group + interaction"

        regression_config = RegressionConfig(
            distribution="normal",
            outcome=outcome,
            formula=formula,
            link="identity",
        )

        mixture_config = MixtureConfig(use_null_mixture=False)

        if priors == PriorConfigSetting.NORMAL_CENTERED_ZERO:
            model = HME(
                fit_df,
                ModelConfig(
                    mean_regression=regression_config, mean_mixture=mixture_config
                ),
            )
        elif priors == PriorConfigSetting.FLAT:
            priors_config = {
                "intercept": PriorConfig("flat", {}),
                "is_after_intervention": PriorConfig("flat", {}),
                "is_treatment_group": PriorConfig("flat", {}),
                "interaction": PriorConfig("flat", {}),
            }
            model = HME(
                fit_df,
                ModelConfig(
                    mean_regression=regression_config,
                    mean_mixture=mixture_config,
                    priors=priors_config,
                ),
            )
        elif priors == PriorConfigSetting.EMPIRICAL_BAYES:
            m00, m01, m10, m11 = tuple(
                fit_df.loc[
                    lambda df: (df["is_after_intervention"] == i[0])
                    & (df["is_treatment_group"] == i[1]),
                    "response",
                ].mean()
                for i in ((0, 0), (0, 1), (1, 0), (1, 1))
            )
            beta_0_est = m00
            beta_1_est = m10 - m00
            beta_2_est = m01 - m00
            beta_3_est = m11 - m01 - m10 + m00
            priors_config = {
                "intercept": PriorConfig("normal", {"mean": beta_0_est, "scale": 2}),
                "is_after_intervention": PriorConfig(
                    "normal", {"mean": beta_1_est, "scale": 2}
                ),
                "is_treatment_group": PriorConfig(
                    "normal", {"mean": beta_2_est, "scale": 2}
                ),
                "interaction": PriorConfig("normal", {"mean": beta_3_est, "scale": 2}),
            }
            model = HME(
                fit_df,
                ModelConfig(
                    mean_regression=regression_config,
                    mean_mixture=mixture_config,
                    priors=priors_config,
                ),
            )
        else:
            raise PriorConfigException(
                f"{priors} prior config not found or not supported.",
            )

        self._posterior_samples, self._diagnostics = model.infer(
            InferConfig(n_iter=n_samples, n_warmup=n_warmup, n_chains=4)
        )
        self._is_fit = True
        return self

    def get_model_params(self) -> pd.DataFrame:
        """
        Returns pandas DataFrame that includes the name, Mean, Acceptance, effective sample size (n_Eff),
        and R_hat values (used for convergence diagnostic).
        """
        if not self._is_fit:
            raise ModelNotFitException("Please fit model first.")
        param_mat = self._diagnostics[: self._num_features + 1].copy()
        param_mat.names = param_mat.names.str.replace("fixed_effect_", "")
        return param_mat

    def get_treatment_effect(
        self,
        credible_intervals: Sequence[float] = (2.5, 50.0, 97.5),
        return_mean: bool = True,
    ) -> Dict[str, float]:
        """
        Returns dictionary of credible intervals and (depending on flag) the mean of the treatment effect.

        Parameters
        ----------
        credible_intervals: Tuple[float, ...]
            Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
        return_mean: bool
            Flag to add mean of treatment effect in addition to credible intervals to the output dictionary.
        """
        if not self._is_fit:
            raise ModelNotFitException("Please fit model first.")
        results = dict(
            zip(
                [f"credible_interval_{i}" for i in credible_intervals],
                np.percentile(
                    self._posterior_samples.fixed_effect_interaction, credible_intervals
                ),
            )
        )
        if return_mean:
            results["mean"] = self._posterior_samples.fixed_effect_interaction.mean()
        return results

    def get_posterior_samples(self) -> pd.DataFrame:
        """
        Returns posterior samples of model. Will raise a ModelNotFitException() if model is not fit() first.
        """
        if not self._is_fit:
            raise ModelNotFitException("Please fit model first.")
        return self._posterior_samples

    def get_diagnostics(self) -> pd.DataFrame:
        """
        Returns diagnostics of model. Will raise a ModelNotFitException() if model is not fit() first.
        """
        if not self._is_fit:
            raise ModelNotFitException("Please fit model first.")
        return self._diagnostics
