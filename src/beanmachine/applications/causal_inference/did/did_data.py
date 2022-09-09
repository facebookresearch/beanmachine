# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .exceptions import (
    DataColumnException,
    InterventionException,
    PostPeriodException,
    PrePeriodException,
)


class DiDData:
    """
    A class to hold all data-related functions for BayesianDiffInDiff(). These
    include plotting functions, checking functions, and functions to return
    a filtered dataframe ready to be fit by the model.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pre_period_keys: Sequence[Union[str, int, float, pd.Timestamp]],
        intervention_key: Union[str, int, float, pd.Timestamp],
        post_period_keys: Sequence[Union[str, int, float, pd.Timestamp]],
    ) -> None:
        """
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be used by the rest of the class functions
        pre_period_keys: Sequence[str, int, float, OR pd.Timestamp]
            Keys that denote the timestamps that occurred before the intervention key that will be considered by the BayesianDiffInDiff() model when fitting.
        intervention_key: str, int, float, OR pd.Timestamp
            Key that denotes the timestamp that the intervention occurred. This intervention timestamp should occur after the pre-period range.
        post_period_keys: Sequence[str, int, float, OR pd.Timestamp]
            Keys that denote the timestamps that occurred after the intervention that will be considered by the BayesianDiffInDiff() model when fitting.
        """
        self._set_df(df)
        self._set_pre_period_keys(pre_period_keys)
        self._set_intervention_key(intervention_key)
        self._set_post_period_keys(post_period_keys)

    def _set_df(self, df: pd.DataFrame) -> DiDData:
        """
        Checks if valid dataframe and sets self.df to given pd.Dataframe. Returns self to allow for chaining methods.
        """
        self._check_if_valid_df(df)
        self.df = df
        return self

    def _set_intervention_key(
        self, intervention_key: Union[str, int, float, pd.Timestamp]
    ) -> DiDData:
        """
        Checks if valid intervention and sets self.intervention to given string. Returns self to allow for chaining methods.
        """
        self._check_if_valid_intervention_key(intervention_key)
        self.intervention_key = intervention_key
        return self

    def _set_pre_period_keys(
        self, pre_period_keys: Sequence[Union[str, int, float, pd.Timestamp]]
    ) -> DiDData:
        """
        Checks if valid sequence of pre-period keys and sets self.pre_period_keys to given sequence. Returns self to allow for chaining methods.
        """
        self._check_if_valid_pre_period_keys(pre_period_keys)
        self.pre_period_keys = pre_period_keys
        return self

    def _set_post_period_keys(
        self, post_period_keys: Sequence[Union[str, int, float, pd.Timestamp]]
    ) -> DiDData:
        """
        Checks if valid sequence of post-period keys and sets self.post_period_keys to given sequence. Returns self to allow for chaining methods.
        """
        self._check_if_valid_post_period_keys(post_period_keys)
        self.post_period_keys = post_period_keys
        return self

    def _check_if_valid_df(self, df: pd.DataFrame) -> None:
        necessary_columns_and_types = {
            "response": ["float", "int", "double"],
            "timestamp": [
                "str",
                "object",
                "int",
                "double",
                "float",
                "datetime64[ns]",
            ],
            "is_after_intervention": ["int", "bool"],
            "is_treatment_group": ["int", "bool"],
            "interaction": ["int", "bool"],
        }
        necessary_cols = set(necessary_columns_and_types.keys())
        if missing_cols := necessary_cols - set(df.columns):
            raise DataColumnException(
                f"Dataframe does not contain the necessary columns: [{', '.join(missing_cols)}]"
            )
        # This is to accomodate the case of converting type 'String' (often the output of PVC and bb queries) to python-readable str type
        if str(df.timestamp.dtype) == "string":
            df.timestamp = df.timestamp.astype(str)
        dtypes = df.dtypes.to_dict()
        for col_name, typ in dtypes.items():
            if typ not in necessary_columns_and_types[col_name]:
                raise DataColumnException(
                    f"Dataframe column {col_name} is type {typ} when type {necessary_columns_and_types[col_name]} is required."
                )
        unique_dummy_tuples = list(
            df.groupby(["is_after_intervention", "is_treatment_group"]).groups
        )
        unique_dummy_tuples.sort()
        necessary_dummy_tuples = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if unique_dummy_tuples != necessary_dummy_tuples:
            raise DataColumnException(
                "Dataframe does not contain all necessary combinations of pre- vs. post-intervention, and control vs. treatment group."
            )

    def _check_if_valid_intervention_key(
        self, intervention_key: Union[str, int, float, pd.Timestamp]
    ) -> None:
        if isinstance(intervention_key, pd.Timestamp):
            intervention_stamp = intervention_key.to_datetime64()
        else:
            intervention_stamp = intervention_key
        if intervention_stamp not in self.df.timestamp.values:
            raise InterventionException(
                f"Intervention timestamp {intervention_stamp} not found in dataset provided."
            )
        if intervention_stamp in self.pre_period_keys:
            raise InterventionException(
                f"Intervention timestamp {intervention_stamp} in pre-period timestamp range [{', '.join(self.pre_period_keys)}]."
            )

    def _check_if_valid_pre_period_keys(
        self, pre_period_keys: Sequence[Union[str, int, float, pd.Timestamp]]
    ) -> None:
        if all(isinstance(key, pd.Timestamp) for key in pre_period_keys):
            pre_period_stamps = {
                pd.Timestamp(key).to_datetime64() for key in pre_period_keys
            }
        else:
            pre_period_stamps = set(pre_period_keys)
        if missing_cols := pre_period_stamps - set(self.df.timestamp.values):
            raise PrePeriodException(
                f"Pre-period key(s) {missing_cols} not found in dataset provided."
            )
        if len(pre_period_keys) != len(pre_period_stamps):
            raise PrePeriodException("Duplicates found in pre-period keys.")

    def _check_if_valid_post_period_keys(
        self, post_period_keys: Sequence[Union[str, int, float, pd.Timestamp]]
    ) -> None:
        if all(isinstance(key, pd.Timestamp) for key in post_period_keys):
            post_period_stamps = {
                pd.Timestamp(key).to_datetime64() for key in post_period_keys
            }
        else:
            post_period_stamps = set(post_period_keys)
        if missing_cols := post_period_stamps - set(self.df.timestamp.values):
            raise PostPeriodException(
                f"Post-period key(s) {missing_cols} not found in dataset provided."
            )
        if overlapped_keys := post_period_stamps.intersection(
            set(self.pre_period_keys)
        ):
            raise PostPeriodException(
                f"Post-period key(s) {overlapped_keys} found in pre-period key selection."
            )
        if len(post_period_keys) != len(post_period_stamps):
            raise PostPeriodException("Duplicates found in post-period keys.")

    def plot_parallel_trends_assumption(
        self,
        y_min: Optional[int] = None,
        y_max: Optional[int] = None,
        legend_loc: str = "upper left",
    ) -> matplotlib.figure.Axes:
        """
        Plots data to assist with checking for parallel trends assumption.
        Produces a graph that shows response plotted against timestamps with
        color-coded pre- and post- periods and intervention timestamps.
        """
        sorted_timestamps = sorted(self.df["timestamp"].unique())
        self.df.sort_values("timestamp", inplace=True)
        if y_min is None:
            y_min = self.df["response"].mean() - (self.df["response"].std())
        if y_max is None:
            y_max = self.df["response"].mean() + (self.df["response"].std())

        sns.lineplot(
            x="timestamp", y="response", hue="is_treatment_group", data=self.df
        )
        plt.vlines(
            x=self.intervention_key,
            ymin=y_min,
            ymax=y_max,
            colors="purple",
            ls="--",
            lw=2,
            label="intervention_timestamp",
        )
        plt.axvspan(
            min(self.pre_period_keys),
            self.intervention_key,
            alpha=0.1,
            color="red",
            label="pre_intervention_period",
        )
        ind = sorted_timestamps.index(max(self.post_period_keys))
        plt.axvspan(
            max(self.post_period_keys),
            sorted_timestamps[ind + 1],
            alpha=0.1,
            color="blue",
            label="post_intervention_period",
        )
        plt.legend(loc=legend_loc)
        return plt.gca()

    def get_fit_df(self) -> pd.DataFrame:
        """
        Returns self.df filtered by just pre- and post- periods, which will be input for the BayesianDiffInDiff fit() function.
        """
        return self.df.loc[
            lambda df: df["timestamp"].isin(self.pre_period_keys)
            | df["timestamp"].isin(self.post_period_keys)
        ][["response", "is_after_intervention", "is_treatment_group", "interaction"]]
