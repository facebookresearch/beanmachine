#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd
from beanmachine.applications.causal_inference.did.did_data import DiDData
from beanmachine.applications.causal_inference.did.exceptions import (
    DataColumnException,
    InterventionException,
    PostPeriodException,
    PrePeriodException,
)


class DidDataTest(unittest.TestCase):
    def setUp(self):
        self.data_dict = {
            "response": [1, 2, 3, 4],
            "timestamp": ["1", "2", "3", "4"],
            "is_after_intervention": [0, 1, 0, 1],
            "is_treatment_group": [0, 1, 1, 0],
            "interaction": [0, 1, 0, 0],
        }
        self.test_correct_df = pd.DataFrame(self.data_dict)
        self.ex_pre_period_keys = ["1", "2"]
        self.ex_intervention = "3"
        self.ex_post_period_days = ["3", "4"]

    def test_set_df(self):
        actual_df = DiDData(
            self.test_correct_df,
            self.ex_pre_period_keys,
            self.ex_intervention,
            self.ex_post_period_days,
        ).df
        pd.testing.assert_frame_equal(actual_df, self.test_correct_df)

    def test_column_names_df(self):
        for col in self.data_dict.keys():
            subset_col_df = pd.DataFrame(
                {
                    key: self.data_dict[key]
                    for key in self.data_dict.keys()
                    if key != col
                }
            )
            self.assertRaises(
                DataColumnException,
                DiDData,
                subset_col_df,
                self.ex_pre_period_keys,
                self.ex_intervention,
                self.ex_post_period_days,
            )

    def test_df_tuple_combinations(self):
        incorrect_df_1 = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": ["1", "2", "3", "4"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 1],
                "interaction": [0, 1, 0, 1],
            }
        )
        incorrect_df_2 = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": ["1", "2", "3", "4"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 1.1],
                "interaction": [0, 1, 0, 1],
            }
        )
        self.assertRaises(
            DataColumnException,
            DiDData,
            incorrect_df_1,
            self.ex_pre_period_keys,
            self.ex_intervention,
            self.ex_post_period_days,
        )
        self.assertRaises(
            DataColumnException,
            DiDData,
            incorrect_df_2,
            self.ex_pre_period_keys,
            self.ex_intervention,
            self.ex_post_period_days,
        )

    def test_wrong_dtype_df(self):
        incorrect_df_1 = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": ["1", "2", "3", "4"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 0.0],
                "interaction": [0, 1, 0, 0],
            }
        )
        incorrect_df_2 = pd.DataFrame(
            {
                "response": ["1", "2", "3", 4],
                "timestamp": ["1", "2", "3", "4"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 1],
                "interaction": [0, 1, 0, 1],
            }
        )
        incorrect_df_3 = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": ["1", "2", "3", "4"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 0],
                "interaction": [0.0, 1.0, 0.0, 0.0],
            }
        )
        pre_period_keys = ["1"]
        intervention = ["2"]
        post_period_keys = ["2", "3"]

        self.assertRaises(
            DataColumnException,
            DiDData,
            incorrect_df_1,
            pre_period_keys,
            intervention,
            post_period_keys,
        )
        self.assertRaises(
            DataColumnException,
            DiDData,
            incorrect_df_2,
            pre_period_keys,
            intervention,
            post_period_keys,
        )
        self.assertRaises(
            DataColumnException,
            DiDData,
            incorrect_df_3,
            pre_period_keys,
            intervention,
            post_period_keys,
        )

    def test_datestamp_functionality(self):
        correct_df_datestamp = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": [
                    pd.Timestamp("2022-01-01"),
                    pd.Timestamp("2022-02-01"),
                    pd.Timestamp("2022-03-01"),
                    pd.Timestamp("2022-04-01"),
                ],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 0],
                "interaction": [0, 1, 0, 0],
            }
        )
        DiDData(
            correct_df_datestamp,
            pre_period_keys=[
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-02-01"),
            ],
            intervention_key=pd.Timestamp("2022-03-01"),
            post_period_keys=[
                pd.Timestamp("2022-03-01"),
                pd.Timestamp("2022-04-01"),
            ],
        )

    def test_full_constructor(self):
        correct_pre_period_keys = [
            "2022-01-01",
            "2022-02-01",
        ]
        correct_intervention = "2022-03-01"
        correct_post_period_keys = [
            "2022-03-01",
            "2022-04-01",
        ]
        correct_df_datestamp = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": [
                    "2022-01-01",
                    "2022-02-01",
                    "2022-03-01",
                    "2022-04-01",
                ],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 0],
                "interaction": [0, 1, 0, 0],
            }
        )
        data = DiDData(
            correct_df_datestamp,
            pre_period_keys=[
                "2022-01-01",
                "2022-02-01",
            ],
            intervention_key="2022-03-01",
            post_period_keys=[
                "2022-03-01",
                "2022-04-01",
            ],
        )
        self.assertEqual(data.pre_period_keys, correct_pre_period_keys)
        self.assertEqual(data.post_period_keys, correct_post_period_keys)
        self.assertEqual(data.intervention_key, correct_intervention)

    def test_fluent_interface(self):
        correct_pre_period_keys = [
            "1",
            "2",
        ]
        correct_intervention = "3"
        correct_post_period_keys = [
            "3",
            "4",
        ]
        data = (
            DiDData(
                self.test_correct_df,
                pre_period_keys=["1"],
                intervention_key="2",
                post_period_keys=["2", "3"],
            )
            ._set_pre_period_keys(correct_pre_period_keys)
            ._set_intervention_key(correct_intervention)
            ._set_post_period_keys(correct_post_period_keys)
        )
        self.assertEqual(data.pre_period_keys, correct_pre_period_keys)
        self.assertEqual(data.post_period_keys, correct_post_period_keys)
        self.assertEqual(data.intervention_key, correct_intervention)

    def test_get_fit_df(self):
        data = DiDData(
            self.test_correct_df,
            pre_period_keys=["2"],
            intervention_key="3",
            post_period_keys=["3"],
        )
        correct_fit_df = pd.DataFrame(
            {
                "response": [
                    2,
                    3,
                ],
                "is_after_intervention": [1, 0],
                "is_treatment_group": [1, 1],
                "interaction": [1, 0],
            }
        )
        pd.testing.assert_frame_equal(
            data.get_fit_df().reset_index(drop=True),
            correct_fit_df.reset_index(drop=True),
        )

    def test_out_of_sample_intervention(self):
        self.assertRaises(
            InterventionException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key="5",
            post_period_keys=self.ex_post_period_days,
        )
        self.assertRaises(
            InterventionException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key="0",
            post_period_keys=self.ex_post_period_days,
        )
        self.assertRaises(
            InterventionException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key="2.5",
            post_period_keys=self.ex_post_period_days,
        )

    def test_out_of_sample_pre_period(self):
        self.assertRaises(
            PrePeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=["5"],
            intervention_key=self.ex_intervention,
            post_period_keys=self.ex_post_period_days,
        )
        self.assertRaises(
            PrePeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=["0"],
            intervention_key=self.ex_intervention,
            post_period_keys=self.ex_post_period_days,
        )
        self.assertRaises(
            PrePeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=["2.5"],
            intervention_key=self.ex_intervention,
            post_period_keys=self.ex_post_period_days,
        )

    def test_out_of_sample_post_period(self):
        self.assertRaises(
            PostPeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key=self.ex_intervention,
            post_period_keys=["5"],
        )
        self.assertRaises(
            PostPeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key=self.ex_intervention,
            post_period_keys=["0"],
        )
        self.assertRaises(
            PostPeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key=self.ex_intervention,
            post_period_keys=["2.5"],
        )

    def test_pre_period_overlap_intervention(self):
        self.assertRaises(
            InterventionException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key="1",
            post_period_keys=self.ex_post_period_days,
        )
        self.assertRaises(
            InterventionException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key="2",
            post_period_keys=self.ex_post_period_days,
        )

    def test_pre_period_overlap_post_period(self):
        self.assertRaises(
            PostPeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key=self.ex_intervention,
            post_period_keys=["2"],
        )
        self.assertRaises(
            PostPeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key=self.ex_intervention,
            post_period_keys=["3", "4", "2"],
        )

    def test_pre_period_duplicates(self):
        self.assertRaises(
            PrePeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=["1", "2", "2"],
            intervention_key=self.ex_intervention,
            post_period_keys=self.ex_post_period_days,
        )

    def test_post_period_duplicates(self):
        self.assertRaises(
            PostPeriodException,
            DiDData,
            self.test_correct_df,
            pre_period_keys=self.ex_pre_period_keys,
            intervention_key=self.ex_intervention,
            post_period_keys=["3", "3", "4"],
        )
