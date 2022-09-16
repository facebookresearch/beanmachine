#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pandas as pd
from beanmachine.applications.causal_inference.did.did_model import (
    BayesianDiffInDiff,
    PriorConfigSetting,
)
from beanmachine.applications.causal_inference.did.exceptions import (
    ModelNotFitException,
    PriorConfigException,
)


class DidModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_df = pd.DataFrame(
            {
                "response": [1, 2, 3, 4],
                "timestamp": ["1", "1", "2", "2"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 0],
                "interaction": [0, 1, 0, 0],
            }
        )
        self.data_df_small_responses = pd.DataFrame(
            {
                "response": [0.1, 0.2, 0.3, 0.4],
                "timestamp": ["1", "1", "2", "2"],
                "is_after_intervention": [0, 1, 0, 1],
                "is_treatment_group": [0, 1, 1, 0],
                "interaction": [0, 1, 0, 0],
            }
        )

    def test_set_data(self):
        model = BayesianDiffInDiff(
            data=self.data_df,
            intervention_key="2",
            pre_period_keys=["1"],
            post_period_keys=["2"],
        )
        self.assertTrue(model._data.df.equals(self.data_df))

    def test_model_fit_empirical_bayes(self):
        model = BayesianDiffInDiff(
            data=self.data_df,
            intervention_key="2",
            pre_period_keys=["1"],
            post_period_keys=["2"],
        )
        ground_truth_treatment_effect = -4.0
        self.assertAlmostEqual(
            model.fit(
                n_warmup=50, n_samples=100, priors=PriorConfigSetting.EMPIRICAL_BAYES
            ).get_treatment_effect()["mean"],
            ground_truth_treatment_effect,
            places=0,
        )
        model.get_posterior_samples()
        model.get_diagnostics()
        model.get_treatment_effect()
        model.get_model_params()

    def test_model_fit_flat(self):
        model = BayesianDiffInDiff(
            data=pd.concat([self.data_df] * 10, ignore_index=True),
            intervention_key="2",
            pre_period_keys=["1"],
            post_period_keys=["2"],
        )
        ground_truth_treatment_effect = -4.0
        self.assertAlmostEqual(
            model.fit(
                n_warmup=50, n_samples=100, priors=PriorConfigSetting.FLAT
            ).get_treatment_effect()["mean"],
            ground_truth_treatment_effect,
            places=0,
        )
        model.get_posterior_samples()
        model.get_diagnostics()
        model.get_treatment_effect()
        model.get_model_params()

    def test_model_fit_normal_centered_zero(self):
        model = BayesianDiffInDiff(
            data=pd.concat([self.data_df_small_responses] * 10, ignore_index=True),
            intervention_key="2",
            pre_period_keys=["1"],
            post_period_keys=["2"],
        )
        ground_truth_treatment_effect = -0.4
        self.assertAlmostEqual(
            model.fit(
                n_warmup=50,
                n_samples=100,
                priors=PriorConfigSetting.NORMAL_CENTERED_ZERO,
            ).get_treatment_effect()["mean"],
            ground_truth_treatment_effect,
            places=1,
        )
        model.get_posterior_samples()
        model.get_diagnostics()
        model.get_treatment_effect()
        model.get_model_params()

    def test_model_not_fit_exceptions(self):
        model = BayesianDiffInDiff(
            data=self.data_df,
            intervention_key="2",
            pre_period_keys=["1"],
            post_period_keys=["2"],
        )
        self.assertRaises(ModelNotFitException, model.get_posterior_samples)
        self.assertRaises(ModelNotFitException, model.get_diagnostics)
        self.assertRaises(ModelNotFitException, model.get_treatment_effect)
        self.assertRaises(ModelNotFitException, model.get_model_params)

    def test_prior_config_warning(self):
        model = BayesianDiffInDiff(
            data=self.data_df,
            intervention_key="2",
            pre_period_keys=["1"],
            post_period_keys=["2"],
        )
        self.assertRaises(
            PriorConfigException,
            model.fit,
            priors="PRIORDOESNOTEXIST",
            n_warmup=5,
            n_samples=10,
        )
