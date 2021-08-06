# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import pandas as pd
import torch
from beanmachine.applications.hme import (
    HME,
    InferConfig,
    MixtureConfig,
    ModelConfig,
    RegressionConfig,
)
from beanmachine.applications.hme.abstract_model import AbstractModel
from torch import tensor


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data1 = pd.DataFrame(
            {
                "yi": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "sei": [0.15] * 6,
                "group": ["a"] * 3 + ["b"] * 3,
                "team": ["x", "y"] * 3,
            }
        )
        data2 = cls.data1.copy()
        data2["yi"] = -1 * data2["yi"]
        cls.data2 = pd.concat([cls.data1, data2], axis=0).reset_index(drop=True)

        cls.data_v = pd.DataFrame(
            {
                "vi": [0.9, 1.0, 1.1, 1.9, 2.0, 2.1],
                "group": ["a"] * 3 + ["b"] * 3,
                "team": ["x", "y"] * 3,
            }
        )
        cls.infer_config = InferConfig(n_iter=2000, n_warmup=10000)
        cls.mean_config = RegressionConfig(
            distribution="normal",
            outcome="yi",
            stderr="sei",
            formula="~ 1 + (1|team)",
            link="identity",
        )

    def test_parse_formula(self) -> None:
        feff, reff = AbstractModel.parse_formula(
            "y ~ 1 + x + (1|a) + (1|b/c) + (1|d+e)"
        )
        self.assertListEqual(feff, ["1", "x"])
        self.assertListEqual(reff, ["a", "b", ("b", "c"), ("d", "e")])

    def test_model_a_infer(self) -> None:
        # fit a random effect model without null mixture on data1
        model = HME(
            self.data1,
            ModelConfig(
                mean_regression=self.mean_config,
                mean_mixture=MixtureConfig(use_null_mixture=False),
            ),
        )
        post_samples, post_diag = model.infer(self.infer_config)

        self.assertTupleEqual(post_samples.shape, (4000, 12))
        self.assertTupleEqual(post_diag.shape, (10, 5))
        est = tensor(
            post_samples[
                [key for key in model.model.query_map.keys() if "yhat" in key]
            ].mean(axis=0),
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(est, tensor([0.2, 0.3] * 3, dtype=torch.float32), atol=0.1)
        )

    def test_model_b_infer(self) -> None:
        # fit a HME model with null mixture on data1
        model = HME(
            self.data1,
            ModelConfig(
                mean_regression=self.mean_config,
                mean_mixture=MixtureConfig(use_null_mixture=True),
            ),
        )
        post_samples, post_diag = model.infer(self.infer_config)

        self.assertTupleEqual(post_samples.shape, (4000, 19))
        self.assertTupleEqual(post_diag.shape, (17, 5))
        est = tensor(
            post_samples[
                [key for key in model.model.query_map.keys() if "mu_H1" in key]
            ].mean(axis=0),
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(est, tensor([0.3, 0.4] * 3, dtype=torch.float32), atol=0.1)
        )

    def test_model_c_infer(self) -> None:
        # fit a HME model with null mixture on data2
        model = HME(
            self.data2,
            ModelConfig(
                mean_regression=self.mean_config,
                mean_mixture=MixtureConfig(use_null_mixture=True),
            ),
        )
        post_samples, post_diag = model.infer(self.infer_config)

        self.assertTupleEqual(post_samples.shape, (4000, 31))
        self.assertTupleEqual(post_diag.shape, (29, 5))
        est = tensor(
            post_samples[
                [key for key in model.model.query_map.keys() if "mu_H1" in key]
            ]
            .mean(axis=0)
            .abs()
        )
        self.assertTrue(all(x < 0.2 for x in est))

    def test_model_d_infer(self) -> None:
        # fit a HME model with null mixture + bi-modal H1 on data2
        model = HME(
            self.data2,
            ModelConfig(
                mean_regression=self.mean_config,
                mean_mixture=MixtureConfig(
                    use_null_mixture=True,
                    use_bimodal_alternative=True,
                    use_asymmetric_modes=True,
                    use_partial_asymmetric_modes=True,
                ),
            ),
        )
        post_samples, post_diag = model.infer(self.infer_config)

        self.assertTupleEqual(post_samples.shape, (4000, 35))
        self.assertTupleEqual(post_diag.shape, (33, 5))
        est = tensor(
            post_samples[
                [key for key in model.model.query_map.keys() if "mu_H1" in key]
            ].median(axis=0),
            dtype=torch.float32,
        )
        pair_sum = est.reshape(2, -1).sum(axis=0).abs()
        self.assertTrue(all(x < 0.5 for x in pair_sum))
