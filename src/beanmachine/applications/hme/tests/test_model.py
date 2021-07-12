# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import pandas as pd
import pytest
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


@pytest.fixture
def infer_config():
    return InferConfig(n_iter=2000, n_warmup=10000)


@pytest.fixture
def mean_config():
    return RegressionConfig(
        distribution="normal",
        outcome="yi",
        stderr="sei",
        formula="~ 1 + (1|team)",
        link="identity",
        random_effect_distribution="normal",
    )


@pytest.mark.parametrize(
    "formula, fixed_effects, random_effects",
    [
        (
            "y ~ 1 + x + (1|a) + (1|b/c) + (1|d+e)",
            ["1", "x"],
            ["a", "b", ("b", "c"), ("d", "e")],
        ),
    ],
)
def test_parse_formula(formula, fixed_effects, random_effects):
    feff, reff = AbstractModel.parse_formula(formula=formula)
    assert feff == fixed_effects and reff == random_effects


@pytest.mark.parametrize(
    "data, mixture_config, param_assert, expected_sample_shape, expected_diag_shape, expected_param_value",
    [
        (
            pd.DataFrame(
                {
                    "yi": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "sei": [0.15] * 6,
                    "group": ["a"] * 3 + ["b"] * 3,
                    "team": ["x", "y"] * 3,
                }
            ),
            MixtureConfig(use_null_mixture=False),
            "yhat",
            (4000, 12),
            (10, 5),
            tensor([0.2, 0.3] * 3, dtype=torch.float32),
        ),
        (
            pd.DataFrame(
                {
                    "yi": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "sei": [0.15] * 6,
                    "group": ["a"] * 3 + ["b"] * 3,
                    "team": ["x", "y"] * 3,
                }
            ),
            MixtureConfig(use_null_mixture=True),
            "mu_H1",
            (4000, 19),
            (17, 5),
            tensor([0.3, 0.4] * 3, dtype=torch.float32),
        ),
    ],
)
def test_model_infer(
    data,
    mean_config,
    mixture_config,
    infer_config,
    param_assert,
    expected_sample_shape,
    expected_diag_shape,
    expected_param_value,
):
    model = HME(
        data,
        ModelConfig(
            mean_regression=mean_config,
            mean_mixture=mixture_config,
        ),
    )
    post_samples, post_diagnostics = model.infer(infer_config)
    assert (
        post_samples.shape == expected_sample_shape
        and post_diagnostics.shape == expected_diag_shape
    )

    est = tensor(
        post_samples[
            [key for key in model.model.query_map.keys() if param_assert in key]
        ].mean(axis=0),
        dtype=torch.float32,
    )
    assert torch.allclose(est, expected_param_value, atol=0.1)
