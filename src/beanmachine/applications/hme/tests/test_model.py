# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from torch import tensor


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "yi": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "sei": [0.15] * 6,
            "group": ["a"] * 3 + ["b"] * 3,
            "team": ["x", "y"] * 3,
        }
    )


@pytest.fixture
def mean_config():
    return RegressionConfig(
        distribution="normal",
        outcome="yi",
        stderr="sei",
        formula="~ 1 + (1|team)",
        link="identity",
    )


@pytest.mark.parametrize(
    "mixture_config, param_assert, expected_sample_shape, expected_diag_shape, expected_param_value",
    [
        (
            MixtureConfig(use_null_mixture=False),
            "yhat",
            (4000, 12),
            (10, 5),
            tensor([0.2, 0.3] * 3, dtype=torch.float32),
        ),
        (
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
    post_samples, post_diagnostics = model.infer(
        InferConfig(n_iter=2000, n_warmup=10000, seed=0)
    )
    assert post_samples.shape == expected_sample_shape
    assert post_diagnostics.shape == expected_diag_shape

    actual = tensor(
        post_samples[
            [key for key in model.model.query_map.keys() if param_assert in key]
        ].mean(axis=0),
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected_param_value, atol=0.1)


@pytest.mark.parametrize(
    "mean_config, mixture_config, new_data, expected_pred_mean",
    [
        (
            RegressionConfig(
                distribution="normal",
                outcome="yi",
                stderr="sei",
                formula="~ 1 + group + (1|team)",
                link="identity",
            ),
            MixtureConfig(use_null_mixture=False),
            pd.DataFrame(
                {
                    "group": ["a"] * 2 + ["b"] * 2,
                    "team": ["x", "y"] * 2,
                }
            ),
            tensor([0.1] * 2 + [0.4] * 2, dtype=torch.float32),
        ),
    ],
)
def test_model_predict(data, mean_config, mixture_config, new_data, expected_pred_mean):
    model = HME(
        data,
        ModelConfig(
            mean_regression=mean_config,
            mean_mixture=mixture_config,
        ),
    )
    post_samples, post_diagnostics = model.infer(
        InferConfig(n_iter=10000, n_warmup=1000, seed=0)
    )
    actual = tensor(
        model.predict(new_data).mean(axis=1),
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected_pred_mean, atol=0.01)
