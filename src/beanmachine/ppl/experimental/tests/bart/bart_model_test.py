# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.bart_model import (
    BART,
    NoiseStandardDeviation,
)


@pytest.fixture
def X():
    return torch.Tensor([[3.0, 1.0], [4.0, 1.0], [1.5, 1.0], [-1.0, 1.0]])


@pytest.fixture
def residual(X):
    return X * 0.1


@pytest.fixture
def sigma():
    return NoiseStandardDeviation(prior_concentration=0.1, prior_rate=0.2)


def test_sigma_sampling(sigma, X, residual):
    prev_val = sigma.val
    sample = sigma.sample(X=X, residual=residual)
    assert not prev_val == sigma.val
    assert sigma.val == sample


@pytest.fixture
def y(X):
    return X[:, 0] + X[:, 1]


@pytest.fixture
def bart(X, y):
    return BART(num_trees=1).fit(X=X, y=y, num_burn=1, num_samples=40)


def test_prediction_with_intervals(X, y, bart):
    coverage = 0.999
    with pytest.raises(ValueError):
        _ = bart.predict_with_intervals(X, coverage=coverage)
    low_coverage = 0.2
    y_pred, lower_bounds, upper_bounds = bart.predict_with_intervals(
        X, coverage=low_coverage
    )
    pred_samples = bart.get_posterior_predictive_samples(X)

    assert torch.all(torch.max(pred_samples, dim=1)[0] >= upper_bounds)
    assert torch.all(torch.min(pred_samples, dim=1)[0] <= lower_bounds)
    assert torch.all(torch.median(pred_samples, axis=1)[0] <= upper_bounds)
    assert torch.all(torch.median(pred_samples, axis=1)[0] >= lower_bounds)
    assert torch.all(y_pred == bart.predict(X))


@pytest.fixture
def X_test():
    return torch.Tensor([[3.1, 2.5]])


@pytest.fixture
def y_test(X_test):
    return X_test[:, 0] + X_test[:, 1]


def test_predict(X_test, y_test, bart):
    y_pred = bart.predict(X_test)
    assert len(X_test) == len(y_pred)
    assert len(y_test) == len(y_pred)
