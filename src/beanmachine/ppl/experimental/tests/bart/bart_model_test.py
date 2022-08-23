# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from beanmachine.ppl.experimental.causal_inference.models.bart.bart_model import (
    BART,
    XBART,
)


@pytest.fixture
def X():
    return torch.Tensor([[3.0, 1.0], [4.0, 1.0], [1.5, 1.0], [-1.0, 1.0]])


@pytest.fixture
def y(X):
    return X[:, 0] + X[:, 1]


@pytest.fixture
def bart(X, y):
    return BART(num_trees=1).fit(X=X, y=y, num_burn=1, num_samples=39)


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


def test_predict_with_quantiles_bart(X_test, bart):
    quantiles = torch.Tensor([0.5])
    y_pred, qvals = bart.predict_with_quantiles(X_test, quantiles=quantiles)
    posterior_samples = bart.get_posterior_predictive_samples(X_test)
    # median for even number of samples is not unique
    assert (1 - bart.num_samples % 2) or torch.all(
        torch.median(posterior_samples, dim=1)[0] == qvals
    )


@pytest.fixture
def xbart(X, y):
    return XBART(num_trees=1).fit(X=X, y=y, num_burn=1, num_samples=9)


def test_predict_xbart(X_test, y_test, xbart):
    y_pred = xbart.predict(X_test)
    assert len(X_test) == len(y_pred)
    assert len(y_test) == len(y_pred)


def test_predict_with_quantiles_xbart(X_test, xbart):
    quantiles = torch.Tensor([0.5])
    y_pred, qvals = xbart.predict_with_quantiles(X_test, quantiles=quantiles)
    posterior_samples = xbart.get_posterior_predictive_samples(X_test)
    # median for even number of samples is not unique
    assert (1 - xbart.num_samples % 2) or torch.all(
        torch.median(posterior_samples, dim=1)[0] == qvals
    )
