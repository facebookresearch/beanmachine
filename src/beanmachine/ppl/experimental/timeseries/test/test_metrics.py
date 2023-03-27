# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import pytest
import torch
from sts.data import DataTensor
from sts.metrics import crps_ensemble, crps_gaussian, log_likelihood_error, MAE


torch.manual_seed(0)
SAMPLE_DATA = [
    DataTensor(
        torch.cat([torch.randn(10, 4), torch.arange(10).unsqueeze(-1)], -1),
        header=["a", "b", "c", "d", "t"],
        normalize_cols=["t"],
    )
]


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_MAE(data):
    train_size = 6
    x = data[:train_size, ["t"]]
    y_true = data[:train_size, "a"].tensor
    noise = torch.randn(x.size()).view(-1) * 1e-4
    y_pred = y_true + noise
    actual = MAE(y_pred, y_true)
    expected = (y_pred - y_true).abs().mean()
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_LL(data):
    var = 1
    train_size = 6
    y_true = data[:train_size, "a"].tensor
    y_pred = y_true + torch.randn(y_true.size()).view(-1) * 1e-6
    actual = -0.5 * torch.log(torch.tensor(2 * math.pi * var)).mean()
    expected = log_likelihood_error(y_pred, y_true, var)
    assert torch.allclose(actual, expected, rtol=1e-4)


@pytest.mark.parametrize("data", SAMPLE_DATA)
@pytest.mark.filterwarnings("ignore: This is a naive implementation:UserWarning")
def test_crps(data):
    shape = (2, 3)
    mu = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape))
    sig = torch.square(torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)))
    obs = torch.normal(mean=mu, std=sig)

    n = 1000
    q = torch.linspace(0.0 + 0.5 / n, 1.0 - 0.5 / n, n)
    # convert to the corresponding normal deviates
    z = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])).icdf(q)
    forecasts = z.reshape(-1, 1, 1) * sig + mu

    expected = crps_ensemble(obs, forecasts, axis=0)
    actual = crps_gaussian(obs, mu, sig).double()
    assert torch.allclose(actual, expected)
