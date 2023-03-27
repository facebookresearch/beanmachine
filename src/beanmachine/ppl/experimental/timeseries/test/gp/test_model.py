# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import numpy as np
import pandas as pd
import pytest
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import LinearKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import LogNormalPrior
from sts.data import DataTensor, get_mvn_stats
from sts.gp.graph import plot_components
from sts.gp.model import AutomaticForecastingGP, ExactGPAdditiveModel
from sts.util import optional


torch.manual_seed(0)
SAMPLE_DATA = [
    DataTensor(
        torch.cat([torch.randn(10, 4), torch.arange(10).unsqueeze(-1)], -1),
        header=["a", "b", "c", "d", "t"],
        normalize_cols=["t"],
    )
]


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_training_respects_requires_grad(data):
    train_size = 6
    x = data[:train_size, ["a", "b", "c", "t"]]
    y = data[:train_size, "t"]
    x_eval = data[train_size:, ["a", "b", "c", "t"]]
    model = ExactGPAdditiveModel(x, y, GaussianLikelihood())
    model.cov.add_regression(["a", "b"], name="Regression")
    model.cov.add_seasonality(time_axis="t", period_length=2, fix_period=True)
    old_outputscale = model.cov.parts["Regression"].kernel.raw_outputscale
    old_outputscale.requires_grad_(False)
    old_outputscale = old_outputscale.clone()
    train_apply = model.train_init(torch.optim.Adam(model.trainable_params, lr=0.1))
    train_apply(x, y)
    new_outputscale = model.cov.parts["Regression"].kernel.raw_outputscale
    assert torch.allclose(new_outputscale, old_outputscale)
    mean, var, (ci_low, ci_high) = get_mvn_stats(model.predict(x_eval))
    for stat in (mean, var, ci_low, ci_high):
        assert not stat.requires_grad


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_exact_gp_smoke(data):
    train_size = 6
    x = data[:train_size, ["a", "b", "c", "t"]]
    y = data[:train_size, "t"]
    x_eval = data[train_size:, ["a", "b", "c", "t"]]
    model = ExactGPAdditiveModel(x, y, GaussianLikelihood())
    model.cov.add_regression(["a", "b"])
    model.cov.add_seasonality(time_axis="t", period_length=2, fix_period=True)
    train_apply = model.train_init(torch.optim.Adam(model.trainable_params))
    for _ in range(10):
        train_apply(x, y)
    p = model.predict(x_eval)
    assert not p.mean.requires_grad
    assert p.mean.shape == (x_eval.shape[0],)
    mean, cov = model.decompose_timeseries(x_eval)
    assert len(mean) == 1
    assert len(cov) == 2
    expected_test_mean = model(x_eval).mean
    sum_of_part_means = sum([model.mean(x_eval)] + [p.mean for p in cov.values()])
    assert torch.allclose(sum_of_part_means, expected_test_mean)
    plot_components(x_eval[:, "t"].squeeze(-1), (mean, cov), y=p, marker="o")


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_additive_spectralmixture_exact_gp(data):
    train_size = 6
    x = data[:train_size, ["t"]]
    y = data[:train_size, "a"]
    x_eval = data[train_size:, ["t"]]
    model = ExactGPAdditiveModel(x, y, GaussianLikelihood())
    model.cov.add_seasonality(time_axis="t", period_length=2, fix_period=True)
    model.cov.add_spectral_mixture(
        time_axis="t", num_mixtures=2, train_x=x.tensor, train_y=y.tensor, name="SM1"
    )
    model.cov.add_spectral_mixture(time_axis="t", num_mixtures=2, name="SM2")
    model.cov.add_trend(time_axis="t", kernel_cls=LinearKernel, name="LinearTrend")
    model.cov.add_trend(
        time_axis="t",
        kernel_cls=RBFKernel,
        lengthscale=100,
        fix_lengthscale=True,
        name="RBFTrend",
    )
    train_apply = model.train_init(torch.optim.Adam(model.trainable_params))
    for _ in range(20):
        train_apply(x, y)
    p = model.predict(x_eval)
    assert not p.mean.requires_grad
    assert p.mean.shape == (x_eval.shape[0],)
    mean, cov = model.decompose_timeseries(x_eval.tensor)
    assert len(mean) == 1
    assert len(cov) == 5
    expected_test_mean = model(x_eval).mean
    sum_of_part_means = sum([model.mean(x_eval)] + [p.mean for p in cov.values()])
    assert torch.allclose(sum_of_part_means, expected_test_mean, rtol=1e-4)


@pytest.mark.parametrize("N", [5, 20, 100, 650])
@pytest.mark.parametrize("lin_covariates", [False, True])
@pytest.mark.parametrize("disable_sm_kernel", [False, True])
@pytest.mark.parametrize("changepoint_locations", [[], [0.0], [0.0, 2.0]])
@pytest.mark.parametrize("detect_changepoints", [False, True])
@pytest.mark.filterwarnings("ignore:`changepoint_locations` will not be used")
def test_autoforecasting_gp_smoke(
    N, lin_covariates, disable_sm_kernel, changepoint_locations, detect_changepoints
):
    if N == 650:
        pytest.skip("See: T131274126")
    dates = np.arange("2018-01", "2021-01", dtype="datetime64[D]")
    data = pd.DataFrame(
        {
            "t": dates,
            "a": np.linspace(0.0, 1.0, len(dates)),
            "y": np.linspace(0.0, 10.0, len(dates)),
        }
    )
    with optional(N < 10, pytest.raises, ValueError):
        if lin_covariates:
            model = AutomaticForecastingGP(
                data[:N],
                "t",
                "y",
                linear_covariates=["a"],
                changepoint_locations=changepoint_locations,
                disable_sm_kernel=disable_sm_kernel,
            )
        else:
            model = AutomaticForecastingGP(
                data[:N],
                "t",
                "y",
                nonlinear_covariates=["a"],
                changepoint_locations=changepoint_locations,
                disable_sm_kernel=disable_sm_kernel,
                detect_changepoints=detect_changepoints,
            )
    if N > 10:
        if model.changepoint_locations:
            assert torch.allclose(
                model.cov.parts["Changepoint - Linear"].kernel.location,
                torch.as_tensor(changepoint_locations) / 365.25,
            )
        data = model.preprocess_data(data)
        x, y = data[:, ["t", "a"]], data[:, "y"]
        x_train = x[:N]
        y_train = y[:N]
        x_test = x[N:]
        train_apply = model.train_init(torch.optim.Adam(model.trainable_params))
        for _ in range(2):
            train_apply(x_train, y_train)
        model.predict(x_test)


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_outputscale_prior_constraint(data):
    outputscale_prior = LogNormalPrior(-1.0, 1.0)
    outputscale_constraint = GreaterThan(1.0)
    train_size = 6
    x = data[:train_size, ["t", "b", "c"]]
    y = data[:train_size, "a"]
    model = ExactGPAdditiveModel(x, y, GaussianLikelihood())
    model.cov.add_seasonality(
        time_axis="t",
        period_length=2,
        fix_period=True,
        outputscale_prior=outputscale_prior,
        outputscale_constraint=outputscale_constraint,
        name="Seasonality",
    )
    model.cov.add_trend(
        time_axis="t",
        kernel_cls=LinearKernel,
        outputscale_prior=outputscale_prior,
        outputscale_constraint=outputscale_constraint,
        name="LinearTrend",
    )
    model.cov.add_regression(
        ["b", "c"],
        kernel_cls=RBFKernel,
        outputscale_prior=outputscale_prior,
        outputscale_constraint=outputscale_constraint,
        name="Regression",
    )
    model.cov.add_spectral_mixture(
        time_axis="t", num_mixtures=2, train_x=x.tensor, train_y=y.tensor, name="SM"
    )
    train_apply = model.train_init(torch.optim.Adam(model.trainable_params))
    for _ in range(20):
        train_apply(x, y)
    for p in ["Seasonality", "LinearTrend", "Regression"]:
        kernel = model.cov.parts[p].kernel
        constraint = kernel.raw_outputscale_constraint
        outputscale = kernel.outputscale
        assert "outputscale_prior" in kernel._priors
        assert constraint.check(outputscale)
