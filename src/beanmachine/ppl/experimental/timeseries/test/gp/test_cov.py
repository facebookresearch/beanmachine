# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import pytest
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.priors import NormalPrior
from sts.data import DataTensor
from sts.gp.cov import (
    ChangePoint,
    Covariance,
    Regression,
    Seasonality,
    SpectralMixture,
    Trend,
    WhiteNoise,
)


torch.manual_seed(0)
SAMPLE_DATA = [
    DataTensor(
        torch.cat([torch.randn(10, 4), torch.arange(10).unsqueeze(-1)], -1),
        header=["a", "b", "c", "d", "t"],
        normalize_cols=["t"],
    )
]


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_composition(x):
    cov = Covariance(x)
    cov.add_regression(features=["a", "b"], kernel_cls=RBFKernel)
    cov.add_seasonality(time_axis="t", period_length=3)
    cov.add_seasonality(time_axis="t", period_length=4)
    cov.add_trend("t", lengthscale=1, name="Trend-short")
    cov.add_trend("t", lengthscale=4, name="Trend-long")
    assert len(cov.parts) == 5
    names = list(cov.parts)
    parts = list(cov.parts.values())
    assert names[0] == parts[0].name == "Regression"
    assert names[1] == parts[1].name == "Seasonality"
    assert names[2] == parts[2].name == "Seasonality_1"
    assert names[3] == parts[3].name == "Trend-short"
    assert names[4] == parts[4].name == "Trend-long"
    assert torch.allclose(
        cov(x).evaluate(), sum(p(x.tensor).evaluate() for p in cov.parts.values())
    )


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_fixed_params(x):
    trend = Trend(x, time_axis="t", lengthscale=2, fix_lengthscale=True)
    seasonality = Seasonality(x, time_axis="t", period_length=3, fix_period=True)
    std = x.transforms["t"].inv.scale
    # test normalization
    assert torch.allclose(trend.kernel.base_kernel.lengthscale, 2.0 / std)
    assert torch.allclose(trend.lengthscale, torch.tensor(2.0))
    # fixed param not optimizable
    for name, val in trend.named_parameters():
        if "lengthscale" in name:
            assert val in trend.fixed_params
    # test normalization
    assert torch.allclose(seasonality.kernel.base_kernel.period_length, 3.0 / std)
    assert torch.allclose(seasonality.period_length, torch.tensor(3.0))
    # fixed param not optimizable
    for name, val in seasonality.named_parameters():
        if "period_length" in name:
            assert val in seasonality.fixed_params


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_regression(x):
    reg = Regression(x, ["a", "b", "c"], kernel_cls=RBFKernel)
    assert len(reg.lengthscale[0]) == 3
    assert reg(x).shape == (x.shape[0], x.shape[0])


@pytest.mark.parametrize("x", SAMPLE_DATA)
@pytest.mark.parametrize("agg", ["sum", "prod"])
def test_agg_kernel(x, agg):
    seasonality = Seasonality(x, "t", 3, fix_period=True)
    short_trend = Trend(x, "t", lengthscale=1, fix_lengthscale=True, name="ShortTrend")
    long_trend = Trend(x, "t", lengthscale=3, fix_lengthscale=True, name="LongTrend")
    if agg == "sum":
        cov = seasonality + short_trend + long_trend
        assert cov.name == "AdditiveKernel(Seasonality,ShortTrend,LongTrend)"
        assert torch.allclose(
            cov(x).evaluate(), sum(p(x).evaluate() for p in cov.parts.values())
        )
    else:
        cov = seasonality * short_trend * long_trend
        assert cov.name == "ProductKernel(Seasonality,ShortTrend,LongTrend)"
        assert torch.allclose(
            cov(x).evaluate(), math.prod(p(x).evaluate() for p in cov.parts.values())
        )


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_spectralmixture(x):
    spectralmixture = SpectralMixture(
        x,
        "t",
        num_mixtures=2,
        mixture_scales=torch.tensor([[[1.0]], [[1.0]]]),
        mixture_means=torch.tensor([[[1.0]], [[1.0]]]),
        mixture_weights=torch.tensor([1.0, 1.0]),
    )
    scale = x.transforms["t"].inv.scale
    assert spectralmixture.kernel.num_mixtures == 2
    assert torch.allclose(spectralmixture.scale, scale)
    assert torch.allclose(
        spectralmixture.mixture_scales,
        spectralmixture.kernel.mixture_scales,
    )
    assert torch.allclose(
        spectralmixture.mixture_means,
        spectralmixture.kernel.mixture_means,
    )
    assert torch.allclose(
        spectralmixture.mixture_weights,
        spectralmixture.kernel.mixture_weights,
    )


@pytest.mark.parametrize("x", SAMPLE_DATA)
def test_noise(x):
    noise = WhiteNoise(x, "t", 1e-3, fix_noise=True, noise_prior=NormalPrior(0, 1))
    assert torch.allclose(noise.kernel.noise, torch.tensor(1e-3))
    for name, val in noise.named_parameters():
        if "noise" in name:
            assert val in noise.fixed_params
    assert type(noise.kernel.noise_prior) == NormalPrior


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_sm_kernel_smoke(data):
    train_size = 6
    x = data[:train_size, ["t"]]
    y = data[:train_size, "a"]
    cov = Covariance(x, y)
    cov.add_spectral_mixture(
        time_axis="t",
        num_mixtures=2,
        train_x=x.tensor,
        train_y=y.tensor,
        name="SM1",
    )
    assert len(cov.parts) == 1
    names = list(cov.parts)
    parts = list(cov.parts.values())

    assert names[0] == parts[0].name == "SM1"

    assert torch.allclose(
        cov(x).evaluate(), sum(p(x.tensor).evaluate() for p in cov.parts.values())
    )


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_changepoint(data):
    train_size = 6
    x = data[:train_size, ["t", "b"]]
    y = data[:train_size, "a"]
    cov = Covariance(x, y)
    trend = Trend(x, "t", kernel=RBFKernel, lengthscale=1.0, fix_lengthscale=True)
    periodic = Seasonality(x, "t", period_length=1.0, fix_period=True)
    cp = ChangePoint(x, "t", (trend, periodic), x[0][0].item(), name="CP")
    cov.append(cp)
    cov(x).evaluate()
    assert not cov.fixed_params.difference(
        trend.fixed_params.union(periodic.fixed_params)
    )
