import math

import pytest
import torch
from gpytorch.kernels import RBFKernel
from sts.data import DataTensor
from sts.gp.cov import (
    Covariance,
    Trend,
    Regression,
    Seasonality,
)


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
