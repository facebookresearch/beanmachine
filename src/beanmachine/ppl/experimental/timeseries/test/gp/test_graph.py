# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch
from gpytorch.likelihoods import GaussianLikelihood
from sts.data import DataTensor
from sts.gp.graph import plot_components
from sts.gp.model import ExactGPAdditiveModel


@pytest.mark.parametrize("n_cols", [1, 2, 3])
@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.filterwarnings("ignore::gpytorch.utils.warnings.GPInputWarning")
def test_plot_components_smoke(n_cols, n_components):
    data = DataTensor(
        torch.cat([torch.arange(10).unsqueeze(-1), torch.randn(10, 1)], dim=-1),
        header=["x", "y"],
        normalize_cols=["y"],
    )
    x = data[:, ["x"]]
    y = data[:, "y"]
    model = ExactGPAdditiveModel(x, y, GaussianLikelihood())
    for i in range(n_components):
        model.cov.add_seasonality(time_axis="x", period_length=i + 1)
    model.predict(x)
    mean, cov = model.decompose_timeseries(x)
    assert len(cov) == n_components
    plot_components(
        data[:, "x"],
        (mean, cov),
        transform=data.transforms["y"].inv,
        ncols=n_cols,
        plot_mean=True,
    )
