import pytest
import torch
from gpytorch.likelihoods import GaussianLikelihood
from sts.data import DataTensor
from sts.gp.graph import plot_components
from sts.gp.model import TimeSeriesExactGPModel

SAMPLE_DATA = [
    DataTensor(
        torch.cat([torch.randn(10, 4), torch.arange(10).unsqueeze(-1)], -1),
        header=["a", "b", "c", "d", "t"],
        normalize_cols=["t"],
    )
]


@pytest.mark.parametrize("data", SAMPLE_DATA)
def test_exact_gp_smoke(data):
    train_size = 6
    x = data[:train_size, ["a", "b", "c", "t"]]
    y = data[:train_size, "t"]
    x_eval = data[train_size:, ["a", "b", "c", "t"]]
    model = TimeSeriesExactGPModel(x, y, GaussianLikelihood())
    model.cov.add_regression(["a", "b"])
    model.cov.add_seasonality(time_axis="t", period_length=2, fix_period=True)
    train_apply = model.train_init(torch.optim.Adam(model.trainable_params))
    for _ in range(10):
        train_apply(x, y)
    p = model.predict(x_eval)
    assert p.mean.shape == (x_eval.shape[0],)
    mean, cov = model.decompose_timeseries(x_eval)
    assert len(mean) == 1
    assert len(cov) == 2
    expected_test_mean = model(x_eval).mean
    sum_of_part_means = sum([model.mean(x_eval)] + [p.mean for p in cov.values()])
    assert torch.allclose(sum_of_part_means, expected_test_mean)
    plot_components(x_eval[:, "t"].squeeze(-1), cov, y=p, marker="o")
