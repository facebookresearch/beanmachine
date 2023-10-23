# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import gpytorch as gp
import pytest
import torch
from sts.abcd.model import ABCDExactGPModel
from sts.data import DataTensor
from sts.gp.kernels import WhiteNoiseKernel


torch.manual_seed(0)
NUM_DATA = 10
X_TRAIN = DataTensor(
    torch.linspace(0, 1, NUM_DATA).unsqueeze(-1), header=["x"], normalize_cols=["x"]
)
Y_TRAIN = DataTensor(
    X_TRAIN.tensor * 8 + 2.0 + torch.randn(X_TRAIN.tensor.shape),
    header=["y"],
    normalize_cols=["y"],
)


@pytest.mark.parametrize("x_train, y_train", [(X_TRAIN, Y_TRAIN)])
@pytest.mark.filterwarnings("ignore::gpytorch.utils.warnings.GPInputWarning")
def test_model(x_train, y_train):
    current_kernel = WhiteNoiseKernel(noise=1e-4)
    likelihood = gp.likelihoods.GaussianLikelihood()
    model = ABCDExactGPModel(x_train, y_train.squeeze(-1), likelihood, current_kernel)
    model.optimize_params(num_epochs=10)
    score = model.score()
    assert isinstance(score, torch.Tensor)
    n = torch.tensor(x_train.shape[0])
    expected = model.loss * 2 * n + len(list(model.cov.parameters())) * torch.log(n)
    assert torch.allclose(score, expected)
    _ = model.predict(x_train)
    mean, cov = model.decompose_timeseries(x_train.tensor)
    expected_train_mean = model(x_train).mean
    sum_of_part_means = sum([model.mean(x_train)] + [p.mean for p in cov.values()])
    assert torch.allclose(sum_of_part_means, expected_train_mean)
