import numpy as np
import pandas as pd
import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from numpy import testing
from sts.data import df_to_tensor, get_mvn_stats
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    ExpTransform,
)


# Needed for autodep
assert_array_equal = testing.assert_array_equal

SAMPLE_DFS = [
    pd.DataFrame({"x": list(range(3, 7)), "y": list(range(4))}),
    pd.DataFrame({"x": list(range(3))}),
]


@pytest.mark.parametrize("df", SAMPLE_DFS)
def test_indexing(df):
    x = df_to_tensor(df, dtype=torch.double)
    cols = list(x.header)
    c0 = cols[0]
    # lhs: DataTensor indexing (equiv to pytorch indexing) | rhs: pandas DF indexing
    assert_array_equal(x[:, c0].numpy(), df[c0].to_numpy())
    assert_array_equal(x[:, [c0]].numpy(), df[[c0]].to_numpy())
    assert_array_equal(x[:, cols].numpy(), df[cols].to_numpy())
    # integer indexing should work as expected
    assert_array_equal(x[:, 0].numpy(), df[c0].to_numpy())
    assert_array_equal(x[:, [0]].numpy(), df[[c0]].to_numpy())
    assert_array_equal(x[:, list(range(len(x.header)))].numpy(), df[cols].to_numpy())


@pytest.mark.parametrize("df", SAMPLE_DFS)
def test_normalization(df):
    x_orig = df_to_tensor(df)
    x = df_to_tensor(df, normalize_cols=["x"])
    # normalize 1 column
    col = df["x"].to_numpy().astype(np.float32)
    expected_val = (col - np.mean(col)) / np.std(col, ddof=1)
    assert_array_equal(x[:, "x"].numpy(), expected_val)
    assert_array_equal(x.denormalize(x).numpy(), x_orig.numpy())
    # normalize all
    x = df_to_tensor(df, normalize_cols=True)
    arr = df.to_numpy().astype(np.float32)
    expected_val = (arr - np.mean(arr, axis=0)) / np.std(arr, ddof=1, axis=0)
    assert_array_equal(x.numpy(), expected_val)
    assert_array_equal(x.denormalize(x).numpy(), x_orig.numpy())
    # assert metadata preserved after squeeze
    assert len(x.unsqueeze(0).transforms) > 0


@pytest.mark.parametrize("n", [2, 5, 6])
@pytest.mark.parametrize("logspace", [False, True])
def test_stats_mvn(n, logspace):
    torch.manual_seed(5)
    L = torch.randn(n, n).tril() * 0.1
    cov = L @ L.T
    affine_transform = AffineTransform(2.0, 3.0)
    mvn = MultivariateNormal(mean=torch.zeros(n), covariance_matrix=cov)
    samples = mvn.sample(torch.Size((100000,)))
    scaled_samples = affine_transform(samples)
    transform = affine_transform
    if logspace:
        scaled_samples = scaled_samples.exp()
        transform = ComposeTransform([affine_transform, ExpTransform()])
    mean, var, ci = get_mvn_stats(mvn, transform=transform)
    assert torch.allclose(mean, scaled_samples.mean(dim=0), rtol=0.05)
    assert torch.allclose(var, scaled_samples.var(dim=0), rtol=0.05)
    q1, q2 = scaled_samples.quantile(q=torch.tensor([0.05, 0.95]), dim=0)
    assert torch.allclose(ci[0], q1, rtol=0.05)
    assert torch.allclose(ci[1], q2, rtol=0.05)
