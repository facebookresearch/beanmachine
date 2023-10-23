# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from numpy import testing
from sts.data import (
    DataTensor,
    df_to_tensor,
    get_mvn_stats,
    OVERRIDES,
    POINTWISE_OPS,
    REDUCTION_OPS,
)
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
def test_indexing_pandas(df):
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


def test_errors_creation():
    # Scalar DataTensor.
    with pytest.raises(ValueError, match="Scalar DataTensor must *"):
        DataTensor(torch.tensor(0), ["a"])
    # Single column tensor with no singleton dim.
    with pytest.raises(ValueError, match="Tensor data dim size *"):
        DataTensor(torch.ones(3), ["a"])
    with pytest.raises(IndexError, match="dim=-3 invalid for a tensor of dim"):
        DataTensor(torch.ones(2, 3), header=("a", "b"), dim=-3)
    with pytest.raises(IndexError, match="dim=2 invalid for a tensor of dim"):
        DataTensor(torch.ones(2, 3), header=("a", "b", "c"), dim=2)


# Tensor indexing should work as expected once corresponding string
# headers are replaced with their integer equivalents
# i.e. dt[:, "a"] and dt[:, 0] should return the same underlying tensor
# if "a" is the first column.
@pytest.mark.parametrize(
    "dt, str_idxs, torch_idxs, expected_header",
    [
        (
            DataTensor(torch.randn(2, 3), ["a", "b", "c"]),
            (slice(None), "a"),
            (slice(None), 0),
            (),
        ),
        (
            DataTensor(torch.randn(2, 3), ["a", "b", "c"]),
            (slice(None), ["a"]),
            (slice(None), [0]),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 3), ["a", "b"], dim=0),
            ("a",),
            (0,),
            (),
        ),
        (
            DataTensor(torch.randn(2, 3), ["a", "b"], dim=0),
            "a",
            0,
            (),
        ),
        (
            DataTensor(torch.randn(2, 3), ["a", "b"], dim=0),
            (["b", "a"]),
            ([1, 0]),
            ("b", "a"),
        ),
        (
            DataTensor(torch.randn(2, 3), ["a", "b", "c"]),
            (Ellipsis, ["b", "a"]),
            (Ellipsis, [1, 0]),
            ("b", "a"),
        ),
        (
            DataTensor(torch.randn(2), ["a", "b"]),
            ("a",),
            (0,),
            (),
        ),
        (
            DataTensor(torch.randn(2), ["a", "b"]),
            (["a"],),
            ([0],),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 1), ["a"]),
            (slice(None), "a"),
            (slice(None), 0),
            (),
        ),
        (
            DataTensor(torch.randn(2, 3), ["a", "b"], dim=0),
            ("a",),
            (0,),
            (),
        ),
        # Advanced indexing: resulting tensor of size 2
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b"], dim=0),
            (["a", "b"], 0, [0, 1]),
            ([0, 1], 0, [0, 1]),
            (),
        ),
        # Unsqueeze and ellipsis
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b"], dim=1),
            (Ellipsis, ["a"], slice(None)),
            (Ellipsis, [0], slice(None)),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b"], dim=1),
            (0, None, ["a"], None, slice(None)),
            (0, None, [0], None, slice(None)),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b", "c"]),
            (Ellipsis, ["a", "b"], None),
            (Ellipsis, slice(0, 2, None), None),
            ("a", "b"),
        ),
        (
            DataTensor(torch.tensor(0)),
            (None, None),
            (None, None),
            (),
        ),
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b"], dim=0),
            (["a"], Ellipsis),
            ([0], Ellipsis),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 2, 3, 3), ["a", "b", "c"], dim=2),
            (Ellipsis, ["a"], None, slice(None)),
            (Ellipsis, [0], None, slice(None)),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 2, 3, 3), ["a", "b", "c"], dim=2),
            (None, Ellipsis, ["a"], None, slice(None)),
            (None, Ellipsis, [0], None, slice(None)),
            ("a",),
        ),
        (
            DataTensor(torch.randn(2, 2, 3, 3), ["a", "b", "c"], dim=2),
            (Ellipsis, None, ["a"], slice(None)),
            (Ellipsis, None, [0], slice(None)),
            ("a",),
        ),
        # Masked Indexing
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b"], dim=0),
            [False, True],
            [False, True],
            (),
        ),
        (
            DataTensor(torch.randn(2, 2, 3), ["a", "b"], dim=0),
            torch.BoolTensor([False, True]),
            torch.BoolTensor([False, True]),
            (),
        ),
    ],
)
def test_indexing(dt, str_idxs, torch_idxs, expected_header):
    str_indexed = dt[str_idxs]
    trch_indexed = dt[torch_idxs]
    assert_array_equal(str_indexed.numpy(), trch_indexed.numpy())
    assert_array_equal(str_indexed.numpy(), trch_indexed.numpy())
    assert str_indexed.header == expected_header
    assert trch_indexed.header == expected_header


def test_indexing_unsqueeze():
    t = torch.randn(2, 3, 4, 4)
    dt = DataTensor(t, header=("a", "b", "c"), dim=1)
    t_1 = t[None]
    dt_1 = dt[None]
    assert dt_1.shape == t_1.shape == (1, 2, 3, 4, 4)
    assert dt_1.data_dim == -3
    assert dt_1.header == dt.header
    t_2 = t[..., None]
    dt_2 = dt[..., None]
    assert dt_2.shape == t_2.shape == (2, 3, 4, 4, 1)
    assert dt_2.data_dim == -4
    assert dt_2.header == dt.header
    t_3 = t[None, 0, ..., None]
    dt_3 = dt[None, 0, ..., None]
    assert dt_3.shape == t_3.shape == (1, 3, 4, 4, 1)
    assert dt_3.data_dim == -4
    assert dt_3.header == dt.header
    t_4 = t[None, :, [1], ..., 0]
    dt_4 = dt[None, :, ["a"], ..., 0]
    assert dt_4.shape == t_4.shape == (1, 2, 1, 4)
    assert dt_4.data_dim == -2
    assert dt_4.header == ("a",)
    t_5 = t[None, :, 1, ..., 0]
    dt_5 = dt[None, :, "a", ..., 0]
    assert dt_5.shape == t_5.shape == (1, 2, 4)
    assert dt_5.data_dim is None
    assert dt_5.header == ()
    t_6 = t[None, :, 1, ..., None, 0]
    dt_6 = dt[None, :, "a", ..., None, 0]
    assert dt_6.shape == t_6.shape == (1, 2, 4, 1)
    assert dt_6.data_dim is None
    assert dt_6.header == ()


@pytest.mark.parametrize(
    "dt, str_idxs, err_type, err_msg",
    [
        (
            DataTensor(torch.tensor(1)),
            ("a"),
            IndexError,
            "too many indices for tensor",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            ("a"),
            IndexError,
            "String indexing at dim 0",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            (1, 1, ["b"]),
            IndexError,
            "String indexing at dim 2",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            (1, ["b", "c"]),
            IndexError,
            "key=c not in",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=0),
            "c",
            IndexError,
            "key=c not in",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            (..., ["b", "c"], ...),
            NotImplementedError,
            "More than 1 ellipsis",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            (1, ["b"], ["a"]),
            IndexError,
            "String indexing at dim 2",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            (..., ["a"]),
            IndexError,
            "String indexing at dim -1",
        ),
        (
            DataTensor(torch.randn(2, 2, 3), header=("a", "b"), dim=1),
            (1, [None, "a"]),
            IndexError,
            "Nested `None` indices are disallowed",
        ),
    ],
)
def test_indexing_errors(dt, str_idxs, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        dt[str_idxs]


@pytest.mark.parametrize(
    "dt",
    [
        DataTensor(torch.randn(2, 3), ["a", "b", "c"]),
        DataTensor(torch.randn(2), ["a", "b"]),
        DataTensor(torch.randn(2, 1), ["a"]),
    ],
)
def test_squeeze_ops(dt):
    orig_shape = dt.shape
    col_dim = dt.data_dim
    # shape: (1, ...)
    dt_unsq = dt.unsqueeze(0)
    assert dt_unsq.shape == (1,) + orig_shape
    assert dt_unsq.data_dim == col_dim
    assert dt_unsq.header == dt.header
    # shape: (1, ..., 1)
    dt_unsq = dt_unsq.unsqueeze(-1)
    assert dt_unsq.shape == (1,) + orig_shape + (1,)
    assert dt_unsq.data_dim == col_dim - 1
    assert dt_unsq.header == dt.header
    # shape: (1, ...)
    dt_sq = dt_unsq.squeeze(-1)
    assert dt_sq.shape == (1,) + orig_shape
    assert dt_sq.data_dim == col_dim
    assert dt_sq.header == dt.header
    # Squeeze on singleton dim removes the header (single column) and results in
    # a reduced tensor; otherwise has no effect.
    dt_sq = dt_sq.squeeze(-1)
    if orig_shape[-1] != 1:
        assert dt_sq.shape == (1,) + orig_shape
        assert dt_sq.data_dim == col_dim
        assert dt_sq.header == dt.header
    else:
        assert dt_sq.shape == (1,) + orig_shape[:-1]
        assert dt_sq.data_dim is None
        assert dt_sq.header == ()


@pytest.mark.parametrize("df", SAMPLE_DFS)
def test_normalization(df):
    x_orig = df_to_tensor(df)
    x = df_to_tensor(df, normalize_cols=["x"])
    # append singleton dims for testing
    x = x.unsqueeze(0).unsqueeze(-1)
    # normalize 1 column
    col = df["x"].to_numpy().astype(np.float32)
    expected_val = (col - np.mean(col)) / np.std(col, ddof=1)
    column = x[..., "x", :].squeeze(0).squeeze(-1).numpy()
    assert_array_equal(column, expected_val)
    denormalized = x.denormalize(x).squeeze(0).squeeze(-1).numpy()
    assert_array_equal(denormalized, x_orig.numpy())
    # normalize all columns
    x = df_to_tensor(df, normalize_cols=True)
    # append singleton dims for testing
    x = x.unsqueeze(0).unsqueeze(-1)
    arr = df.to_numpy().astype(np.float32)
    expected_val = (arr - np.mean(arr, axis=0)) / np.std(arr, ddof=1, axis=0)
    x_sq = x.squeeze(0).squeeze(-1)
    assert_array_equal(x_sq.numpy(), expected_val)
    denormalized = x_sq.denormalize(x_sq)
    assert_array_equal(denormalized.numpy(), x_orig.numpy())
    # assert metadata preserved after squeeze
    assert len(x.unsqueeze(0).transforms) > 0


@pytest.mark.parametrize("df", SAMPLE_DFS)
def test_normalization_custom(df):
    x_orig = df_to_tensor(df)
    x = df_to_tensor(
        df, normalize_cols={"x": AffineTransform(0.0, 10.0).inv}, dtype=torch.float64
    )
    # normalize x column
    column = x[..., "x"].numpy()
    expected_val = df["x"] / 10
    assert_array_equal(column, expected_val)
    denormalized = x.denormalize(x).numpy()
    assert_array_equal(denormalized, x_orig.numpy())


def test_normalization_errors():
    df = pd.DataFrame({"x": list(range(3, 7)), "y": list(range(4))})
    with pytest.raises(TypeError):
        df_to_tensor(df, normalize_cols=set())

    with pytest.raises(TypeError):
        df_to_tensor(df, normalize_cols={"x": lambda x: (x - x.mean) / x.std()})

    dt = df_to_tensor(df, normalize_cols=["x"])
    with pytest.raises(ValueError, match="Column x already normalized."):
        dt.normalize(transform_columns=["x"])

    with pytest.raises(ValueError, match="Columns {'x'} already normalized."):
        dt.normalize(transform_columns={"x": AffineTransform(0, 1)})
    # Should not raise an error as y is unnormalized
    dt.normalize(transform_columns=["y"])
    # Should not raise an error even though number of rows is different
    dt.denormalize(torch.randn(3, 2))
    # Error as target ndim is different
    with pytest.raises(ValueError):
        dt.denormalize(torch.randn(2))
    # Error as size of column dim is different
    with pytest.raises(ValueError):
        dt.denormalize(torch.randn(2, 3))


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


@pytest.mark.parametrize(
    "x",
    [
        DataTensor(torch.rand(()), header=[]),
        DataTensor(torch.rand(1), header=["a"]),
        DataTensor(torch.rand(1, 2), header=["a", "b"]),
    ],
)
@pytest.mark.parametrize("op", POINTWISE_OPS)
def test_pointwise_ops(x, op):
    sig = inspect.signature(OVERRIDES[op])
    x_float = x
    x_long = x.long()
    if op.__name__ in ("clamp", "clip"):
        args = (x, 0.5)
    else:
        args = []
        for p in sig.parameters.values():
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty:
                args.append(x_long if op.__name__.startswith("bitwise") else x_float)
    ret_dt = op(*args)
    ret_trch = op(*(x.tensor if isinstance(x, DataTensor) else x for x in args))
    assert_array_equal(ret_dt.tensor, ret_trch)
    assert ret_dt.header == x.header


@pytest.mark.parametrize(
    "x",
    [
        DataTensor(torch.rand(()), header=[]),
        DataTensor(torch.rand(1), header=["a"]),
        DataTensor(torch.rand(1, 2), header=["a", "b"]),
    ],
)
@pytest.mark.parametrize("op", REDUCTION_OPS)
def test_reduction_ops(x, op):
    sig = inspect.signature(OVERRIDES[op])
    if "quantile" in op.__name__:
        args = (x, 0.9)
    elif op.__name__ == "logsumexp":
        args = (x, 0)
    else:
        args = []
        for p in sig.parameters.values():
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty:
                args.append(x)
    ret_dt = op(*args)
    ret_trch = op(*(x.tensor if isinstance(x, DataTensor) else x for x in args))
    bound_sig = sig.bind(*args)
    bound_sig.apply_defaults()
    # We check for header retention when the data dim is not reduced
    is_reduced = x.data_dim is None or bound_sig.arguments.get("dim", None) in (
        None,
        x.data_dim,
        x.ndim + x.data_dim,
    )
    if isinstance(ret_dt, tuple):
        for x, y in zip(ret_dt, ret_trch):
            if isinstance(x, DataTensor):
                assert_array_equal(x.tensor, y)
                if not is_reduced:
                    assert ret_dt.header == x.header
            else:
                assert_array_equal(x, y)
    else:
        assert_array_equal(ret_dt.tensor, ret_trch)
        if not is_reduced:
            assert ret_dt.header == x.header
