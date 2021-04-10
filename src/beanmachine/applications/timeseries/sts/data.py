from collections.abc import Sequence
from typing import Callable, Dict, Iterable, List, Optional, Union

import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import transforms


class DataTensor:
    """
    Data structure used by time series model to store additional metadata with
    :class:`torch.Tensor` objects such as header (column) information and
    normalization.

    param tensor: a 1-D or 2-D :class:`torch.Tensor` instance.
    param header: list of column names having the same size as the innermost dim
        of `tensor`.
    param normalize_cols: list of columns to normalize by scaling to unit norm.
    """

    # TODO: Consider implementing `__torch_function__` interface to preserve
    # metadata info.
    def __init__(
        self,
        tensor: torch.Tensor,
        header: Iterable[str],
        normalize_cols: Optional[List[str]] = None,
    ):
        assert isinstance(tensor, torch.Tensor)
        if tensor.ndim < 1:
            raise ValueError("DataTensor must have ndim >= 1.")
        elif tensor.ndim == 1:
            if len(header) != 1:
                raise ValueError(
                    "For a 1-D DataTensor, header must be a singleton list."
                )
        elif tensor.shape[-1] != len(header):
            raise ValueError(
                f"Tensor rightmost dim size {tensor.shape[-1]} does not match header length {len(header)}."
            )
        self.tensor = tensor
        self.header = tuple(header)
        self.rev_index = {h: i for i, h in enumerate(header)}
        self.transforms = {}
        if normalize_cols:
            self.normalize(normalize_cols)

    def new_datatensor(self, tensor, header=None):
        header = header if header is not None else self.header
        data_tensor = DataTensor(tensor, header)
        data_tensor.transforms = self.transforms.copy()
        return data_tensor

    def normalize(self, columns: List[str]):
        """
        Scale the specified columns to unit norm.

        :param columns: List of columns (subset of `self.header`).
        """
        # TODO: support arbitrary normalization strategies
        for c in columns:
            if c in self.transforms:
                # already normalized
                return
            idx = self.rev_index[c]
            mean = (
                self.tensor[:, idx].mean()
                if self.tensor.ndim > 1
                else self.tensor.mean()
            )
            std = (
                self.tensor[:, idx].std() if self.tensor.ndim > 1 else self.tensor.std()
            )
            transform = transforms.AffineTransform(mean, std).inv
            self.transforms[c] = transform
            if self.tensor.ndim > 1:
                self.tensor[:, idx] = transform(self.tensor[:, idx])
            else:
                self.tensor = transform(self.tensor)

    def denormalize(
        self, data: Union["DataTensor", torch.Tensor]
    ) -> Union["DataTensor", torch.Tensor]:
        """
        Rescale the given tensor so that column values are in their original
            scale using the internal normalization metadata.

        :param data: a :class:`DataTensor` or :class:`torch.Tensor` instance.
        :return: a new :class:`DataTensor` or :class:`torch.Tensor` instance
            with rescaled columns.
        """
        if isinstance(data, DataTensor):
            header = data.header
            tensor = data.tensor
        else:
            tensor, header = data, None
        for c, transform in self.transforms.items():
            idx = self.rev_index[c]
            if tensor.ndim > 1:
                tensor[:, idx] = transform.inv(tensor[:, idx])
            else:
                tensor = transform.inv(tensor)
        return tensor if header is None else DataTensor(tensor, header)

    def get_index(self, key):
        col_key = key
        if isinstance(key, str):
            if key not in self.rev_index:
                raise ValueError(f"key={key} not in {self.header}.")
            col_key = self.rev_index[key]
        elif isinstance(key, Sequence) and len(key) > 0 and isinstance(key[0], str):
            col_idxs = []
            for c in key:
                if c not in self.rev_index:
                    raise ValueError(f"key={c} not in {self.header}.")
                col_idxs.append(self.rev_index[c])
            col_key = tuple(col_idxs)
        return col_key

    def squeeze(self, dim):
        return self.new_datatensor(self.tensor.squeeze(dim))

    def unsqueeze(self, dim):
        return self.new_datatensor(self.tensor.unsqueeze(dim))

    def __getattr__(self, name):
        return self.tensor.__getattribute__(name)

    def __getitem__(self, key):
        header = self.header
        if isinstance(key, str):
            raise ValueError("Rows cannot be indexed by header names.")
        elif isinstance(key, tuple):
            if len(key) < 2:
                raise ValueError(
                    f"Index incorrect for DataTensor with {self.tensor.ndim} dims."
                )
            col_key = key[-1]
            col_idx = self.get_index(col_key)
            header = (
                (self.header[col_idx],)
                if isinstance(col_idx, int)
                else tuple(self.header[c] for c in col_idx)
            )
            key = key[:-1] + (col_idx,)
        val = self.tensor[key]
        # for a scalar quantity, unwrap to return the value without header info
        if val.ndim < 1:
            return val
        return self.new_datatensor(self.tensor[key], header)

    def __len__(self):
        return len(self.tensor)

    def __repr__(self):
        return f"DataTensor(tensor={self.tensor}, header={self.header})"


def df_to_tensor(
    df: pd.DataFrame,
    columns: Optional[Iterable] = None,
    df_convert_fns: Dict[str, Callable] = None,
    dtype: torch.dtype = torch.float32,
    normalize_cols: Union[bool, Iterable] = False,
) -> DataTensor:
    """
    Construct and return a :class:`DataTensor` instance from a pandas DataFrame.

    :param df: Pandas DataFrame object.
    :param columns: Subset of columns to select from `df` to construct the
        `DataTensor` instance.
    :param df_convert_fns: A dict of post-processing functions by column names.
        This is useful when custom pandas data types such as datetime values
        into a `torch.float` value.
    :param dtype: default dtype for the backing `torch.Tensor` object. Defaults
        to `torch.float32`.
    :param normalize_cols: flag to indicate whether columns should be normalized
        to unit norm. A list of columns can be passed too in which case only the
        specified columns will be normallized.
    :return: `DataTensor` instance.
    """
    df = df.copy()
    header = list(df.columns)
    if df_convert_fns:
        for name, fn in df_convert_fns.items():
            df[name] = fn(df[name])
    if columns is not None:
        header = list(columns)
        df = df[columns]
    normalize_cols = header if normalize_cols is True else normalize_cols
    return DataTensor(
        torch.tensor(df.to_numpy(), dtype=dtype), header, normalize_cols=normalize_cols
    )


def _extract_transforms(transform):
    def _extract_affine_params(t):
        if isinstance(t, transforms.AffineTransform) or (
            isinstance(t, transforms._InverseTransform)
            and isinstance(t._inv, transforms.AffineTransform)
        ):
            return t.loc, t.scale
        return None, None

    loc, scale, exp_transform = None, None, False

    if transform is None:
        return 0.0, 1.0, False
    elif isinstance(transform, transforms.ExpTransform):
        return 0.0, 1.0, True
    elif isinstance(transform, transforms.ComposeTransform):
        if len(transform.parts) == 2:
            loc, scale = _extract_affine_params(transform.parts[0])
            if isinstance(transform.parts[1], transforms.ExpTransform):
                exp_transform = True
    else:
        loc, scale = _extract_affine_params(transform)
    return loc, scale, exp_transform


def get_mvn_stats(
    mvn: MultivariateNormal,
    transform: Optional[torch.distributions.Transform] = None,
    ci: int = 90,
):
    """
    Get mean, variance and confidence interval from a
        :class:`MultivariateNormal` distribution.

    :param mvn: the multivariatenormal distribution of interest.
    :param transform: any transforms to rescale the samples from `mvn` to
        their original scale.
    :param ci: confidence interval from 0-100 (defaults to 90)
    :return: tuple of `(mean, variance, ci)`.
    """
    mean = mvn.mean
    # XXX: calling .variance can blow up the memory due to some gpytorch
    # broadcasting ops.
    var = mvn.covariance_matrix.diag()
    mean_t, scale_t, exp_transform = _extract_transforms(transform)
    # TODO: We can draw multiple samples and compute stats when other
    # transforms are provided.
    if mean_t is None or scale_t is None:
        raise NotImplementedError(f"Unsupported transform {transform}.")
    mean = mean_t + mean * scale_t
    var = scale_t ** 2 * var
    p1, p2 = torch.tensor((100 - ci) / 200.0), torch.tensor((100 + ci) / 200.0)
    mean_orig, var_orig = mean, var
    q1 = mean + (2 * var).sqrt() * torch.erfinv(2 * p1 - 1)
    q2 = mean + (2 * var).sqrt() * torch.erfinv(2 * p2 - 1)
    if exp_transform:
        mean_orig = torch.exp(mean + var / 2)
        var_orig = mean_orig ** 2 * (var.exp() - 1)
        q1 = torch.exp(q1)
        q2 = torch.exp(q2)
    return (mean_orig, var_orig, (q1, q2))
