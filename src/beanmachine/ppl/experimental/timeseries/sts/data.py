# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import functools
import inspect
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import transforms
from torch.overrides import get_testing_overrides
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten


POINTWISE_OPS = [
    torch.abs,
    torch.absolute,
    torch.acos,
    torch.arccos,
    torch.acosh,
    torch.arccosh,
    torch.add,
    torch.addcdiv,
    torch.addcmul,
    torch.angle,
    torch.asin,
    torch.arcsin,
    torch.asinh,
    torch.arcsinh,
    torch.atan,
    torch.arctan,
    torch.atanh,
    torch.arctanh,
    torch.atan2,
    torch.bitwise_not,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.ceil,
    torch.clamp,
    torch.clip,
    torch.conj,
    torch.copysign,
    torch.cos,
    torch.cosh,
    torch.deg2rad,
    torch.div,
    torch.divide,
    torch.digamma,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.fix,
    torch.float_power,
    torch.floor,
    torch.fmod,
    torch.frac,
    torch.ldexp,
    torch.lerp,
    torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logaddexp,
    torch.logaddexp2,
    torch.logical_and,
    torch.logical_not,
    torch.logical_or,
    torch.logical_xor,
    torch.logit,
    torch.hypot,
    torch.i0,
    torch.igamma,
    torch.igammac,
    torch.mul,
    torch.multiply,
    torch.nan_to_num,
    torch.neg,
    torch.negative,
    torch.nextafter,
    torch.pow,
    torch.rad2deg,
    torch.reciprocal,
    torch.remainder,
    torch.round,
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sgn,
    torch.signbit,
    torch.sin,
    torch.sinc,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.sub,
    torch.subtract,
    torch.tan,
    torch.tanh,
    torch.true_divide,
    torch.trunc,
    torch.xlogy,
]


REDUCTION_OPS = [
    torch.argmax,
    torch.argmin,
    torch.amax,
    torch.amin,
    torch.all,
    torch.any,
    torch.max,
    torch.min,
    torch.dist,
    torch.logsumexp,
    torch.mean,
    torch.median,
    torch.nanmedian,
    torch.mode,
    torch.norm,
    torch.nansum,
    torch.prod,
    torch.quantile,
    torch.nanquantile,
    torch.std,
    torch.std_mean,
    torch.sum,
    torch.unique,
    torch.unique_consecutive,
    torch.var,
    torch.var_mean,
]


HANDLED_FNS = {}
OVERRIDES = get_testing_overrides()


def implements(op):
    @functools.wraps(op)
    def _wrapped(fn):
        HANDLED_FNS[op] = fn
        return fn

    return _wrapped


# TODO: This can be removed in the next PyTorch version (> 1.9)
def tree_map(fn: Any, pytree: PyTree) -> PyTree:
    flat_args, spec = tree_flatten(pytree)
    return tree_unflatten([fn(i) for i in flat_args], spec)


def _check_arg_consistency(dt_nodes):
    check_eq = [(dt.data_dim, dt.header) for dt in dt_nodes]
    if not all(x == check_eq[0] for x in check_eq):
        raise ValueError(
            f"Different headers or data_dim found - {check_eq}."
            " Consider using underlying tensors directly."
        )


def _map_unwrap(args):
    """
    Simple unwrap mapping operation that does not work on nested PyTree.
    """
    unpacked, dt_nodes = [], []
    for x in args:
        if isinstance(x, DataTensor):
            unpacked.append(x.tensor)
            dt_nodes.append(x)
        else:
            if isinstance(x, Sequence):
                warnings.warn("For nested unpacking, use `tree_map`")
            unpacked.append(x)
    # Check that args have same header/data_dim
    _check_arg_consistency(dt_nodes)
    return unpacked, dt_nodes


def _handle_pointwise_ops(func, *args, **kwargs):
    args, dt_nodes = _map_unwrap(args)
    ret = func(*args, **kwargs)
    return DataTensor(ret, dt_nodes[0].header, dim=dt_nodes[0].data_dim)


def _handle_reduction_ops(func, *args, **kwargs):
    args, dt_nodes = _map_unwrap(args)
    prototype = dt_nodes[0]
    func_sign = inspect.signature(OVERRIDES[func]).bind(*args, **kwargs)
    func_sign.apply_defaults()
    dim = func_sign.arguments.get("dim", None)
    is_reduction = (prototype.data_dim is None) or dim in (
        None,
        prototype.data_dim,
        prototype.data_dim + prototype.ndim,
    )
    ret = func(*args, **kwargs)

    def _wrap(x):
        if not isinstance(x, torch.Tensor):
            return x
        if is_reduction:
            return DataTensor(x, header=[])
        return DataTensor(x, prototype.header, dim=prototype.data_dim)

    return tree_map(_wrap, ret)


@implements(torch.squeeze)
def _squeeze(input, dim):
    new_tensor = input.tensor.squeeze(dim)
    # Squeeze has no effect - early exit.
    if new_tensor.ndim == input.tensor.ndim:
        return input.new_datatensor(new_tensor)
    dim = dim if dim < 0 else dim - input.ndim
    if not input.data_dim:
        header, col_dim = (), None
    elif dim < input.data_dim:
        header = input.header
        col_dim = input.data_dim
    elif dim == input.data_dim:
        header = []
        col_dim = None
    else:
        header = input.header
        col_dim = input.data_dim + 1
        assert col_dim < 0
    return input.new_datatensor(new_tensor, header, dim=col_dim)


@implements(torch.unsqueeze)
def _unsqueeze(input, dim):
    new_tensor = input.tensor.unsqueeze(dim)
    dim = dim if dim < 0 else dim - input.ndim - 1
    if not input.data_dim or dim < input.data_dim:
        col_dim = input.data_dim
    else:
        col_dim = input.data_dim - 1
    return input.new_datatensor(new_tensor, input.header, dim=col_dim)


for _fn in POINTWISE_OPS:
    HANDLED_FNS[_fn] = functools.partial(_handle_pointwise_ops, _fn)

for _fn in REDUCTION_OPS:
    HANDLED_FNS[_fn] = functools.partial(_handle_reduction_ops, _fn)


class DataTensor(object):
    """
    Data structure used by time series model to store additional metadata with
    :class:`torch.Tensor` objects such as header (column) information and
    normalization. Note that the header information may be empty for scalar
    tensor or tensors that result from a reduction op. e.g.

    >>> d = DataTensor(torch.arange(4).view(2, 2), ['a', 'b'])
    DataTensor(tensor=tensor([[0, 1],
        [2, 3]]), header=('a', 'b'))
    >>> d[:, ['b']]
    DataTensor(tensor=tensor([[1],
        [3]]), header=('b',))
    >>> torch.sum(d, 0)
    DataTensor(tensor=tensor([2, 4]), header=('a', 'b'))
    >>> torch.sum(d)
    DataTensor(tensor=6, header=())

    :param tensor: a :class:`torch.Tensor` instance.
    :param header: list of column names having the same size as the innermost dim
        of `tensor`.
    :param normalize_cols: list of columns to normalize by scaling to 0 mean unit
        norm, or a dictionary of column names mapped to the corresponding
        :class:`torch.distributions.Transform`.
    :param dim: Dimension for column indexing. By default, it is the
        rightmost dimension.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        header: Iterable[str] = (),
        normalize_cols: Union[Dict[str, transforms.Transform], List[str]] = None,
        dim: int = -1,
    ):
        assert isinstance(tensor, torch.Tensor)
        self.reduced = not header
        self.scalar = tensor.ndim == 0
        # dim should be None for scalar or reduced DataTensors.
        if self.reduced or self.scalar:
            self.data_dim = None
        else:
            if dim < -tensor.ndim or dim >= tensor.ndim:
                raise IndexError(
                    f"dim={dim} invalid for a tensor of dim {tensor.ndim}."
                )
            self.data_dim = dim if dim < 0 else dim - tensor.ndim
        if self.scalar:
            if header:
                raise ValueError("Scalar DataTensor must have empty header.")
        elif (not self.reduced) and (len(header) != tensor.shape[self.data_dim]):
            raise ValueError(
                f"Tensor data dim size {tensor.shape[self.data_dim]} does not match header length {len(header)}."
            )
        self.tensor = tensor
        self.header = tuple(header)
        self.rev_index = {h: i for i, h in enumerate(header)}
        # Transforms applied to the columns
        self.transforms = {}
        self._normalized = set()
        if normalize_cols is not None:
            self.normalize(normalize_cols)

    def new_datatensor(self, tensor, header=None, dim=None):
        """
        Return a new DataTensor instance defaulting to the current instance
        for header/data_dim information.

        :param tensor: New underlying tensor data.
        :param header: Optional header information.
        :param dim: Dimension for column indexing. By default, we use this is
            the same as `self.data_dim`.
        """
        dim = self.data_dim if dim is None else dim
        header = self.header if header is None else header
        data_tensor = DataTensor(tensor, header, dim=dim)
        for h in data_tensor.header:
            if h in self.transforms:
                data_tensor.transforms[h] = self.transforms[h]
        return data_tensor

    def normalize(
        self, transform_columns: Union[Dict[str, transforms.Transform], List[str]]
    ):
        """
        Scale the specified columns using the transforms specified for each `column`.
        If `transform_columns` is a list, use standard normalization - scaling the data to
        0 mean and unit norm. Else, we can also provide a dict of column names
        mapped to the corresponding :class:`torch.distributions.Transform`.

        :param transform_columns: List of columns (subset of `self.header`), or
            a dictionary of column names mapped to the corresponding
            :class:`torch.distributions.Transform`.
        """
        if self.tensor.ndim <= 1:
            return
        if isinstance(transform_columns, dict):
            intersection = self._normalized & transform_columns.keys()
            if intersection:
                raise ValueError(f"Columns {intersection} already normalized.")
            self.transforms = transform_columns
        # Use default transform
        elif isinstance(transform_columns, list):
            for c in transform_columns:
                if c in self._normalized:
                    raise ValueError(f"Column {c} already normalized.")
                idx = self.tensor.new_full((), self.rev_index[c], dtype=torch.long)
                selected_column = self.tensor.index_select(self.data_dim, idx)
                mean = selected_column.mean()
                std = selected_column.std()
                transform = transforms.AffineTransform(mean, std).inv
                self.transforms[c] = transform
        else:
            raise TypeError(f"Invalid type {type(transform_columns)} for `columns`.")
        for c, transform in self.transforms.items():
            if c in self._normalized:
                continue
            if not isinstance(transform, transforms.Transform):
                raise TypeError(
                    f"Invalid transform type {type(transform)}. Must be an instance of "
                    f"`torch.distributions.Transform`."
                )
            idx = self.tensor.new_full((), self.rev_index[c], dtype=torch.long)
            selected_column = self.tensor.index_select(self.data_dim, idx)
            transformed_column = transform(selected_column)
            self.tensor.index_copy_(self.data_dim, idx, transformed_column)
            self._normalized.add(c)

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
            tensor = data.tensor.clone()
        else:
            tensor, header = data.clone(), None
        if (
            tensor.ndim != self.tensor.ndim
            or tensor.shape[self.data_dim] != self.tensor.shape[self.data_dim]
        ):
            raise ValueError(
                "Tensor's ndim and size of column dim must match the source tensor."
            )
        for c, transform in self.transforms.items():
            idx = tensor.new_full((), self.rev_index[c], dtype=torch.long)
            selected_column = tensor.index_select(self.data_dim, idx)
            transformed_column = transform.inv(selected_column)
            tensor.index_copy_(self.data_dim, idx, transformed_column)
        return tensor if header is None else self.new_datatensor(tensor, header)

    def get_index(self, key):
        """Get integer indices corresponding to the column names in `key`.

        :param key: A sequence, int or str values corresonding to indices in the
            to be selected from the `data_dim`.
        :return: Integer indices for (potential) column names referenced in `key`.
        """
        col_key = key
        if isinstance(key, str):
            if key not in self.rev_index:
                raise IndexError(f"key={key} not in {self.header}.")
            col_key = self.rev_index[key]
        elif isinstance(key, Sequence) and any(isinstance(k, str) for k in key):
            col_idxs = []
            for c in key:
                if c not in self.rev_index:
                    raise IndexError(f"key={c} not in {self.header}.")
                col_idxs.append(self.rev_index[c])
            # tensor[(0,)] doesn't behave the same as tensor[[0]]
            col_key = tuple(col_idxs) if isinstance(key, tuple) else col_idxs
        return col_key

    def clone(self):
        return self.new_datatensor(self.tensor.clone())

    def squeeze(self, dim):
        return torch.squeeze(self, dim)

    def unsqueeze(self, dim):
        return torch.unsqueeze(self, dim)

    def float(self):
        return self.new_datatensor(self.tensor.float())

    def long(self):
        return self.new_datatensor(self.tensor.long())

    def __add__(self, other):
        return torch.add(self, other)

    def __sub__(self, other):
        return torch.subtract(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __div__(self, other):
        return torch.divide(self, other)

    def __getattr__(self, name):
        return self.tensor.__getattribute__(name)

    @staticmethod
    def _is_index_type(key, t):
        return isinstance(key, t) or (
            isinstance(key, Sequence) and any(isinstance(k, t) for k in key)
        )

    def _forward_parse(self, key, parsed):
        """
        Parse key until we reach the end or an Ellipsis is encountered, while
        gathering the index of empty dims in the possibly reduced tensor and
        the number of reduced dims to the right of `self.data_dim`.
        """
        empty_dims = []
        data_dim = self.ndim + self.data_dim
        nonempty_idx = 0
        num_reduced = 0  # number of reduced dims
        num_reduced_ddim = 0  # number of reduced dims to the right of data_dim
        ellipsis_idx = len(key)
        for i, k in enumerate(key):
            if k is None:
                empty_dims.append(i - num_reduced)
            elif k is Ellipsis:
                ellipsis_idx = i
                break
            else:
                if self._is_index_type(k, str) and nonempty_idx != data_dim:
                    raise IndexError(
                        f"String indexing at dim {i} does not match self.data_dim={data_dim}."
                    )
                if isinstance(k, (str, int)):
                    num_reduced += 1
                    if nonempty_idx > data_dim:
                        num_reduced_ddim += 1
                parsed[nonempty_idx] = k
                nonempty_idx += 1
        return parsed, empty_dims, num_reduced_ddim, ellipsis_idx

    def _reverse_parse(self, key, parsed, stop):
        """
        Parse key in reverse order until we hit `stop` index, while
        gathering the index of empty dims in the possibly reduced tensor and
        the number of reduced dims to the right of `self.data_dim`. Note that
        indexing is done from the right starting from -1.
        """
        empty_dims = []
        nonempty_idx = -1
        num_reduced = 0  # number of reduced dims
        num_reduced_ddim = 0  # number of reduced dims to the right of data_dim
        for i, k in enumerate(reversed(key)):
            j = -i - 1
            if len(key) + j <= stop:
                break
            if k is None:
                empty_dims.append(j + num_reduced)
            elif k is Ellipsis:
                raise NotImplementedError(
                    "More than 1 ellipsis for indexing is not supported."
                )
            else:
                if self._is_index_type(k, str) and nonempty_idx != self.data_dim:
                    raise IndexError(
                        f"String indexing at dim {j} does not match self.data_dim={self.data_dim}."
                    )
                if isinstance(k, (str, int)):
                    num_reduced += 1
                    if nonempty_idx > self.data_dim:
                        num_reduced_ddim += 1
                parsed[nonempty_idx] = k
                nonempty_idx -= 1
        return parsed, empty_dims, num_reduced_ddim

    def _get_normalized_dims(self, key):
        """
        Remove Ellipsis and return key of the same size as `self.ndim`.
        Also returns (i) the indices for the empty (`None`) dims in the output
        tensor (possibly of a reduced dim), and (ii) the
        number of reduced dimensions after the `data_dim`.

        e.g. for a 3-D tensor:
        [..., "a"] --> [slice(None), slice(None), "a"]
        """
        if not isinstance(key, Sequence):
            if key is None:
                return tuple(slice(None) for _ in range(self.ndim)), [0], 0
            elif isinstance(key, (str, int, slice)):
                return (
                    (key,) + tuple(slice(None) for _ in range(self.ndim - 1)),
                    [],
                    0,
                )
            raise ValueError(f"Unsupported type {type(key)} for key.")
        # indexing along dim 0
        if isinstance(key, list):
            key = (key,) + tuple(slice(None) for _ in range(self.ndim - 1))
        for k in key:
            if isinstance(k, Sequence) and any(k_ is None for k_ in k):
                raise IndexError("Nested `None` indices are disallowed.")
        # Convert all indexing ops into multi-dim indexing with one
        # index per dim. e.g. (slice(None, None, None), ['a'], 1) for a 3D tensor.
        parsed = [slice(None)] * self.ndim
        parsed, empty_dims_lhs, num_reduced_lhs, ellipsis_idx = self._forward_parse(
            key, parsed
        )
        parsed, empty_dims_rhs, num_reduced_rhs = self._reverse_parse(
            key, parsed, ellipsis_idx
        )
        return (
            tuple(parsed),
            empty_dims_lhs + empty_dims_rhs,
            num_reduced_lhs + num_reduced_rhs,
        )

    def __getitem__(self, key):
        header = self.header
        # 1. If self.data_dim is None (scalar or reduced tensor),
        # exit early.
        if self.data_dim is None:
            return DataTensor(self.tensor[key])
        # 2. For boolean masked indexing or advanced indexing via LongTensor
        # exit early
        if self._is_index_type(key, (bool, torch.Tensor)):
            return DataTensor(self.tensor[key])

        # 3. Handle remaining cases by first getting a normalized key
        # having an entry for each dim, determining if the `data_dim`
        # is affected, and handling the header information appropriately.

        # Positive indexed dim for column indexing.
        data_dim = self.tensor.ndim + self.data_dim
        # Get normalized key (with entry for each dim)
        key_nonempty, empty_dims, num_reduced = self._get_normalized_dims(key)
        col_idxs = self.get_index(key_nonempty[data_dim])
        # Default
        tensor_key = key_nonempty
        header = self.header
        # Column indexing has occured via string/integer indexing.
        if self._is_index_type(col_idxs, int):
            tensor_key = (
                key_nonempty[:data_dim] + (col_idxs,) + key_nonempty[data_dim + 1 :]
            )
            header = (
                ()
                if isinstance(col_idxs, int)
                else tuple(self.header[c] for c in col_idxs)
            )
            # If all entries are ints => advanced indexing, exit early
            if self.ndim > 1 and all(self._is_index_type(k, int) for k in tensor_key):
                return DataTensor(self.tensor[tensor_key])
        # Column indexing has occured via a slice operator
        # Only need to adjust the header
        elif isinstance(col_idxs, slice):
            header = tuple(self.header[col_idxs])
        val = self.new_datatensor(
            self.tensor[tensor_key], header, dim=self.data_dim + num_reduced
        )
        # Handle None dims => apply unsqueezes
        for d in empty_dims:
            val = val.unsqueeze(d)
        return val

    def __len__(self):
        return len(self.tensor)

    def __repr__(self):
        return f"DataTensor(tensor={self.tensor}, header={self.header})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in HANDLED_FNS:
            return HANDLED_FNS[func](*args, **kwargs)

        # No special handling for these operations - the result from the
        # corresponding PyTorch ops is returned with empty header.
        args = tree_map(lambda x: x.tensor if isinstance(x, DataTensor) else x, args)
        ret = func(*args, **kwargs)
        return tree_map(
            lambda x: DataTensor(x, header=[]) if isinstance(x, torch.Tensor) else x,
            ret,
        )


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
        This is useful to convert custom pandas data types such as datetime values
        into a `torch.float` value.
    :param dtype: default dtype for the backing `torch.Tensor` object. Defaults
        to `torch.float32`.
    :param normalize_cols: flag to indicate whether columns should be normalized
        to 0 mean and unit norm. Two other possibilities are: (i) A list of columns
        can be passed instead in which case only the specified columns will be
        normallized. (ii) To have custom transforms per column, a dict of column names
        with the corresponding applicable :class:`torch.distributions.Transform` instance
        can be passed.
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
    if normalize_cols is True:
        normalize_cols = header
    elif normalize_cols is False:
        normalize_cols = []
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
    with torch.no_grad():
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
        var = scale_t**2 * var
        p1, p2 = torch.tensor((100 - ci) / 200.0), torch.tensor((100 + ci) / 200.0)
        mean_orig, var_orig = mean, var
        q1 = mean + (2 * var).sqrt() * torch.erfinv(2 * p1 - 1)
        q2 = mean + (2 * var).sqrt() * torch.erfinv(2 * p2 - 1)
        if exp_transform:
            mean_orig = torch.exp(mean + var / 2)
            var_orig = mean_orig**2 * (var.exp() - 1)
            q1 = torch.exp(q1)
            q2 = torch.exp(q2)
        return (mean_orig, var_orig, (q1, q2))
