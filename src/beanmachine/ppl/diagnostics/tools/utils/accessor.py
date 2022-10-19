# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Accessor definition for extending Bean Machine `MonteCarloSamples` objects.

These methods are heavily influenced by the implementations by pandas and xarray.

- `pandas`: https://github.com/pandas-dev/pandas/blob/main/pandas/core/accessor.py
- `xarray`: https://github.com/pydata/xarray/blob/main/xarray/core/extensions.py
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Callable

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


class CachedAccessor:
    """
    A descriptor for caching accessors.

    Args:
        name (str): Namespace for the accessor.
        accessor (object): Class that defines the extension methods.

    Attributes:
        _name (str): Namespace for the accessor.
        _accessor (object): Class that defines the extension methods.

    Raises:
        RuntimeError: Returned if attempting to overwrite an existing accessor on the
            object.
    """

    def __init__(self: CachedAccessor, name: str, accessor: object) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self: CachedAccessor, obj: object, cls: object) -> object:
        """
        Method to retrieve the accessor namespace.

        Args:
            obj (object): Object that the accessor is attached to.
            cls (object): Needed for registering the accessor.

        Returns:
            object: The accessor object.
        """
        # Accessing an attribute of the class.
        if obj is None:
            return self._accessor

        try:
            cache = obj._cache  # type: ignore
        except AttributeError:
            cache = obj._cache = {}

        try:
            return cache[self._name]
        except KeyError:
            contextlib.suppress(KeyError)

        try:
            accessor_obj = self._accessor(obj)  # type: ignore
        except Exception as error:
            msg = f"error initializing {self._name!r} accessor."
            raise RuntimeError(msg) from error

        cache[self._name] = accessor_obj
        return accessor_obj  # noqa: R504 (unnecessary variable assignment)


def _register_accessor(name: str, cls: object) -> Callable:
    """
    Method used for registering an accessor to a given object.

    Args:
        name (str): The name for the accessor.
        cls (object): The object the accessor should be attached to.

    Returns:
        Callable: A decorator for creating accessors.

    Raises:
        RuntimeError: Returned if attempting to overwrite an existing accessor on the
            object.
    """

    def decorator(accessor: object) -> object:
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
                stacklevel=2,
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_mcs_accessor(name: str) -> Callable:
    """
    Register an accessor object for `MonteCarloSamples` objects.

    Args:
        name (str): The name for the accessor.

    Returns:
        Callable: A decorator for creating the `MonteCarloSamples` accessor.

    Raises:
        RuntimeError: Returned if attempting to overwrite an existing accessor on the
            object.

    Example:
        >>> from __future__ import annotations
        >>> from typing import Dict, List
        >>>
        >>> import beanmachine.ppl as bm
        >>> import numpy as np
        >>> import torch.distributions as dist
        >>> from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
        >>> from beanmachine.ppl.diagnostics.tools.utils import accessor
        >>> from torch import tensor
        >>>
        >>> @bm.random_variable
        >>> def alpha():
        >>>     return dist.Normal(0, 1)
        >>>
        >>> @bm.random_variable
        >>> def beta():
        >>>     return dist.Normal(0, 1)
        >>>
        >>> @accessor.register_mcs_accessor("magic")
        >>> class MagicAccessor:
        >>>     def __init__(self: MagicAccessor, mcs: MonteCarloSamples) -> None:
        >>>         self.mcs = mcs
        >>>     def show_me(self: MagicAccessor) -> Dict[str, List[List[float]]]:
        >>>         # Return a JSON serializable object from a MonteCarloSamples object.
        >>>         return dict(
        >>>             sorted(
        >>>                 {
        >>>                     str(key): value.tolist()
        >>>                     for key, value in self.mcs.items()
        >>>                 }.items(),
        >>>                 key=lambda item: item[0],
        >>>             ),
        >>>         )
        >>>
        >>> chain_results = {
        >>>     beta(): tensor([4, 3], [2, 1]),
        >>>     alpha(): tensor([[1, 2], [3, 4]]),
        >>> }
        >>> samples = MonteCarloSamples(chain_results=chain_results)
        >>> samples.magic.show_me()
        {'alpha()': [[1, 2], [3, 4]], 'beta()': [[4, 3], [2, 1]]}
    """
    return _register_accessor(name, MonteCarloSamples)
