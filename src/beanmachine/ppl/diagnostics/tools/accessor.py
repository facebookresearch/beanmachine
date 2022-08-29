"""Accessor definition for extending Bean Machine `MonteCarloSamples` objects."""
from __future__ import annotations

import contextlib
import warnings
from typing import Callable, TypeVar

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


T = TypeVar("T", bound="CachedAccessor")


class CachedAccessor:
    """A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``samples.accessor_name``.
    accessor : cls
        Class with the extension methods.
    """

    def __init__(self: T, name: str, accessor: object) -> None:
        """Initialize."""
        self._name = name
        self._accessor = accessor

    def __get__(self: T, obj: object, cls: object) -> object:
        """Access the accessor object."""
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
        return accessor_obj  # noqa: R504


def _register_accessor(name: str, cls: object) -> Callable:
    """Register the accessor to the object."""

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
    """Register the accessor to object."""
    return _register_accessor(name, MonteCarloSamples)
