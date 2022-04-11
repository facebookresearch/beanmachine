"""Accessor definition for extending Bean Machine `MonteCarloSamples` objects."""
import warnings

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples


class AccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class _CachedAccessor:
    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor

        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}

        try:
            return cache[self._name]
        except KeyError:
            pass

        try:
            accessor_obj = self._accessor(obj)
        except Exception as error:
            raise RuntimeError(
                f"error initializing {self._name!r} accessor."
            ) from error

        cache[self._name] = accessor_obj
        return accessor_obj


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for"
                f" type {cls!r} is overriding a preexisting attribute with the "
                " same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor

    return decorator


def register_mcs_accessor(name: str):
    """Register the accessor to object."""
    return _register_accessor(name, MonteCarloSamples)
