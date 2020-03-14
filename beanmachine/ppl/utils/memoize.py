# Copyright (c) Facebook, Inc. and its affiliates.
from functools import wraps
from typing import Any, Dict

from beanmachine.ppl.model import StatisticalModel


_get_memoization_key = StatisticalModel.get_func_key


def memoize(f):
    """
    Decorator to be used to memoize arbitrary functions.
    """

    cache: Dict[Any, Any] = {}

    @wraps(f)
    def wrapper(*args):
        key = _get_memoization_key(f, args)
        if key not in cache:
            result = f(*args)
            cache[key] = result
        return cache[key]

    f._wrapper = wrapper
    return wrapper
