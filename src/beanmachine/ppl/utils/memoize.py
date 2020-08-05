# Copyright (c) Facebook, Inc. and its affiliates.
from functools import wraps
from typing import Any, Dict, List

from beanmachine.ppl.model import StatisticalModel
from torch import Tensor


def _get_memoization_key(f, args):
    # The problem is that tensors can only be compared for equality with
    # torch.equal(t1, t2), and tensors do not hash via value equality.
    # If we have an argument that is a tensor, we'll replace it with
    # the tensor as a string and hope for the best.
    new_args = tuple(str(a) if isinstance(a, Tensor) else a for a in args)
    return StatisticalModel.get_func_key(f, new_args)


class RecursionError(Exception):
    pass


def memoize(f):
    """
    Decorator to be used to memoize arbitrary functions.
    """

    cache: Dict[Any, Any] = {}
    # TODO: Can we use a more efficient type than a list? We don't know
    # TODO: if the key is hashable.
    in_flight: List[Any] = []

    @wraps(f)
    def wrapper(*args):
        key = _get_memoization_key(f, args)
        if key not in cache:
            if key in in_flight:
                # TODO: Better error
                raise RecursionError()
            in_flight.append(key)
            try:
                result = f(*args)
                cache[key] = result
            finally:
                in_flight.pop()
        return cache[key]

    f._wrapper = wrapper
    return wrapper
