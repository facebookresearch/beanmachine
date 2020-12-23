# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
from functools import wraps
from typing import Any, Dict, List

from beanmachine.ppl.model import StatisticalModel
from beanmachine.ppl.utils.item_counter import ItemCounter
from torch import Tensor


def _get_memoization_key(wrapper, args):
    # The problem is that tensors can only be compared for equality with
    # torch.equal(t1, t2), and tensors do not hash via value equality.
    # If we have an argument that is a tensor, we'll replace it with
    # the tensor as a string and hope for the best.
    new_args = tuple(str(a) if isinstance(a, Tensor) else a for a in args)
    return StatisticalModel.get_func_key(wrapper, new_args)


class RecursionError(Exception):
    pass


total_memoized_functions = 0
total_memoized_calls = 0
total_cache_misses = 0
count_calls = False
function_calls = ItemCounter()


def memoizer_report():
    call_report = [
        f"{item.__name__}: {count}\n" for (item, count) in function_calls.items.items()
    ]
    return (
        f"funcs: {total_memoized_functions} "
        + f"calls: {total_memoized_calls} "
        + f"misses: {total_cache_misses}\n"
        + "".join(call_report)
    )


def memoize(f):
    """
    Decorator to be used to memoize arbitrary functions.
    """

    global total_memoized_functions
    total_memoized_functions += 1

    cache: Dict[Any, Any] = {}
    # TODO: Can we use a more efficient type than a list? We don't know
    # TODO: if the key is hashable.
    in_flight: List[Any] = []

    @wraps(f)
    def wrapper(*args):
        if count_calls:
            global total_memoized_calls
            total_memoized_calls += 1
            function_calls.add_item(f)

        key = _get_memoization_key(wrapper, args)
        if key not in cache:
            global total_cache_misses
            total_cache_misses += 1
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

    if inspect.ismethod(f):
        meth_name = f.__name__ + "_wrapper"
        setattr(f.__self__, meth_name, wrapper)
    else:
        f._wrapper = wrapper
    return wrapper
