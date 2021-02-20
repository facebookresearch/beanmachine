# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Tuple

from beanmachine.ppl.utils.item_counter import ItemCounter
from torch import Tensor


def _tuplify(t: Any) -> Any:
    if isinstance(t, list):
        return tuple(_tuplify(y) for y in t)
    return t


class MemoizationKey:
    # It would be nice to just use a tuple (wrapper, args) for the memoization
    # key, but tensors can only be compared for equality with torch.equal(t1, t2),
    # and tensors do not hash via value equality.
    #
    # We therefore replace tensors with tuples that contain all the values of the
    # tensor.  For example, if our arguments are (1, tensor([2, 3]), 4) then our
    # new arguments are (1, (2, 3), 4)

    wrapper: Callable
    arguments: Tuple
    hashcode: int

    def __init__(self, wrapper: Callable, arguments: Tuple) -> None:
        self.arguments = (
            wrapper,
            tuple(
                _tuplify(a.tolist()) if isinstance(a, Tensor) else a for a in arguments
            ),
        )
        self.wrapper = wrapper
        self.hashcode = hash(self.arguments)

    def __hash__(self) -> int:
        return self.hashcode

    def __eq__(self, o) -> bool:
        return (
            isinstance(o, MemoizationKey)
            and self.hashcode == o.hashcode
            and self.wrapper == o.wrapper
            and self.arguments == o.arguments
        )


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

    @wraps(f)
    def wrapper(*args):
        if count_calls:
            global total_memoized_calls
            total_memoized_calls += 1
            function_calls.add_item(f)

        key = MemoizationKey(wrapper, args)
        if key not in cache:
            global total_cache_misses
            total_cache_misses += 1
            result = f(*args)
            cache[key] = result
            return result
        return cache[key]

    if inspect.ismethod(f):
        meth_name = f.__name__ + "_wrapper"
        setattr(f.__self__, meth_name, wrapper)
    else:
        f._wrapper = wrapper
    return wrapper
