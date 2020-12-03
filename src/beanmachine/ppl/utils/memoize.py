# Copyright (c) Facebook, Inc. and its affiliates.
import dataclasses
from functools import update_wrapper
from typing import Any, Callable, Dict, List

from beanmachine.ppl.model import RVIdentifier
from torch import Tensor


def _get_memoization_key(wrapper, args):
    # The problem is that tensors can only be compared for equality with
    # torch.equal(t1, t2), and tensors do not hash via value equality.
    # If we have an argument that is a tensor, we'll replace it with
    # the tensor as a string and hope for the best.
    new_args = tuple(str(a) if isinstance(a, Tensor) else a for a in args)
    return RVIdentifier(arguments=new_args, wrapper=wrapper)


class RecursionError(Exception):
    pass


def memoize(f):
    """
    Decorator to be used to memoize arbitrary functions.
    """
    return MemoizeWrapper(function=f)


@dataclasses.dataclass(eq=True)
class MemoizeWrapper:
    """
    The actual class that decorates the function to memoize
    """

    function: Callable
    cache: Dict[Any, Any] = dataclasses.field(default_factory=dict)
    # TODO: Can we use a more efficient type than a list? We don't know
    # TODO: if the key is hashable.
    in_flight: List[Any] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """dataclasses.dataclass generates the __init__ for us and will run
        __post_init__ every time after __init__ is called"""
        # update metadata on the wrapper object
        update_wrapper(wrapper=self, wrapped=self.function)

    def __get__(self, instance, owner=None):
        """Bind a reference to the instance object to self.funtion. Please refer to
        beanmachine.ppl.model.rv_wrapper for details about the usage of this function"""
        bound_method = self.function.__get__(instance, owner)
        return dataclasses.replace(self, function=bound_method)

    def __call__(self, *args):
        """Main body of the decorator."""
        key = _get_memoization_key(self, args)
        if key not in self.cache:
            if key in self.in_flight:
                # TODO: Better error
                raise RecursionError()
            self.in_flight.append(key)
            try:
                result = self.function(*args)
                self.cache[key] = result
            finally:
                self.in_flight.pop()
        return self.cache[key]

    def __hash__(self):
        return hash(self.function)
