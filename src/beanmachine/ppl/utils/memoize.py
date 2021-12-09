# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Tuple

from beanmachine.ppl.utils.item_counter import ItemCounter
from torch import Tensor


def _tuplify(t: Any) -> Any:
    if isinstance(t, list):
        return tuple(_tuplify(y) for y in t)
    return t


# This returns a tuple or value whose shape is the same as the input tensor.
# That is:
#
# tensor(1)                --> 1
# tensor([])               --> ()
# tensor([1])              --> (1,)
# tensor([1, 2])           --> (1, 2)
# tensor([[1, 2], [3, 4]]) --> ((1, 2), (3, 4))
#
# and so on
def tensor_to_tuple(t: Tensor) -> Any:
    result = _tuplify(t.tolist())
    return result


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
                tensor_to_tuple(a) if isinstance(a, Tensor) else a for a in arguments
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


# In Python, how do we memoize a constructor to ensure that instances of
# a class with the same constructor arguments are reference-equal? We could
# put the @memoize attribute on the class, but this leads to many problems.
#
# ASIDE: What problems? And why?
#
# A class is a function that constructs instances; a decorator is a function
# from functions to functions. "@memoize class C: ..." passes the instance-
# construction function to the decorator and assigns the result to C; this means
# that C is no longer a *type*; it is the *function* returned by the decorator.
# This in turn means that "instanceof(c, C)" no longer works because C is not a type.
# Similarly C cannot be a base class because it is not a type. And so on.
#
# END ASIDE
#
# The correct way to do this in Python is to create a metaclass. A class is a factory
# for instances; a metaclass is a factory for classes. We can create a metaclass which
# produces classes that are memoized.
#
# The default metaclass in Python is "type"; if you call "type(name, bases, attrs)"
# where name is the name of the new type, bases is a tuple of base types, and attrs
# is a dictionary of name-value pairs, then you get back a new class with that name,
# base classes, and attributes.  We can derive from type to make new metaclasses:


class MemoizedClass(type):
    # __new__ is called when the metaclass creates a new class.
    # metacls is the "self" of the metaclass
    # name is the name of the class we're creating
    # bases is a tuple of base types
    # attrs is a dictionary of attributes
    def __new__(
        metacls, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]
    ) -> type:
        # The memoized values will be stored in a per-class dictionary called
        # _cache, so make sure that the attributes dictionary has that.
        if "_cache" not in attrs:
            attrs["_cache"] = {}
        # That's the only special thing we need to do, so defer the actual
        # type creation to the "type" metaclass -- our base type.
        return super(MemoizedClass, metacls).__new__(metacls, name, bases, attrs)

    # A class is a function which constructs instances; when that function is
    # called to construct an instance, the __call__ handler is invoked in the
    # metaclass. By default type.__call__ simply creates a new instance. We
    # can replace that behavior by overriding the __call__ handler to do something
    # else.
    #
    # cls is the class that we are trying to create an instance of; *args is
    # the argument list passed to the constructor.
    def __call__(cls, *args):
        # TODO: We do not collect statistics on memoization use here.
        # TODO: We do not canonicalize arguments as the memoizer does above.
        if args not in cls._cache:
            # This is the first time we've constructed this class with these
            # arguments. Defer to the __call__ behavior of "type", which is the
            # superclass of this metaclass.
            new_instance = super(MemoizedClass, cls).__call__(*args)
            cls._cache[args] = new_instance
            return new_instance
        return cls._cache[args]


# You then use this as
# class Foo(FooBase, metaclass=MemoizedClass): ...
