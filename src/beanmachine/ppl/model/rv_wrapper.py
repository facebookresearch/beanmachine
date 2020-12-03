# Copyright (c) Facebook, Inc. and its affiliates.
import dataclasses
import inspect
from functools import update_wrapper
from typing import Callable

import beanmachine.ppl.world
from beanmachine.ppl.model.utils import RVIdentifier


@dataclasses.dataclass(eq=True)
class RVWrapper:
    """
    A class that wraps any function that's defined with @random_variable decorator, e.g.
    ```
    @random_variable
    def foo(i):
        return Normal(0, 1)

    print(type(foo)) # <- should be `RVWrapper`

    foo(1) # <- this will invoke `RVWrapper.__call__`
    ```
    """

    function: Callable
    is_random_variable: bool = True

    def __post_init__(self):
        """dataclasses.dataclass generates the __init__ for us and will run
        __post_init__ every time after __init__ is called"""
        # Copy metadata (docstring, etc.) from original function over to wrapper
        update_wrapper(wrapper=self, wrapped=self.function)

    def __get__(self, instance, owner=None):
        """
        In Python, calling `a.b` is basically equivalent to
        ```
        type(a).__dict__['b'].__get__(a, type(a))
        ```

        Additionally, class methods have type "function" at definition time; they only
        become "method" when we try to access `some_instance.some_method`... where
        `some_method.__get__` is invoked under the hood.

        So, by defining `RVWrapper.__get__`, we are able to obtain a reference to the
        instance object, e.g.
        ```
        class Foo:
            @bm.random_variable
            def bar(self):
                ...

        type(Foo.bar)  # <- should be RVWrapper (this is also a class attribute)
        type(Foo.bar.function)  # <- should be "function" (i.e. not bound)
        foo = Foo()
        type(foo.bar)  # <- should be another RVWrapper. This'll be a copy of Foo.bar
                       # with function replaced by a "method" (i.e. bound function)
        type(foo.bar.function)  # <- should be "method"
        type(foo.bar.function.__self__)  # <- should be foo
        ```

        Related reading:
        https://docs.python.org/3/reference/datamodel.html#invoking-descriptors
        """
        # "function" becomes "method" (with reference to "self") when __get__ is called
        # while trying to call __get__ on a "method" changes nothing
        bound_method = self.function.__get__(instance, owner)
        # dataclasses.replace return a new instance (without modifying the current one)
        return dataclasses.replace(self, function=bound_method)

    def __call__(self, *args):
        """The main body of the decorator, which will invoked on regular function call"""
        identifier = RVIdentifier(arguments=args, wrapper=self)
        world = beanmachine.ppl.world.world_context.get()

        if world:
            if self.is_random_variable:
                return world.update_graph(node=identifier)
            else:
                return world.update_functionals(node=identifier)
        else:
            return identifier

    def __hash__(self):
        return hash(self.function)

    @property
    def model(self):
        """A quicker way to retrieve a reference to the instance object"""
        if inspect.ismethod(self.function):
            return self.function.__self__
        return None
