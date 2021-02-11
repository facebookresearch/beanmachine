# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(eq=True, frozen=True)
class RVIdentifier:
    wrapper: Callable
    arguments: Tuple

    def __str__(self):
        return str(self.function.__name__) + str(self.arguments)

    @property
    def function(self):
        return self.wrapper.__wrapped__

    @property
    def is_functional(self):
        w = self.wrapper
        assert hasattr(w, "is_functional")
        return w.is_functional

    @property
    def is_random_variable(self):
        w = self.wrapper
        assert hasattr(w, "is_random_variable")
        return w.is_random_variable
