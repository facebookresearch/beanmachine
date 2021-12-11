# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(eq=True, frozen=True)
class RVIdentifier:
    """
    Struct representing the unique key corresponding to a
    BM random variable.
    """

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
