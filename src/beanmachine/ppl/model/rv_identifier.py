# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch


@dataclass(eq=True, frozen=True)
class RVIdentifier:
    """
    Struct representing the unique key corresponding to a
    BM random variable.
    """

    wrapper: Callable
    arguments: Tuple

    def __post_init__(self):
        for arg in self.arguments:
            if torch.is_tensor(arg):
                warnings.warn(
                    "PyTorch tensors are hashed by memory address instead of value. "
                    "Therefore, it is not recommended to use tensors as indices of random variables.",
                    # display the warning on where the RVIdentifier is created
                    stacklevel=5,
                )

    def __str__(self):
        return str(self.function.__name__) + str(self.arguments)

    def __lt__(self, other: Any) -> bool:
        # define comparison so that functorch doesn't raise when it tries to
        # sort dictionary keys (https://fburl.com/0gomiv80). This can be
        # removed with the v0.2.1+ release of functorch.
        if isinstance(other, RVIdentifier):
            return str(self) < str(other)
        return NotImplemented

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
