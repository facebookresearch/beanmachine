# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(eq=True, frozen=True)
class RVIdentifier:
    function: Callable
    arguments: Tuple

    def __str__(self):
        return str(self.function.__name__) + str(self.arguments)
