# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


@dataclass(eq=True, frozen=True)
class RVIdentifier:
    function: Any
    arguments: Any

    def __str__(self):
        return str(self.function.__name__) + str(self.arguments)


class Mode(Enum):
    """
    Stages/modes that a model will go through
    """

    INITIALIZE = 1
    INFERENCE = 2


float_types = (torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor)
