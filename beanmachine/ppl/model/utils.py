# Copyright (c) Facebook, Inc. and its affiliates.
from collections import namedtuple
from enum import Enum

import torch


RandomVariable = namedtuple("RandomVariable", "function arguments")


class Mode(Enum):
    """
    Stages/modes that a model will go through
    """

    INITIALIZE = 1
    INFERENCE = 2


float_types = (torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor)
