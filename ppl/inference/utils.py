# Copyright (c) Facebook, Inc. and its affiliates.
from enum import Enum


class Mode(Enum):
    """
    stages/modes that an inference algorithm will go through
    """

    INITIALIZE = 1
    OBSERVE = 2
    INFERENCE = 3
