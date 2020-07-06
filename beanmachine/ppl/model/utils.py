# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


@dataclass(frozen=True)
class RVIdentifier:
    function: Any
    arguments: Any

    def __str__(self):
        return str(self.function.__name__) + str(self.arguments)

    def __eq__(self, other):
        # NOTE: equality comparison on function is overridden because
        # deserialization may result in the function id changing; arguments are
        # still compared by id to ensure the same method on different instances
        # are not equal
        return (
            (self.function.__name__ == other.function.__name__)
            and (self.function.__code__ == other.function.__code__)
            and (self.function.__module__ == other.function.__module__)
            and (self.function.__globals__ == other.function.__globals__)
            and (self.function.__closure__ == other.function.__closure__)
            and self.arguments == other.arguments
        )

    def __hash__(self):
        return hash((self.function.__name__, self.function.__code__, self.arguments))


class Mode(Enum):
    """
    Stages/modes that a model will go through
    """

    INITIALIZE = 1
    INFERENCE = 2


class LogLevel(Enum):
    """
    Enum class mapping the logging levels to numeric values.
    """

    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG_UPDATES = 16
    DEBUG_PROPOSER = 14
    DEBUG_GRAPH = 12


def get_beanmachine_logger(
    console_level: LogLevel = LogLevel.WARNING, file_level: LogLevel = LogLevel.INFO
) -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level.value)
    file_handler = logging.FileHandler("beanmachine.log")
    file_handler.setLevel(file_level.value)

    logger = logging.getLogger("beanmachine")
    logger.setLevel(
        file_level.value
        if file_level.value < console_level.value
        else console_level.value
    )
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


float_types = (torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor)
