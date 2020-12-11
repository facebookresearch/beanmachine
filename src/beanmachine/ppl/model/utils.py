# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
from enum import Enum

import torch


def get_wrapper(f):
    """
    Gets wrapped function given a Python callable
    """
    if inspect.ismethod(f):
        return getattr(f.__self__, f.__name__ + "_wrapper")
    return f._wrapper


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
