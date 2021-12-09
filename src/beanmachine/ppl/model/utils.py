# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import torch


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
