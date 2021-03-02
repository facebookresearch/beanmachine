# Copyright (c) Facebook, Inc. and its affiliates.
"""Operations that are intended as hints to the Beanstalk compiler"""

import math

import torch


def math_log1mexp(x):
    return math.log(1.0 - math.exp(x))


def log1mexp(x):
    return torch.log(1.0 - torch.exp(x))
