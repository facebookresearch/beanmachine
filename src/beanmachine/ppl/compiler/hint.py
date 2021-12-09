# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Operations that are intended as hints to the Beanstalk compiler"""

import math

import torch


def math_log1mexp(x):
    return math.log(1.0 - math.exp(x))


def log1mexp(x):
    return torch.log(1.0 - torch.exp(x))
