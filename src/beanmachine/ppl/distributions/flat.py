# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist


class Flat(dist.Distribution):
    """
    Flat distribution used as an improper prior. Samples 0 with
    shape sample shape

    Args
        shape: pass a tuple, and give a shape of Flat prior.
    """

    has_enumerate_support = False
    support = dist.constraints.real
    has_rsample = True

    def __init__(self, shape):
        self.shape = shape
        self._event_shape = [shape]

    def rsample(self, sample_shape):
        return torch.zeros(sample_shape)

    def sample(self):
        return torch.zeros(self.shape)

    def log_prob(self, value):
        return torch.tensor(0.0)
