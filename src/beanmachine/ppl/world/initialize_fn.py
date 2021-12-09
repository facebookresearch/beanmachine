# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16, 20]
from typing import Callable

import torch
import torch.distributions as dist


InitializeFn = Callable[[dist.Distribution], torch.Tensor]


def init_to_uniform(distribution: dist.Distribution) -> torch.Tensor:
    sample_val = distribution.sample()
    if distribution.has_enumerate_support:
        support = distribution.enumerate_support(expand=False).flatten()
        return support[torch.randint_like(sample_val, support.numel()).long()]
    elif not distribution.support.is_discrete:
        transform = dist.biject_to(distribution.support)
        return transform(torch.rand_like(transform.inv(sample_val)) * 4 - 2)
    else:
        # fall back to sample from prior
        return init_from_prior(distribution)


def init_from_prior(distribution: dist.Distribution) -> torch.Tensor:
    return distribution.sample()
