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
    else:
        transform = dist.biject_to(distribution.support)
        return transform(torch.rand_like(transform.inv(sample_val)) * 4 - 2)


def init_from_prior(distribution: dist.Distribution) -> torch.Tensor:
    return distribution.sample()
