# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16, 20]
from typing import Callable
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World

import torch
import torch.distributions as dist


InitializeFn = Callable[[World, RVIdentifier], torch.Tensor]


def init_to_uniform(world: World, rv: RVIdentifier) -> torch.Tensor:
    """
    Initializes a uniform distribution to sample from transformed to the
    support of ``distribution``.  A Categorical is used for discrete distributions,
    a bijective transform is used for constrained continuous distributions, and
    ``distribution`` is used otherwise.

    Used as an arg for ``World``

    Args:
        distribution: ``torch.distribution.Distribution`` of the RV, usually
                      the prior distribution.

    """
    distribution = world.get_variable(rv).distribution
    sample_val = distribution.sample()
    if distribution.has_enumerate_support:
        support = distribution.enumerate_support(expand=False).flatten()
        return support[torch.randint_like(sample_val, support.numel()).long()]
    elif not distribution.support.is_discrete:
        transform = dist.biject_to(distribution.support)
        return transform(torch.rand_like(transform.inv(sample_val)) * 4 - 2)
    else:
        # fall back to sample from prior
        return init_from_prior(world, rv)


def init_from_prior(world: World, rv: RVIdentifier) -> torch.Tensor:
    """
    Samples from the distribution.

    Used as an arg for ``World``

    Args:
        distribution: ``torch.distribution.Distribution`` corresponding to
                      the distribution to sample from
    """
    distribution = world.get_variable(rv).distribution
    return distribution.sample()
