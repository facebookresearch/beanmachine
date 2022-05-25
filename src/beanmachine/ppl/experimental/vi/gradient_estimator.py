# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"Gradient estimators of f-divergences."

from typing import Callable, Mapping

import torch
from beanmachine.ppl.experimental.vi.variational_world import VariationalWorld
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict, World


_CPU_DEVICE = torch.device("cpu")
DiscrepancyFn = Callable[[torch.Tensor], torch.Tensor]


# NOTE: right now it is either all reparameterizable
# or all score function gradient estimators. We should
# be able to support both depending on the guide used.
def monte_carlo_approximate_reparam(
    observations: RVDict,
    num_samples: int,
    discrepancy_fn: DiscrepancyFn,
    params: Mapping[RVIdentifier, torch.Tensor],
    queries_to_guides: Mapping[RVIdentifier, RVIdentifier],
    device: torch.device = _CPU_DEVICE,
) -> torch.Tensor:
    """The pathwise derivative / reparameterization trick
    (https://arxiv.org/abs/1312.6114) gradient estimator."""

    loss = torch.zeros(1).to(device)
    for _ in range(num_samples):
        variational_world = VariationalWorld.initialize_world(
            queries=queries_to_guides.values(),
            observations=observations,
            initialize_fn=lambda d: d.rsample(),
            params=params,
            queries_to_guides=queries_to_guides,
        )
        world = World.initialize_world(
            queries=[],
            observations={
                **{
                    query: variational_world[guide]
                    for query, guide in queries_to_guides.items()
                },
                **observations,
            },
        )

        # form log density ratio logu = logp - logq
        logu = world.log_prob() - variational_world.log_prob(queries_to_guides.values())

        loss += discrepancy_fn(logu)  # reparameterized estimator
    return loss / num_samples


def monte_carlo_approximate_sf(
    observations: RVDict,
    num_samples: int,
    discrepancy_fn: DiscrepancyFn,
    params: Mapping[RVIdentifier, torch.Tensor],
    queries_to_guides: Mapping[RVIdentifier, RVIdentifier],
    device: torch.device = _CPU_DEVICE,
) -> torch.Tensor:
    """The score function / log derivative trick surrogate loss
    (https://arxiv.org/pdf/1506.05254) gradient estimator."""

    loss = torch.zeros(1).to(device)
    for _ in range(num_samples):
        variational_world = VariationalWorld.initialize_world(
            queries=queries_to_guides.values(),
            observations=observations,
            initialize_fn=lambda d: d.sample(),
            params=params,
            queries_to_guides=queries_to_guides,
        )
        world = World.initialize_world(
            queries=[],
            observations={
                **{
                    query: variational_world[guide]
                    for query, guide in queries_to_guides.items()
                },
                **observations,
            },
        )

        # form log density ratio logu = logp - logq
        logq = variational_world.log_prob(queries_to_guides.values())
        logu = world.log_prob() - logq

        # score function estimator surrogate loss
        loss += discrepancy_fn(logu).detach().clone() * logq + discrepancy_fn(logu)
    return loss / num_samples
