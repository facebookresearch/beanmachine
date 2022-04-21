# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict

import torch
from beanmachine.ppl.experimental.vi.variational_world import VariationalWorld
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict
from beanmachine.ppl.world.initialize_fn import init_from_prior

"Gradient estimators of f-divergences."

_CPU_DEVICE = torch.device("cpu")
DiscrepancyFn = Callable[[torch.Tensor], torch.Tensor]

# NOTE: right now it is either all reparameterizable
# or all score function gradient estimators. We should
# be able to support both depending on the guide used.
def monte_carlo_approximate_reparam(
    observations: RVDict,
    num_samples: int,
    discrepancy_fn: DiscrepancyFn,
    queries_to_guides: Dict[RVIdentifier, RVIdentifier],
    params: RVDict,
    device: torch.device = _CPU_DEVICE,
) -> torch.Tensor:
    "The reparameterization trick gradient estimator."

    def init_from_guides_rsample(
        world: VariationalWorld, rv: RVIdentifier
    ) -> torch.Tensor:
        if rv not in queries_to_guides:
            return init_from_prior(world, rv)
        guide_rv = queries_to_guides[rv]
        guide_dist, _ = world._run_node(guide_rv)
        return guide_dist.rsample()

    loss = torch.zeros(1).to(device)
    for _ in range(num_samples):
        world = VariationalWorld(
            observations=observations,
            initialize_fn=init_from_guides_rsample,
            params=params,
        )._initialize_world(
            queries=queries_to_guides.keys(),
        )

        # form log density ratio logu = logp - logq
        logu = world.log_prob(observations.keys())
        for rv in queries_to_guides:
            # STL
            stop_grad_params = {p: params[p].detach().clone() for p in params}
            world.set_params(stop_grad_params)
            guide_dist, _ = world._run_node(queries_to_guides[rv])
            world.set_params(params)

            var = world.get_variable(rv)
            logu += (
                (var.distribution.log_prob(var.value) - guide_dist.log_prob(var.value))
                .sum()
                .squeeze()
            )
        loss += discrepancy_fn(logu)  # reparameterized estimator
    loss /= num_samples
    return loss


def monte_carlo_approximate_sf(
    observations: RVDict,
    num_samples: int,
    discrepancy_fn: DiscrepancyFn,
    queries_to_guides: Dict[RVIdentifier, RVIdentifier],
    params: RVDict,
    device: torch.device = _CPU_DEVICE,
) -> torch.Tensor:
    "The score function / log derivative trick gradient estimator."

    def init_from_guides_sample(
        world: VariationalWorld, rv: RVIdentifier
    ) -> torch.Tensor:
        if rv not in queries_to_guides:
            return init_from_prior(world, rv)
        guide_rv = queries_to_guides[rv]
        guide_dist, _ = world._run_node(guide_rv)
        return guide_dist.sample()

    loss = torch.zeros(1).to(device)
    for _ in range(num_samples):
        world = VariationalWorld(
            observations=observations,
            initialize_fn=init_from_guides_sample,
            params=params,
        )._initialize_world(
            queries=queries_to_guides.keys(),
        )

        # form log density ratio logu = logp - logq
        logu = world.log_prob(observations.keys())
        logq = torch.zeros(1)
        for rv in queries_to_guides:
            guide_dist, _ = world._run_node(queries_to_guides[rv])
            var = world.get_variable(rv)
            logu += (
                var.distribution.log_prob(var.value) - guide_dist.log_prob(var.value)
            ).squeeze()
            logq += guide_dist.log_prob(var.value)
        loss += discrepancy_fn(logu).detach().clone() * logq + discrepancy_fn(
            logu
        )  # score function estimator
    loss /= num_samples
    return loss
