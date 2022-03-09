# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from typing import Dict
from beanmachine.ppl.world.initialize_fn import init_from_prior

import torch
import torch.optim as optim
from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.world import RVDict, World
from tqdm.auto import tqdm


def kl_reverse(logu):
    """
    Reverse KL-divergence D(q||p) in log-space.

    Args:
        logu: `p.log_prob`s evaluated at samples from q
    """
    return -logu

def kl_forward(logu):
    return torch.exp(logu) * logu


def monte_carlo_approximate_reparam(
    observations,
    num_samples: int,
    discrepancy_fn,
    queries_to_guides,
    params,
) -> torch.Tensor:
    def init_from_guides_rsample(world: World, rv: RVIdentifier):
        if rv not in queries_to_guides:
            return init_from_prior(world, rv)
        guide_rv = queries_to_guides[rv]
        guide_dist, _ = world._run_node(guide_rv)
        return guide_dist.rsample()

    loss = torch.zeros(1)
    for _ in range(num_samples):
        world = BaseInference._initialize_world(queries_to_guides.keys(), observations, initialize_fn=init_from_guides_rsample, params=params)

        # form log density ratio logu = logp - logq
        logu = world.log_prob(observations.keys())
        print(logu)
        for rv in queries_to_guides:
            guide_dist, _ = world._run_node(queries_to_guides[rv])
            print(guide_dist)
            var = world.get_variable(rv)
            logu += (var.distribution.log_prob(var.value) - guide_dist.log_prob(var.value)).sum().squeeze()
        loss += discrepancy_fn(logu)  # reparameterized estimator
    loss /= num_samples
    return loss

def monte_carlo_approximate_sf(
    observations,
    num_samples: int,
    discrepancy_fn,
    queries_to_guides,
    params,
) -> torch.Tensor:
    def init_from_guides_sample(world: World, rv: RVIdentifier):
        if rv not in queries_to_guides:
            return init_from_prior(world, rv)
        guide_rv = queries_to_guides[rv]
        guide_dist, _ = world._run_node(guide_rv)
        return guide_dist.sample()

    loss = torch.zeros(1)
    for _ in range(num_samples):
        world = BaseInference._initialize_world(queries_to_guides.keys(), observations, initialize_fn=init_from_guides_sample, params=params)

        # form log density ratio logu = logp - logq
        logu = world.log_prob(observations.keys())
        logq = torch.zeros(1)
        for rv in queries_to_guides:
            guide_dist, _ = world._run_node(queries_to_guides[rv])
            var = world.get_variable(rv)
            logu += (var.log_prob - guide_dist.log_prob(var.value)).squeeze()
            logq += guide_dist.log_prob(var.value)
        loss += discrepancy_fn(logu).detach().clone() * logq # score function estimator + STL
    loss /= num_samples
    return loss


class VariationalInfer:
    def __init__(self) -> None:
        super().__init__()

    def infer(
        self,
        queries_to_guides: Dict[RVIdentifier, RVIdentifier],
        observations: RVDict,
        num_steps: int,
        optimizer = lambda params: optim.Adam(params, lr=1e-2),
        num_samples: int = 1,
        discrepancy_fn = kl_reverse,
        mc_approx = monte_carlo_approximate_reparam, # TODO: support both reparam and SF in same guide
    ):
        """
        Performs variational inference using reparameterizable guides.

        Args:
            queries_to_guides: Pairing between random variables and their variational guide/surrogate
            observations: Observations as an RVDict keyed by RVIdentifier
            num_steps: Number of steps of stochastic variational inference to perform.
            optimizer: A `torch.Optimizer` to use for optimizing variational parameters.
            num_samples: Number of samples used to Monte-Carlo approximate the discrepancy.
            discrepancy_fn: f-divergence to optimize.
            mc_approx: Monte Carlo gradient estimator to use
        """
        # runs all guides to reify `param`s for `optimizer`
        # NOTE: assumes `params` is static and same across all worlds, consider MultiOptimizer (see Pyro)
        params = {}
        world = World(observations, params=params)
        for guide in queries_to_guides.values():
            world.call(guide)
        opt = optimizer(params.values())

        for _ in tqdm(range(num_steps)):
            opt.zero_grad()
            loss = mc_approx(observations, num_samples, discrepancy_fn, queries_to_guides, params)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                opt.step()

        # update all guides' cached `Variable.distribution` to use latest `param`s so `.get_variable``
        # returns correct distributions
        for guide in queries_to_guides.values():
            world.initialize_value(guide)
        return world
