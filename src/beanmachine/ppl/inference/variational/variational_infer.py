# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from typing import Dict

import torch.optim as optim
from beanmachine.ppl.inference.base_inference import BaseInference
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.world import RVDict, InitializeFn, World, init_from_prior 
import torch
from tqdm.auto import tqdm


def kl_reverse(logu):
    """
    Reverse KL-divergence D(q||p) in log-space.

    Args:
        logu: `p.log_prob`s evaluated at samples from q
    """
    return -logu

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
        num_importance_samples: int = 1,
        discrepancy_fn = kl_reverse,
    ):
        """
        Performs variational inference.

        Args:
            queries_to_guides: Pairing between random variables and their variational guide/surrogate
            observations: Observations as an RVDict keyed by RVIdentifier
            num_steps: Number of steps of stochastic variational inference to perform.
            optimizer: A `torch.Optimizer` to use for optimizing variational parameters.
            num_samples: Number of samples used to Monte-Carlo approximate the discrepancy.
            num_importance_samples: Number of samples used to form an importance-weighted divergence.
            discrepancy_fn: f-divergence to optimize.
        """

        def init_from_guides(world: World, rv: RVIdentifier):
            guide_rv = queries_to_guides[rv]
            guide_dist, _ = world._run_node(guide_rv)
            return guide_dist.rsample()

        params = {}
        world = BaseInference._initialize_world(queries_to_guides.keys(), observations, initialize_fn=init_from_guides, params=params)
        opt = optimizer(params.values())
        for _ in tqdm(range(num_steps)):
            opt.zero_grad()
            world = BaseInference._initialize_world(queries_to_guides.keys(), observations, initialize_fn=init_from_guides, params=params)

            # form discrepancy
            loss = kl_reverse(world.log_prob(
                itertools.chain(queries_to_guides.keys(), observations.keys())))

            # add entropy H(q) term for ELBO
            # TODO: how does tfp hide this, activity regularization?
            for rv in queries_to_guides:
                guide_dist, _ = world._run_node(queries_to_guides[rv])
                loss += guide_dist.log_prob(world.get_variable(rv).value)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                opt.step()

        return world
