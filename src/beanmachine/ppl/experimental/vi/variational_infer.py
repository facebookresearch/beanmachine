# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.optim as optim
from beanmachine.ppl.experimental.vi.discrepancy import kl_reverse
from beanmachine.ppl.experimental.vi.gradient_estimator import (
    monte_carlo_approximate_reparam,
)
from beanmachine.ppl.experimental.vi.variational_world import VariationalWorld
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.world import RVDict
from tqdm.auto import tqdm


_CPU_DEVICE = torch.device("cpu")


class VariationalInfer:
    def __init__(
        self,
        queries_to_guides: Dict[RVIdentifier, RVIdentifier],
        observations: RVDict,
        optimizer: Callable[
            [torch.Tensor], optim.Optimizer
        ] = lambda params: optim.Adam(params, lr=1e-2),
        device: torch.device = _CPU_DEVICE,
    ):
        """
        Performs variational inference using reparameterizable guides.

        Args:
            queries_to_guides: Pairing between random variables and their variational guide/surrogate
            observations: Observations as an RVDict keyed by RVIdentifier
            optimizer: A function returning a ``torch.Optimizer`` to use for optimizing variational parameters.
            device: a ``torch.device`` to use for pytorch tensors
        """
        super().__init__()

        self.observations = observations
        self.queries_to_guides = queries_to_guides

        # runs all guides to reify `param`s for `optimizer`
        # NOTE: assumes `params` is static and same across all worlds, consider MultiOptimizer (see Pyro)
        # TODO: what happens if not all the params are encountered
        # in this execution pass, eg an if/else, consider MultiOptimizer
        world = VariationalWorld(
            observations=observations,
            params={},
            queries_to_guides=queries_to_guides,
        )
        for guide in queries_to_guides.values():
            world.call(guide)
        self.params = world._params
        self._optimizer = optimizer(self.params.values())

    def infer(
        self,
        num_steps: int,
        num_samples: int = 1,
        discrepancy_fn=kl_reverse,
        mc_approx=monte_carlo_approximate_reparam,  # TODO: support both reparam and SF in same guide
        step_callback: Optional[
            Callable[[torch.Tensor, VariationalInfer], None]
        ] = None,
    ) -> VariationalWorld:
        """
        Perform variatonal inference.

        Args:
            num_steps: number of optimizer steps
            num_samples: number of samples per Monte-Carlo gradient estimate of E[f(logp - logq)]
            discrepancy_fn: discrepancy function f, use ``kl_reverse`` to minimize negative ELBO
            mc_approx: Monte-Carlo gradient estimator to use
            step_callback: callback function invoked each optimizer step

        Returns:
            VariationalWorld: A world with variational guide distributions
            initialized with optimized parameters
        """
        for _ in tqdm(range(num_steps)):
            loss, _ = self.step(num_samples, discrepancy_fn, mc_approx)
            if step_callback:
                step_callback(loss, self)

        return VariationalWorld.initialize_world(
            queries=self.queries_to_guides.values(),
            observations=self.observations,
            params=self.params,
            queries_to_guides=self.queries_to_guides,
            initialize_fn=lambda d: d.sample(),
        )

    def step(
        self,
        num_samples: int = 1,
        discrepancy_fn=kl_reverse,
        mc_approx=monte_carlo_approximate_reparam,  # TODO: support both reparam and SF in same guide
    ) -> Tuple[torch.Tensor, VariationalInfer]:
        """
        Perform one step of variatonal inference.

        Args:
            num_samples: number of samples per Monte-Carlo gradient estimate of E[f(logp - logq)]
            discrepancy_fn: discrepancy function f, use ``kl_reverse`` to minimize negative ELBO
            mc_approx: Monte-Carlo gradient estimator to use

        Returns:
            Tuple[torch.Tensor, VariationalInfer]: the loss value (before the
            step) and the ``VariationalInfer`` instance
        """
        self._optimizer.zero_grad()
        loss = mc_approx(
            self.observations,
            num_samples,
            discrepancy_fn,
            self.params,
            self.queries_to_guides,
        )
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            self._optimizer.step()
        else:
            logging.warn("Encountered NaN/inf loss, skipping step.")
        return loss, self
