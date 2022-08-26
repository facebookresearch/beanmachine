# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
import torch.optim as optim
from beanmachine.ppl.inference.vi.discrepancy import kl_reverse
from beanmachine.ppl.inference.vi.gradient_estimator import (
    monte_carlo_approximate_reparam,
)
from beanmachine.ppl.inference.vi.variational_world import VariationalWorld
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
        self._device = device

    def infer(
        self,
        num_steps: int,
        num_samples: int = 1,
        discrepancy_fn=kl_reverse,
        mc_approx=monte_carlo_approximate_reparam,  # TODO: support both reparam and SF in same guide
        step_callback: Optional[
            Callable[[int, torch.Tensor, VariationalInfer], None]
        ] = None,
        subsample_factor: float = 1,
    ) -> VariationalWorld:
        """
        Perform variatonal inference.

        Args:
            num_steps: number of optimizer steps
            num_samples: number of samples per Monte-Carlo gradient estimate of E[f(logp - logq)]
            discrepancy_fn: discrepancy function f, use ``kl_reverse`` to minimize negative ELBO
            mc_approx: Monte-Carlo gradient estimator to use
            step_callback: callback function invoked each optimizer step
            subsample_factor: subsampling factor used for subsampling, helps scale the observations to avoid overshrinking towards the prior

        Returns:
            VariationalWorld: A world with variational guide distributions
            initialized with optimized parameters
        """
        assert subsample_factor > 0 and subsample_factor <= 1
        for it in tqdm(range(num_steps)):
            loss = self.step(num_samples, discrepancy_fn, mc_approx, subsample_factor)
            if step_callback:
                step_callback(it, loss, self)

        return self.initialize_world()

    def step(
        self,
        num_samples: int = 1,
        discrepancy_fn=kl_reverse,
        mc_approx=monte_carlo_approximate_reparam,  # TODO: support both reparam and SF in same guide
        subsample_factor: float = 1,
    ) -> torch.Tensor:
        """
        Perform one step of variatonal inference.

        Args:
            num_samples: number of samples per Monte-Carlo gradient estimate of E[f(logp - logq)]
            discrepancy_fn: discrepancy function f, use ``kl_reverse`` to minimize negative ELBO
            mc_approx: Monte-Carlo gradient estimator to use
            subsample_factor: subsampling factor used for subsampling, helps scale the observations to avoid overshrinking towards the prior

        Returns:
            torch.Tensor: the loss value (before the step)
        """
        self._optimizer.zero_grad()
        loss = mc_approx(
            self.observations,
            num_samples,
            discrepancy_fn,
            self.params,
            self.queries_to_guides,
            subsample_factor=subsample_factor,
            device=self._device,
        )
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            self._optimizer.step()
        else:
            logging.warn("Encountered NaN/inf loss, skipping step.")
        return loss

    def initialize_world(self) -> VariationalWorld:
        """
        Initializes a `VariationalWorld` using samples from guide distributions
        evaluated at the current parameter values.

        Returns:
            VariationalWorld: a `World` where guide samples and distributions
            have replaced their corresponding queries
        """
        return VariationalWorld.initialize_world(
            queries=self.queries_to_guides.values(),
            observations=self.observations,
            params=self.params,
            queries_to_guides=self.queries_to_guides,
            initialize_fn=lambda d: d.sample(),
        )
