# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.optim as optim
from beanmachine.ppl.experimental.vi.discrepanancy import kl_reverse
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
            num_steps: Number of steps of stochastic variational inference to perform.
            optimizer: A function returning a ``torch.Optimizer`` to use for optimizing variational parameters.
            device: a ``torch.device`` to use for pytorch tensors
        """
        super().__init__()

        # runs all guides to reify `param`s for `optimizer`
        # NOTE: assumes `params` is static and same across all worlds, consider MultiOptimizer (see Pyro)
        self.params = {}
        self.observations = observations
        self.queries_to_guides = queries_to_guides

        self._world = VariationalWorld(
            observations=observations,
            params=self.params,
        )

        # TODO: what happens if not all the params are encountered
        # in this execution pass, eg an if/else, consider MultiOptimizer
        for guide in queries_to_guides.values():
            self._world.call(guide)
        self._optimizer = optimizer(self.params.values())

    def infer(
        self,
        num_steps: int,
        num_samples: int = 1,
        discrepancy_fn=kl_reverse,
        mc_approx=monte_carlo_approximate_reparam,  # TODO: support both reparam and SF in same guide
        on_step: Optional[Callable[[torch.Tensor, VariationalInfer], None]] = None,
    ) -> VariationalWorld:
        for _ in tqdm(range(num_steps)):
            loss, self = self.step(num_samples, discrepancy_fn, mc_approx)
            if on_step:
                on_step(loss, self)

        # NOTE: we skip reinitializing guide `Variable`s in the `World` within
        # the main optimization loop, but for `Variable.distribution` to use the
        # latest `params` we need to recompute them before returning
        for guide in self.queries_to_guides.values():
            self._world.initialize_value(guide)

        return self._world

    def step(
        self,
        num_samples: int = 1,
        discrepancy_fn=kl_reverse,
        mc_approx=monte_carlo_approximate_reparam,  # TODO: support both reparam and SF in same guide
    ) -> Tuple[torch.Tensor, VariationalInfer]:
        self._optimizer.zero_grad()
        loss = mc_approx(
            self.observations,
            num_samples,
            discrepancy_fn,
            self.queries_to_guides,
            self.params,
        )
        if not torch.isnan(loss) and not torch.isinf(loss):
            loss.backward()
            self._optimizer.step()
        return loss, self
