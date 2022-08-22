# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import init_from_prior, World
from beanmachine.ppl.world.initialize_fn import InitializeFn
from beanmachine.ppl.world.world import RVDict


class VariationalWorld(World):
    """A World which also contains (variational) parameters."""

    def __init__(
        self,
        observations: Optional[RVDict] = None,
        initialize_fn: InitializeFn = init_from_prior,
        params: Optional[MutableMapping[RVIdentifier, torch.Tensor]] = None,
        queries_to_guides: Optional[Mapping[RVIdentifier, RVIdentifier]] = None,
    ) -> None:
        self._params = params or {}
        self._queries_to_guides = queries_to_guides or {}
        super().__init__(observations, initialize_fn)

    def copy(self):
        world_copy = VariationalWorld(
            observations=self.observations.copy(),
            initialize_fn=self._initialize_fn,
            params=self._params.copy(),
            queries_to_guides=self._queries_to_guides.copy(),
        )
        world_copy._variables = self._variables.copy()
        return world_copy

    # TODO: distinguish params vs random_variables at the type-level
    def get_param(self, param: RVIdentifier) -> torch.Tensor:
        """Gets a parameter or initializes it if not found."""
        if param not in self._params:
            init_value = param.function(*param.arguments)
            assert isinstance(init_value, torch.Tensor)
            self._params[param] = init_value
            self._params[param].requires_grad = True
        return self._params[param]

    def set_params(self, params: MutableMapping[RVIdentifier, torch.Tensor]):
        """Sets the parameters in this World to specified values."""
        self._params = params

    def get_guide_distribution(self, rv: RVIdentifier) -> dist.Distribution:
        guide_rv = self._queries_to_guides[rv]
        return self.get_variable(guide_rv).distribution

    def update_graph(self, node: RVIdentifier) -> torch.Tensor:
        """
        Initialize a new node using its guide if available and
        the prior otherwise.

        Args:
          node (RVIdentifier): RVIdentifier of node to update in the graph.

        Returns:
          The value of the node stored in world (in original space).
        """
        if node in self._queries_to_guides:
            node = self._queries_to_guides[node]
        return super().update_graph(node)
