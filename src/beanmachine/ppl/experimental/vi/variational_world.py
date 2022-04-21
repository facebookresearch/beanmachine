# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

import torch
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import World
from beanmachine.ppl.world.world import RVDict


class VariationalWorld(World):
    "A World which also contains (variational) parameters."

    def __init__(self, params: Dict, *args, **kwargs) -> None:
        self._params = params
        super().__init__(*args, **kwargs)

    def copy(self):
        world_copy = VariationalWorld(
            observations=self.observations.copy(),
            initialize_fn=self._initialize_fn,
            params=self._params.copy(),
        )
        world_copy._variables = self._variables.copy()
        return world_copy

    # TODO: distinguish params vs random_variables at the type-level
    def get_param(self, param: RVIdentifier) -> torch.Tensor:
        "Gets a parameter or initializes it if not found."
        if param not in self._params:
            self._params[param] = param.function(*param.arguments)
            self._params[param].requires_grad = True
        return self._params[param]

    def set_params(self, params: RVDict):
        "Sets the parameters in this World to specified values."
        self._params = params
