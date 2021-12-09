# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from beanmachine.ppl.model.rv_identifier import RVIdentifier

_WORLD_STACK = []


def get_world_context():
    return _WORLD_STACK[-1] if _WORLD_STACK else None


class BaseWorld(metaclass=ABCMeta):
    def __enter__(self):
        """
        This method, together with __exit__, allow us to use world as a context, e.g.
        ```
        with World():
            # invoke random variables to update the graph
        ```
        By keeping a stack of context tokens, we can easily nest multiple worlds and
        restore the outer context if needed, e.g.
        ```
        world1, world2 = World(), World()
        with world1:
            # do some graph update specific to world1
            with world2:
                # update world2
            # back to updating world1
        ```
        """
        _WORLD_STACK.append(self)
        return self

    def __exit__(self, *args):
        _WORLD_STACK.pop()

    def call(self, node: RVIdentifier):
        """
        A helper function that invokes the random variable and return its value
        """
        with self:
            return node.wrapper(*node.arguments)

    @abstractmethod
    def update_graph(self, node: RVIdentifier) -> torch.Tensor:
        raise NotImplementedError
