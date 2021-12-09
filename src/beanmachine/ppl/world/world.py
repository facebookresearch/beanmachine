# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Mapping, Optional, Set, Tuple, Collection

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.base_world import BaseWorld
from beanmachine.ppl.world.initialize_fn import (
    InitializeFn,
    init_from_prior,
)
from beanmachine.ppl.world.variable import Variable


RVDict = Dict[RVIdentifier, torch.Tensor]


@dataclasses.dataclass
class _TempVar:
    node: RVIdentifier
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)


class World(BaseWorld, Mapping[RVIdentifier, torch.Tensor]):
    """
    A World represents an instantiation of the graphical model and can be manipulated or evaluated.
    In the contest of MCMC inference, a world represents a single Monte Carlo posterior sample.

    A World can also be used as a context manager to run and sample random variables.
    Example::

        @bm.random_variable
        def foo():
          return Normal(0., 1.)

        world = World()
        with world:
          x = foo()  # returns a sample, ie tensor.

        with world:
          y = foo()  # same world = same tensor

        assert x == y

    Args:
      observations (Optional): Optional observations, which fixes the random variables to observed values
      initialize_fn (callable, Optional): Callable which takes a ``torch.distribution`` object as argument and returns a ``torch.Tensor``
    """

    def __init__(
        self,
        observations: Optional[RVDict] = None,
        initialize_fn: InitializeFn = init_from_prior,
    ):
        self.observations: RVDict = observations or {}
        self._initialize_fn: InitializeFn = initialize_fn
        self._variables: Dict[RVIdentifier, Variable] = {}

        self._call_stack: List[_TempVar] = []

    def __getitem__(self, node: RVIdentifier) -> torch.Tensor:
        """
        Args:
          node (RVIdentifier): RVIdentifier of node.

        Returns:
          torch.Tensor: The sampled value.
        """
        return self._variables[node].value

    def get_variable(self, node: RVIdentifier) -> Variable:
        """
        Args:
          node (RVIdentifier): RVIdentifier of node.

        Returns:
          Variable object that contains the metadata of the current node
          in the world.
        """
        return self._variables[node]

    def replace(self, values: RVDict) -> World:
        """
        Args:
          values (RVDict): Dict of RVIdentifiers and their values to replace.

        Returns:
          A new world where values specified in the dictionary are replaced.
          This method will update the internal graph structure.
        """
        assert not any(node in self.observations for node in values)
        new_world = self.copy()
        for node, value in values.items():
            new_world._variables[node] = new_world._variables[node].replace(
                value=value.clone()
            )
        # changing the value of a node can change the dependencies of its children nodes
        nodes_to_update = set().union(
            *(self._variables[node].children for node in values)
        )
        for node in nodes_to_update:
            # Invoke node conditioned on the provided values
            new_distribution, new_parents = new_world._run_node(node)
            # Update children's dependencies
            old_node_var = new_world._variables[node]
            new_world._variables[node] = old_node_var.replace(
                parents=new_parents, distribution=new_distribution
            )
            dropped_parents = old_node_var.parents - new_parents
            for parent in dropped_parents:
                parent_var = new_world._variables[parent]
                new_world._variables[parent] = parent_var.replace(
                    children=parent_var.children - {node}
                )
        return new_world

    def __iter__(self) -> Iterator[RVIdentifier]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    @property
    def latent_nodes(self) -> Set[RVIdentifier]:
        """
        All the latent nodes in the current world.
        """
        return self._variables.keys() - self.observations.keys()

    def copy(self) -> World:
        """
        Returns:
          Shallow copy of the current world.
        """
        world_copy = World(self.observations.copy(), self._initialize_fn)
        world_copy._variables = self._variables.copy()
        return world_copy

    def initialize_value(self, node: RVIdentifier) -> None:
        # recursively calls into parent nodes
        distribution, parents = self._run_node(node)

        if node in self.observations:
            node_val = self.observations[node]
        else:
            node_val = self._initialize_fn(distribution)

        self._variables[node] = Variable(
            value=node_val,
            distribution=distribution,
            parents=parents,
        )

    def update_graph(self, node: RVIdentifier) -> torch.Tensor:
        """
        This function adds a node to the graph and initialize its value if the node
        is not found in the graph already.

        Args:
          node (RVIdentifier): RVIdentifier of node to update in the graph.

        Returns:
          The value of the node stored in world (in original space).
        """
        if node not in self._variables:
            self.initialize_value(node)
        node_var = self._variables[node]
        if len(self._call_stack) > 0:
            tmp_child_var = self._call_stack[-1]
            tmp_child_var.parents.add(node)
            node_var.children.add(tmp_child_var.node)

        return node_var.value

    def log_prob(
        self, nodes: Optional[Collection[RVIdentifier]] = None
    ) -> torch.Tensor:
        """
        Args:
          nodes (Optional): Optional collection of RVIdentifiers to evaluate the log prob of a subset of
                 the graph. If none is specified, then all the variables in the world are used.
        Returns:
          The joint log prob of all of the nodes in the current world
        """
        if nodes is None:
            nodes = self._variables.keys()

        log_prob = torch.tensor(0.0)
        for node in set(nodes):
            log_prob += torch.sum(self._variables[node].log_prob)
        return log_prob

    def enumerate_node(self, node: RVIdentifier) -> torch.Tensor:
        """
        Args:
          node (RVIdentifier): RVIdentifier of node.

        Returns:
          A tensor enumerating the support of the node.
        """
        distribution = self._variables[node].distribution
        # pyre-ignore[16]
        if not distribution.has_enumerate_support:
            raise ValueError(str(node) + " is not enumerable")
        # pyre-ignore[16]
        return distribution.enumerate_support()

    def _run_node(
        self, node: RVIdentifier
    ) -> Tuple[dist.Distribution, Set[RVIdentifier]]:
        """
        Invoke a random variable function conditioned on the current world.

        Args:
          node (RVIdentifier): RVIdentifier of node.

        Returns:
          Its distribution and set of parent nodes
        """
        self._call_stack.append(_TempVar(node))
        with self:
            distribution = node.function(*node.arguments)
        temp_var = self._call_stack.pop()
        if not isinstance(distribution, dist.Distribution):
            raise TypeError("A random_variable is required to return a distribution.")
        return distribution, temp_var.parents
