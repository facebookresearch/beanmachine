from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Mapping, Optional, Set

import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.utils.initialize_fn import (
    InitializeFn,
    init_from_prior,
)
from beanmachine.ppl.experimental.global_inference.variable import Variable
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.base_world import BaseWorld
from beanmachine.ppl.world.utils import get_default_transforms


RVDict = Dict[RVIdentifier, torch.Tensor]


@dataclasses.dataclass
class _TempVar:
    node: RVIdentifier
    parents: Set[RVIdentifier] = dataclasses.field(default_factory=set)


class SimpleWorld(BaseWorld, Mapping[RVIdentifier, torch.Tensor]):
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
        node_var = self._variables[node]
        return node_var.transform.inv(node_var.transformed_value)

    def get_transformed(self, node: RVIdentifier) -> torch.Tensor:
        """Return the value of the node in the unconstrained space"""
        return self._variables[node].transformed_value

    def set_transformed(self, node: RVIdentifier, value: torch.Tensor) -> None:
        """Set the value of the node in the unconstrained space"""
        self._variables[node].transformed_value = value

    def __iter__(self) -> Iterator[RVIdentifier]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    @property
    def latent_nodes(self) -> Set[RVIdentifier]:
        """Return a KeysView of all latent nodes in the current world"""
        return self._variables.keys() - self.observations.keys()

    def copy(self) -> SimpleWorld:
        """Returns a shallow copy of the current world"""
        world_copy = SimpleWorld(self.observations.copy(), self._initialize_fn)
        world_copy._variables = {
            node: var.copy() for node, var in self._variables.items()
        }
        return world_copy

    def initialize_value(self, node: RVIdentifier) -> None:
        # recursively calls into parent nodes
        self._call_stack.append(_TempVar(node))
        with self:
            distribution = node.function(*node.arguments)
        temp_var = self._call_stack.pop()

        if node in self.observations:
            transformed_value = self.observations[node]
            transform = dist.identity_transform
        else:
            transform = get_default_transforms(distribution)
            transformed_value = transform(self._initialize_fn(distribution))

        self._variables[node] = Variable(
            transformed_value=transformed_value,
            transform=transform,
            parents=temp_var.parents,
        )

    def update_graph(self, node: RVIdentifier) -> torch.Tensor:
        """This function adds a node to the graph and initialize its value if the node
        is not found in the graph already. It then returns the value of the node stored
        in world (in original space)."""
        if node not in self._variables:
            self.initialize_value(node)
        node_var = self._variables[node]
        if len(self._call_stack) > 0:
            tmp_child_var = self._call_stack[-1]
            tmp_child_var.parents.add(node)
            node_var.children.add(tmp_child_var.node)

        return node_var.transform.inv(node_var.transformed_value)

    def log_prob(self) -> torch.Tensor:
        """Returns the joint log prob of all of the nodes in the current world"""
        log_prob = torch.tensor(0.0)
        with self:
            for node, node_var in self._variables.items():
                distribution = node.function(*node.arguments)
                y = node_var.transformed_value
                x = node_var.transform.inv(y)
                log_prob += torch.sum(
                    distribution.log_prob(x)
                    - node_var.transform.log_abs_det_jacobian(x, y)
                )
        return log_prob

    def enumerate_node(self, node: RVIdentifier) -> torch.Tensor:
        """Returns a tensor enumerating the support of the node"""
        with self:
            distribution = node.function(*node.arguments)
            if not distribution.has_enumerate_support:
                raise ValueError(str(node) + " is not enumerable")
            return distribution.enumerate_support()
