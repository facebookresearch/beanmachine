from __future__ import annotations

from typing import Dict, Iterator, MutableMapping, Optional, Set

import torch
import torch.distributions as dist
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.base_world import BaseWorld
from beanmachine.ppl.world.utils import get_default_transforms


RVDict = Dict[RVIdentifier, torch.Tensor]


class SimpleWorld(BaseWorld, MutableMapping[RVIdentifier, torch.Tensor]):
    def __init__(
        self,
        observations: Optional[RVDict] = None,
        initialize_from_prior: bool = False,
        transforms: Optional[Dict[RVIdentifier, dist.Transform]] = None,
    ):
        self.observations: RVDict = observations or {}
        self.initialize_from_prior: bool = initialize_from_prior
        self._transformed_values: RVDict = {}
        self.transforms = transforms or {}

    def __getitem__(self, node: RVIdentifier) -> torch.Tensor:
        return self.transforms[node].inv(self._transformed_values[node])

    def __setitem__(self, node: RVIdentifier, value: torch.Tensor) -> None:
        assert node in self.transforms
        if node not in self.observations:
            self._transformed_values[node] = self.transforms[node](value)

    def get_transformed(self, node: RVIdentifier) -> torch.Tensor:
        """Return the value of the node in the unconstrained space"""
        return self._transformed_values[node]

    def set_transformed(self, node: RVIdentifier, value: torch.Tensor) -> None:
        """Set the value of the node in the unconstrained space"""
        assert node in self.transforms
        self._transformed_values[node] = value

    def __delitem__(self, node: RVIdentifier) -> None:  # pyre-ignore[14]
        del self._transformed_values[node]
        del self.transforms[node]

    def __iter__(self) -> Iterator[RVIdentifier]:
        return iter(self._transformed_values)

    def __len__(self) -> int:
        return len(self._transformed_values)

    @property
    def latent_nodes(self) -> Set[RVIdentifier]:
        """Return a KeysView of all latent nodes in the current world"""
        return self._transformed_values.keys() - self.observations.keys()

    def copy(self) -> SimpleWorld:
        """Returns a shallow copy of the current world"""
        world_copy = SimpleWorld(
            self.observations.copy(), self.initialize_from_prior, self.transforms.copy()
        )
        world_copy._transformed_values = self._transformed_values.copy()
        return world_copy

    def initialize_value(self, node: RVIdentifier) -> None:
        # calling node.function will initialize parent nodes recursively
        distribution = node.function(*node.arguments)
        if node in self.observations:
            value = self.observations[node]
            transform = dist.identity_transform
        else:
            transform = self.transforms.get(node, get_default_transforms(distribution))
            sample_val = distribution.sample()
            if self.initialize_from_prior or distribution.has_enumerate_support:
                value = transform(sample_val)
            else:
                # initialize to Uniform(-2, 2) in unconstrained space
                value = torch.rand_like(sample_val) * 4 - 2
        self._transformed_values[node] = value
        self.transforms[node] = transform

    def update_graph(self, node: RVIdentifier) -> torch.Tensor:
        """This function adds a node to the graph and initialize its value if the node
        is not found in the graph already. It then returns the value of the node stored
        in world (in original space)."""
        if node not in self._transformed_values:
            self.initialize_value(node)
        return self.transforms[node].inv(self._transformed_values[node])

    def log_prob(self) -> torch.Tensor:
        """Returns the joint log prob of all of the nodes in the current world"""
        log_prob = torch.tensor(0.0)
        with self:
            for node, y in self._transformed_values.items():
                distribution = node.function(*node.arguments)
                transform = self.transforms[node]
                x = transform.inv(y)
                log_prob += torch.sum(
                    distribution.log_prob(x) - transform.log_abs_det_jacobian(x, y)
                )
        return log_prob

    def enumerate_node(self, node: RVIdentifier) -> torch.Tensor:
        """Returns a tensor enumerating the support of the node"""
        with self:
            dist = node.function(*node.arguments)
            if not dist.has_enumerate_support:
                raise ValueError(str(node) + " is not enumerable")
            return dist.enumerate_support()
