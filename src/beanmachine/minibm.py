# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
The file contains MiniBM, a minimal implementation of Bean Machine PPL with a Metropolis
Hastings implementation and a coin flipping model at the end. It is standalone, in that
MiniBM does not depend on the Bean Machine framework at all. The only two dependencies
for MiniBM are the PyTorch library and tqdm (for progress bar).
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributions as dist
from tqdm.auto import tqdm


class RVIdentifier(NamedTuple):
    """
    A struct whose attributes uniquely identifies a random variable in Bean Machine.

    Args:
        wrapper: A reference to the decorated random variable function
        args: Arguments taken by the
    """

    wrapper: Callable
    args: Tuple

    @property
    def function(self):
        """A pointer to the original function that returns the distribution object"""
        return self.wrapper.__wrapped__  # calls the original function


def random_variable(f: Callable[Any, dist.Distribution]):
    """A decorator that convert a Python function that returns a distribution into a
    function that evaluates to a Bean Machine random variable.

    In Bean Machine, a @random_variable function can be used in two ways:

    1. When being invoked outside of an inference scope, it returns an RVIdentifier
       without evaluating the original function.
    2. During inference, or when being invoked from another random variable, the
       function will update the graph (if needed) and return its value at state of
       inference.

    For example::

        @random_variable
        def foo():
            return dist.Normal(0., 1.0)

        print(foo())  # RVIdentifier(wrapper=foo, args=())

        @random_variable
        def bar():
            mean = foo()  # evaluates to value of theta() during inference
            return dist.Normal(mean, 1.0)


    Args:
        f: A function that returns a PyTorch Distribution object
    """

    @wraps(f)
    def wrapper(*args):
        rvid = RVIdentifier(wrapper, args)
        # Bean Machine inference methods use the World class to store and control the
        # state of inference
        world = get_world_context()
        if world is None:
            # We're not in an active inference. Return an ID for the random variable
            return rvid
        else:
            # Update the graph and return current value of the random variable in world
            return world.update_graph(rvid)

    return wrapper


RVDict = Dict[RVIdentifier, torch.Tensor]  # alias for typing
WORLD_STACK: List[World] = []


def get_world_context() -> Optional[World]:
    """Returns the active World (if any) or None"""
    return WORLD_STACK[-1] if WORLD_STACK else None


class World:
    """
    A World is Bean Machine's internal representation of a state of the model. At the
    high level, it stores a value for each of the random variables. It can also be used
    as a context manager to control the behavior of random variables. For example::

        @random_variable
        def foo():
            return dist.Normal(0., 1.0)

        @random_variable
        def bar():
            return dist.Normal(foo(), 1.0)

        # initialize world and add bar() and its ancesters to it
        world = World.initialize_world([bar()])
        world[bar()]  # returns the value of bar() in world
        world[foo()]  # since foo() is bar()'s parent, it is also initialized in world

        # World is also used within inference as a context manager to control the
        # behavior of random variable
        with world:
            foo()  # returns the value of foo() in world, which equals to world[foo()]
    """

    def __init__(self, observations: Optional[RVDict] = None):
        self.observations: RVDict = observations or {}
        self.variables: RVDict = {}

    def __getitem__(self, node: RVIdentifier) -> torch.Tensor:
        return self.variables[node]

    def __enter__(self) -> World:
        WORLD_STACK.append(self)
        return self

    def __exit__(self, *args) -> None:
        WORLD_STACK.pop()

    def update_graph(self, node: RVIdentifier) -> torch.Tensor:
        """Update the graphy by adding node to self (if needed) and retuurn the value
        of node in self."""
        if node not in self.variables:
            # parent nodes will be invoked when calling node.get_distribution
            distribution = self.get_distribution(node)
            if node in self.observations:
                self.variables[node] = self.observations[node]
            else:
                self.variables[node] = distribution.sample()
        return self.variables[node]

    def replace(self, values: RVDict) -> World:
        """Return a new world where the values of the random variables are replaced by
        the provided values"""
        new_world = World(self.observations)
        new_world.variables = {**self.variables, **values}
        return new_world

    def log_prob(self) -> torch.Tensor:
        """Return the joint log prob on all random variables in the world"""
        log_prob = torch.tensor(0.0)
        for node, value in self.variables.items():
            distribution = self.get_distribution(node)
            log_prob += distribution.log_prob(value).sum()
        return log_prob

    def get_distribution(self, node: RVIdentifier) -> dist.Distribution:
        """A utility method that activate the current world and invoke the function
        associated with node. Bean Machine requires random variable functions to return
        a distribution object, so this method will also return a distribution object."""
        with self:
            return node.function(*node.args)

    @staticmethod
    def initialize_world(
        queries: List[RVIdentifier], observations: Optional[RVDict] = None
    ) -> World:
        """Initializes and returns a new world. Starting from the queries and
        observations, the parent nodes will be added recursively to the world."""
        observations = observations or {}
        world = World(observations)
        for node in itertools.chain(queries, observations):
            world.update_graph(node)
        return world


class MetropolisHastings:
    """A naive implementation of the `Metropolis-Hastings algorithm
    <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`_"""

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Optional[RVDict],
        num_samples: int,
    ) -> RVDict:
        world = World.initialize_world(queries, observations)

        samples = defaultdict(list)
        # the main inference loop
        for _ in tqdm(range(num_samples)):
            latent_nodes = world.variables.keys() - world.observations.keys()
            random.shuffle(latent_nodes)
            # randomly select a node to be updated at a time
            for node in latent_nodes:
                proposer_distribution = world.get_distribution(node)
                new_value = proposer_distribution.sample()
                new_world = world.replace({node: new_value})
                backward_distribution = new_world.get_distribution(node)

                # log P(x, y)
                old_log_prob = world.log_prob()
                # log P(x', y)
                new_log_prob = new_world.log_prob()
                # log g(x'|x)
                forward_log_prob = proposer_distribution.log_prob(new_value).sum()
                # log g(x|x')
                backward_log_prob = backward_distribution.log_prob(world[node]).sum()

                accept_log_prob = (
                    new_log_prob + backward_log_prob - old_log_prob - forward_log_prob
                )
                if torch.bernoulli(accept_log_prob.exp().clamp(max=1)):
                    # accept the new state
                    world = new_world

            # collect the samples before moving to the next iteration
            for node in queries:
                samples[node].append(world[node])
        # stack the list of tensors into a single tensor
        samples = {node: torch.stack(samples[node]) for node in samples}
        return samples


def main():
    # coin fliping model adapted from our tutorial
    # (https://beanmachine.org/docs/overview/tutorials/Coin_flipping/CoinFlipping/)
    @random_variable
    def weight():
        return dist.Beta(2, 2)

    @random_variable
    def y():
        return dist.Bernoulli(weight()).expand((N,))

    # data generation
    true_weight = 0.75
    true_y = dist.Bernoulli(true_weight)
    N = 100
    y_obs = true_y.sample((N,))

    print("Head rate:", y_obs.mean())

    # running inference
    samples = MetropolisHastings().infer([weight()], {y(): y_obs}, num_samples=500)
    print("Estimated weight of the coin:", samples[weight()].mean())


if __name__ == "__main__":
    main()
