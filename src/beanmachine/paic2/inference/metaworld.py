from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable

import torch
from beanmachine.ppl import RVIdentifier
from beanmachine.ppl.world import World


class MetaWorld(metaclass=ABCMeta):
    @abstractmethod
    def print(self):
        raise NotImplementedError()


class RealWorld(MetaWorld):
    def __init__(
        self,
        queries: Iterable[RVIdentifier],
        observations: Dict[RVIdentifier, torch.Tensor],
    ):
        self.python_world = World.initialize_world(queries, observations)

    def print(self):
        print(str(self.python_world))


MetaWorld.register(RealWorld)
