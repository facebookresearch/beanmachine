from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
from beanmachine.ppl.world import World


class BaseProposer(metaclass=ABCMeta):
    @abstractmethod
    def propose(self, world: World) -> Tuple[World, torch.Tensor]:
        raise NotImplementedError

    def do_adaptation(self, world, accept_log_prob, *args, **kwargs) -> None:
        ...

    def finish_adaptation(self) -> None:
        ...
