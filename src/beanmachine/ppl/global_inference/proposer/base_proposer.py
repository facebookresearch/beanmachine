from abc import ABCMeta, abstractmethod
from typing import Optional

from beanmachine.ppl.global_inference.simple_world import SimpleWorld


class BaseProposer(metaclass=ABCMeta):
    @abstractmethod
    def propose(self, world: Optional[SimpleWorld] = None) -> SimpleWorld:
        raise NotImplementedError

    def init_adaptation(self, num_adaptive_samples: int) -> None:
        ...

    def do_adaptation(self) -> None:
        ...

    def finish_adaptation(self) -> None:
        ...
