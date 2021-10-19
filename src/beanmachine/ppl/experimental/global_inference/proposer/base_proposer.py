from abc import ABCMeta, abstractmethod

from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


class BaseProposer(metaclass=ABCMeta):
    @abstractmethod
    def propose(self, world: SimpleWorld) -> SimpleWorld:
        raise NotImplementedError

    def do_adaptation(self) -> None:
        ...

    def finish_adaptation(self) -> None:
        ...
