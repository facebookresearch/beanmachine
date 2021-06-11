from types import TracebackType
from typing import Generator, NoReturn, Optional, Type

from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


class Sampler(Generator[SimpleWorld, Optional[SimpleWorld], None]):
    def __init__(
        self,
        proposer: BaseProposer,
        num_samples: Optional[int] = None,
        num_adaptive_samples: int = 0,
    ):
        self.proposer = proposer
        self.proposer.init_adaptation(num_adaptive_samples)
        self._num_samples_remaining = (
            float("inf") if num_samples is None else num_samples
        )
        self._num_samples_remaining += num_adaptive_samples
        self._num_adaptive_sample_remaining = num_adaptive_samples

    def send(self, world: Optional[SimpleWorld] = None) -> SimpleWorld:
        if self._num_samples_remaining > 0:
            world = self.proposer.propose(world)
            if self._num_adaptive_sample_remaining > 0:
                self.proposer.do_adaptation()
                self._num_adaptive_sample_remaining -= 1
                if self._num_samples_remaining == 0:
                    # we just reach the end of adaptation period
                    self.proposer.finish_adaptation()
            self._num_samples_remaining -= 1
            return world
        else:
            raise StopIteration

    def throw(
        self,
        typ: Type[BaseException],
        val: Optional[BaseException] = None,
        tb: Optional[TracebackType] = None,
    ) -> NoReturn:
        """Use the default error handling behavior (thorw Exception as-is)"""
        super().throw(typ, val, tb)
