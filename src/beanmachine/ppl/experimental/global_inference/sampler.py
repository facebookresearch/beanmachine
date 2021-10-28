from __future__ import annotations

import random
import warnings
from types import TracebackType
from typing import (
    Generator,
    NoReturn,
    Optional,
    Type,
    TYPE_CHECKING,
)


if TYPE_CHECKING:
    from beanmachine.ppl.experimental.global_inference.base_inference import (
        BaseInference,
    )

from beanmachine.ppl.experimental.global_inference.simple_world import (
    SimpleWorld,
)


class Sampler(Generator[SimpleWorld, Optional[SimpleWorld], None]):
    def __init__(
        self,
        kernel: BaseInference,
        initial_world: SimpleWorld,
        num_samples: Optional[int] = None,
        num_adaptive_samples: int = 0,
    ):
        self.kernel = kernel
        self.world = initial_world
        self._num_samples_remaining = (
            float("inf") if num_samples is None else num_samples
        )
        self._num_samples_remaining += num_adaptive_samples
        self._num_adaptive_sample_remaining = num_adaptive_samples

    def send(self, world: Optional[SimpleWorld] = None) -> SimpleWorld:
        if world is None:
            world = self.world

        if self._num_samples_remaining <= 0:
            raise StopIteration

        proposers = self.kernel.get_proposers(
            world, world.latent_nodes, self._num_adaptive_sample_remaining
        )
        random.shuffle(proposers)

        for proposer in proposers:
            try:
                world = proposer.propose(world)
            except RuntimeError as e:
                if "singular U" in str(e) or "input is not positive-definite" in str(e):
                    # since it's normal to run into cholesky error during GP, instead of
                    # throwing an error, we simply skip current proposer (which is
                    # equivalent to a rejection) and will retry in the next iteration
                    warnings.warn(f"Proposal rejected: {e}", RuntimeWarning)
                    continue
                else:
                    raise e

            if self._num_adaptive_sample_remaining > 0:
                proposer.do_adaptation()
                if self._num_samples_remaining == 1:
                    # we just reach the end of adaptation period
                    proposer.finish_adaptation()

        # update attributes at last, so that exceptions during inference won't leave
        # self in an invalid state
        self.world = world
        if self._num_adaptive_sample_remaining > 0:
            self._num_adaptive_sample_remaining -= 1
        self._num_samples_remaining -= 1
        return self.world

    def throw(
        self,
        typ: Type[BaseException],
        val: Optional[BaseException] = None,
        tb: Optional[TracebackType] = None,
    ) -> NoReturn:
        """Use the default error handling behavior (thorw Exception as-is)"""
        super().throw(typ, val, tb)
