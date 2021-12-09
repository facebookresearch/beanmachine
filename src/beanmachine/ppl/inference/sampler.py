# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

import torch


if TYPE_CHECKING:
    from beanmachine.ppl.inference.base_inference import (
        BaseInference,
    )

from beanmachine.ppl.world import World


class Sampler(Generator[World, Optional[World], None]):
    """
    Samplers are generators of Worlds that generate samples from the joint.
    It is used to generate Monte Carlo samples during MCMC inference.
    At each iteration, the proposer(s) proposer a values for the random variables, which
    are then accepted according to the MH ratio. The next world is then returned.

    Args:
        kernel (BaseInference): Inference class to get proposers from.
        initial_world (World): Optional initial world to initialize from.
        num_samples (int, Optional): Number of samples. If none is specified, num_samples = inf.
        num_adaptive_samples (int, Optional): Number of adaptive samples, defaults to 0.
    """

    def __init__(
        self,
        kernel: BaseInference,
        initial_world: World,
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

    def send(self, world: Optional[World] = None) -> World:
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
                new_world, accept_log_prob = proposer.propose(world)
                accepted = torch.rand_like(accept_log_prob).log() < accept_log_prob
                if accepted:
                    world = new_world
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
                proposer.do_adaptation(
                    world=world, accept_log_prob=accept_log_prob, is_accepted=accepted
                )
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
