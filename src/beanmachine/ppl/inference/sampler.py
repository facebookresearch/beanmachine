# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union, cast

import torch
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.utils import merge_dicts
from beanmachine.ppl.model.rv_identifier import RVIdentifier


if TYPE_CHECKING:
    from beanmachine.ppl.inference.abstract_mh_infer import AbstractMHInference

RVDict = Dict[RVIdentifier, torch.Tensor]


class Sampler(Iterator[RVDict]):
    def __init__(
        self,
        kernel: AbstractMHInference,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: Optional[int] = None,
        num_adaptive_samples: int = 0,
        initialize_from_prior: bool = False,
        return_adaptive_samples: bool = False,
    ):
        if num_samples is None:
            self._iterations = itertools.count(0)
        else:
            self._iterations = iter(range(num_samples + num_adaptive_samples))
        self.num_adaptive_samples = num_adaptive_samples
        self.iteration = None

        # initialize kernel
        self.kernel = copy.copy(kernel)
        self.kernel.queries_ = queries
        self.kernel.observations_ = observations
        self.kernel.initialize_world(initialize_from_prior)
        self.kernel.world_.set_initialize_from_prior(True)

        if not return_adaptive_samples:
            for _ in range(self.num_adaptive_samples):
                next(self)

    def __next__(self) -> RVDict:
        """
        Run a single MCMC iteration and return a dict containing the queried samples
        """
        # this will propagate StopIteration from self._iterations
        self.iteration = next(self._iterations)
        return self.kernel._single_iteration_run(
            self.iteration, self.num_adaptive_samples
        )

    @staticmethod
    def to_monte_carlo_samples(
        sample_list: Union[List[RVDict], List[List[RVDict]]],
        num_adaptive_samples: int = 0,
    ) -> MonteCarloSamples:
        """
        A helper function that convert a list of dicts of random variables generated
        from a Sampler to a single MonteCarloSamples instance. Samples from multiple
        chains can also be merged using this method by providing a list of lists, each
        correspond to the result from a single chain.
        """
        if all(isinstance(sample, dict) for sample in sample_list):
            # the cast is necessary to silent Pyre though it does nothing in run time
            sample_list = [cast(List[RVDict], sample_list)]
        chain_list = [
            merge_dicts(chain) for chain in cast(List[List[RVDict]], sample_list)
        ]
        return MonteCarloSamples(chain_list, num_adaptive_samples)
