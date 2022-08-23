# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import platform
import random
from abc import ABCMeta, abstractmethod
from typing import ClassVar, Dict, List

import torch
import torch.multiprocessing as mp
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.utils import (
    _verify_queries_and_observations,
    VerboseLevel,
)
from beanmachine.ppl.legacy.world import World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.model.utils import LogLevel
from torch import Tensor
from torch.multiprocessing import Queue


LOGGER = logging.getLogger("beanmachine")


class AbstractInference(object, metaclass=ABCMeta):
    """
    Abstract inference object that all inference algorithms inherit from.
    """

    world_: World
    _rand_int_max: ClassVar[int] = 2**62

    def __init__(self):
        self.initial_world_ = World()
        self.world_ = self.initial_world_
        self.queries_ = []
        self.observations_ = {}

    @staticmethod
    def set_seed(seed: int):
        torch.manual_seed(seed)
        random.seed(seed)

    def initialize_world(
        self,
        initialize_from_prior: bool = False,
    ):
        """
        Initializes the world variables with queries and observation calls.

        :param initialize_from_prior: boolean to initialize samples from prior
        approximation.
        """
        self.world_ = self.initial_world_.copy()
        self.world_.set_observations(self.observations_)
        self.world_.set_initialize_from_prior(initialize_from_prior)

        for node in self.observations_:
            # makes the call for the observation node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world
            self.world_.call(node)
        for node in self.queries_:
            # makes the call for the query node, which will run sample(node())
            # that results in adding its corresponding Variable and its dependent
            # Variable to the world.
            self.world_.call(node)
        self.world_.accept_diff()

    def reset(self):
        """
        Resets world, mode and observation
        """
        self.world_ = self.initial_world_.copy()
        self.queries_ = []
        self.observations_ = {}


class AbstractMCInference(AbstractInference, metaclass=ABCMeta):
    """
    Abstract inference object for Monte Carlo inference.
    """

    _observations_must_be_rv: bool = True

    @staticmethod
    def set_seed_for_chain(random_seed: int, chain: int):
        AbstractInference.set_seed(random_seed + chain * 31)

    @abstractmethod
    def _infer(
        self,
        num_samples: int,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = False,
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractInference.
        """
        raise NotImplementedError("Inference algorithm must implement _infer.")

    def _parallel_infer(
        self,
        queue: Queue,
        chain: int,
        num_samples: int,
        random_seed: int,
        num_adaptive_samples: int,
        verbose: VerboseLevel,
    ):
        try:
            AbstractMCInference.set_seed_for_chain(random_seed, chain)
            rv_dict = self._infer(num_samples, num_adaptive_samples, verbose)
            string_dict = {str(rv): tensor.detach() for rv, tensor in rv_dict.items()}
            queue.put((None, chain, string_dict))
        except BaseException as x:
            LOGGER.log(
                LogLevel.ERROR.value, "Error: Parallel infererence chain failed."
            )
            queue.put((x, chain, {}))

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_samples: int,
        num_chains: int = 4,
        run_in_parallel: bool = False,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = False,
    ) -> MonteCarloSamples:
        """
        Run inference algorithms and reset the world/mode at the end.

        All tensors in `queries` and `observations` must be allocated on the
        same `torch.device`. Inference algorithms will attempt to allocate
        intermediate tensors on the same device.

        :param queries: random variables to query
        :param observations: observed random variables with their values
        :param num_samples: number of samples excluding adaptation to collect.
        :param num_chains: number of chains to run
        :param num_adaptive_samples: number of steps to allow proposer adaptation.
        :param verbose: Integer indicating how much output to print to stdio
        :param initialize_from_prior: boolean to initialize samples from prior
        :returns: view of data for chains and samples for query
        """

        _verify_queries_and_observations(
            queries, observations, self._observations_must_be_rv
        )
        random_seed = torch.randint(AbstractInference._rand_int_max, (1,)).int().item()
        self.queries_ = queries
        self.observations_ = observations
        if num_chains > 1 and run_in_parallel:
            if platform.system() == "Windows":
                raise RuntimeError(
                    "Running inference in parallel is not currently support on Windows"
                )

            ctx = mp.get_context("fork")
            manager = ctx.Manager()
            q = manager.Queue()
            for chain in range(num_chains):
                p = ctx.Process(
                    target=self._parallel_infer,
                    args=(
                        q,
                        chain,
                        num_samples,
                        random_seed,
                        num_adaptive_samples,
                        verbose,
                    ),
                )
                p.start()

            chain_queries = [{}] * num_chains
            for _ in range(num_chains):
                (error, chain, string_dict) = q.get()
                if error is not None:
                    raise error
                rv_dict = {rv: string_dict[str(rv)] for rv in queries}
                chain_queries[chain] = rv_dict
        else:
            chain_queries = []
            for chain in range(num_chains):
                # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool,
                #  float, int]`.
                AbstractMCInference.set_seed_for_chain(random_seed, chain)
                rv_dicts = self._infer(
                    num_samples,
                    num_adaptive_samples,
                    verbose,
                    initialize_from_prior,
                )
                chain_queries.append(rv_dicts)
        monte_carlo_samples = MonteCarloSamples(chain_queries, num_adaptive_samples)
        self.reset()
        return monte_carlo_samples
