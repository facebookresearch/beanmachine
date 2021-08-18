# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import random
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Callable, ClassVar, Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from beanmachine.ppl.vi.mean_field_variational_approximation import (
    MeanFieldVariationalApproximation,
)
from torch import Tensor
from torch.multiprocessing import Queue

from ..model.rv_identifier import RVIdentifier
from ..model.utils import LogLevel
from ..world import World
from .monte_carlo_samples import MonteCarloSamples


LOGGER = logging.getLogger("beanmachine")

# Detect and report if a user fails to meet the inference contract.
def _verify_queries(queries: List[RVIdentifier]) -> None:
    if not isinstance(queries, list):
        t = type(queries).__name__
        raise TypeError(
            f"Parameter 'queries' is required to be a list but is of type {t}."
        )

    for query in queries:
        if not isinstance(query, RVIdentifier):
            t = type(query).__name__
            raise TypeError(
                f"A query is required to be a random variable but is of type {t}."
            )
        for arg in query.arguments:
            if isinstance(arg, RVIdentifier):
                raise TypeError(
                    "The arguments to a query must not be random variables."
                )


def _verify_observations(
    observations: Dict[RVIdentifier, Tensor], must_be_rv: bool
) -> None:
    if not isinstance(observations, dict):
        t = type(observations).__name__
        raise TypeError(
            f"Parameter 'observations' is required to be a dictionary but is of type {t}."
        )

    for rv, value in observations.items():
        if not isinstance(rv, RVIdentifier):
            t = type(rv).__name__
            raise TypeError(
                f"An observation is required to be a random variable but is of type {t}."
            )
        if not isinstance(value, Tensor):
            t = type(value).__name__
            raise TypeError(
                f"An observed value is required to be a tensor but is of type {t}."
            )
        if must_be_rv and rv.is_functional:
            raise TypeError(
                "An observation must observe a random_variable, not a functional."
            )
        for arg in rv.arguments:
            if isinstance(arg, RVIdentifier):
                raise TypeError(
                    "The arguments to an observation must not be random variables."
                )


def _verify_queries_and_observations(
    queries: List[RVIdentifier],
    observations: Dict[RVIdentifier, Tensor],
    observations_must_be_rv: bool,
) -> None:
    _verify_queries(queries)
    _verify_observations(observations, observations_must_be_rv)


class VerboseLevel(Enum):
    """
    Enum class which is used to set how much output is printed during inference.
    LOAD_BAR enables tqdm for full inference loop.
    """

    OFF = 0
    LOAD_BAR = 1


class AbstractInference(object, metaclass=ABCMeta):
    """
    Abstract inference object that all inference algorithms inherit from.
    """

    world_: World
    _rand_int_max: ClassVar[int] = 2 ** 62

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
        vi_dicts: Optional[
            Callable[[RVIdentifier], MeanFieldVariationalApproximation]
        ] = None,
        model_to_guide_ids: Optional[Dict[RVIdentifier, RVIdentifier]] = None,
        params: Optional[Dict[RVIdentifier, nn.Parameter]] = None,
    ):
        """
        Initializes the world variables with queries and observation calls.

        :param initialize_from_prior: boolean to initialize samples from prior
        :param vi_dicts: mapping from `RVIdentifier`s to their corresponding mean-field
        approximation. Only used for `MeanFieldVariationalInference`.
        :param model_to_guide_ids: mapping between model `RVIdentifier`s and their
        corresponding guide `RVIdentifier`. Every `random_variable` in `self.queries_`
        must be present as a key. Only used for `VariationalInference`.
        :params: mapping from `bm.param`s to their current values. Only used
        for `VariationalInference`.
        """
        self.world_ = self.initial_world_.copy()
        self.world_.set_observations(self.observations_)
        self.world_.set_initialize_from_prior(initialize_from_prior)

        self.world_.vi_dicts = vi_dicts
        self.world_.model_to_guide_ids_ = model_to_guide_ids
        self.world_.params_ = params or {}
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
        self.world_.vi_dicts = None
        self.world_.model_to_guide_ids_ = None

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
        :params num_samples: number of samples excluding adaptation to collect.
        :params num_chains: number of chains to run
        :params num_adaptive_samples: number of steps to allow proposer adaptation.
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
            manager = mp.Manager()
            q = manager.Queue()
            for chain in range(num_chains):
                p = mp.Process(
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
