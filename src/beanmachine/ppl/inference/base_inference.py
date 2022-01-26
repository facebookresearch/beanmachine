# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Set

import torch
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.inference.sampler import Sampler
from beanmachine.ppl.inference.utils import (
    VerboseLevel,
    _verify_queries_and_observations,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict, World, InitializeFn, init_to_uniform
from tqdm.auto import tqdm


class BaseInference(metaclass=ABCMeta):
    """
    Abstract class all inference methods should inherit from.
    """

    @staticmethod
    def _initialize_world(
        queries: List[RVIdentifier],
        observations: RVDict,
        initialize_fn: InitializeFn = init_to_uniform,
        max_retries: int = 100,
    ) -> World:
        """
        Initializes a world with all of the random variables (queries and observations).
        In case of initializing values outside of support of the distributions, the
        method will keep resampling until a valid initialization is found up to
        ``max_retries`` times.

        Args:
            queries: A list of random variables that need to be inferred.
            observations: A dictionary from random variables to their corresponding values.
            initialize_fn: A callable that takes in a distribution and returns a Tensor.
                The default behavior is to sample from Uniform(-2, 2) then biject to
                the support of the distribution.
            max_retries: The number of attempts this method will make before throwing an
                error (default to 100).
        """
        for _ in range(max_retries):
            world = World(observations, initialize_fn)
            # recursively add parent nodes to the graph
            for node in queries:
                world.call(node)
            for node in observations:
                world.call(node)

            # check if the initial state is valid
            log_prob = world.log_prob()
            if not torch.isinf(log_prob) and not torch.isnan(log_prob):
                return world

        # None of the world gives us a valid initial state
        raise ValueError(
            f"Cannot find a valid initialization after {max_retries} retries. The model"
            " might be misspecified."
        )

    @abstractmethod
    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        """
        Returns the proposer(s) corresponding to every non-observed variable
        in target_rvs.  Should be implemented by the specific inference algorithm.
        """
        raise NotImplementedError

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: int,
        num_chains: int = 4,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_fn: InitializeFn = init_to_uniform,
        max_init_retries: int = 100,
    ) -> MonteCarloSamples:
        """
        Performs inference and returns a ``MonteCarloSamples`` object with samples from the posterior.

        Args:
            queries: List of queries
            observations: Observations as an RVDict keyed by RVIdentifier
            num_samples: Number of samples.
            num_chains: Number of chains to run, defaults to 4.
            num_adaptive_samples:  Number of adaptive samples, defaults to 0.
            verbose: Verbose level for logging.
            initialize_fn: A callable that takes in a distribution and returns a Tensor.
                The default behavior is to sample from Uniform(-2, 2) then biject to
                the support of the distribution.
            max_init_retries: The number of attempts to make to initialize values for an
                inference before throwing an error (default to 100).
        """
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        chain_results = []
        chain_logll = []
        for _ in range(num_chains):
            sampler = self.sampler(
                queries,
                observations,
                num_samples,
                num_adaptive_samples,
                initialize_fn,
            )
            samples = {query: [] for query in queries}
            log_likelihoods = {obs: [] for obs in observations}
            # Main inference loop
            for world in tqdm(
                sampler,
                total=num_samples + num_adaptive_samples,
                desc="Samples collected",
                disable=verbose == VerboseLevel.OFF,
            ):
                for obs in observations:
                    log_likelihoods[obs].append(world.log_prob([obs]))
                # Extract samples
                for query in queries:
                    raw_val = world.call(query)
                    if not isinstance(raw_val, torch.Tensor):
                        raise TypeError(
                            "The value returned by a queried function must be a tensor."
                        )
                    samples[query].append(raw_val)

            samples = {node: torch.stack(val) for node, val in samples.items()}
            chain_results.append(samples)

            log_likelihoods = {
                node: torch.stack(val) for node, val in log_likelihoods.items()
            }
            chain_logll.append(log_likelihoods)

        return MonteCarloSamples(
            chain_results,
            num_adaptive_samples,
            chain_logll,
            observations,
        )

    def sampler(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: Optional[int] = None,
        num_adaptive_samples: int = 0,
        initialize_fn: InitializeFn = init_to_uniform,
        max_init_retries: int = 100,
    ) -> Sampler:
        """
        Returns a generator that returns a new world (represents a new state of the
        graph) each time it is iterated. If num_samples is not provided, this method
        will return an infinite generator.

        Args:
            queries: List of queries
            observations: Observations as an RVDict keyed by RVIdentifier
            num_samples: Number of samples, defaults to None for an infinite sampler.
            num_adaptive_samples:  Number of adaptive samples, defaults to 0.
            initialize_fn: A callable that takes in a distribution and returns a Tensor.
                The default behavior is to sample from Uniform(-2, 2) then biject to
                the support of the distribution.
            max_init_retries: The number of attempts to make to initialize values for an
                inference before throwing an error (default to 100).
        """
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        world = self._initialize_world(queries, observations, initialize_fn)
        # start inference with a copy of self to ensure that multi-chain or multi
        # inference runs all start with the same pristine state
        kernel = copy.deepcopy(self)
        sampler = Sampler(kernel, world, num_samples, num_adaptive_samples)
        return sampler
