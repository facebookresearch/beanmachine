# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Optional, Set, Tuple

import torch
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.inference.proposer.base_proposer import BaseProposer
from beanmachine.ppl.inference.sampler import Sampler
from beanmachine.ppl.inference.utils import (
    _execute_in_new_thread,
    _verify_queries_and_observations,
    seed as set_seed,
    VerboseLevel,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import init_to_uniform, InitializeFn, RVDict, World
from torch import multiprocessing as mp
from tqdm.auto import tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from typing_extensions import Literal


class BaseInference(metaclass=ABCMeta):
    """
    Abstract class all inference methods should inherit from.
    """

    # maximum value of a seed
    _MAX_SEED_VAL: int = 2**32 - 1

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

    def _get_default_num_adaptive_samples(self, num_samples: int) -> int:
        """
        Returns a reasonable default number of adaptive samples for the algorithm.
        """
        return 0

    def _single_chain_infer(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: int,
        num_adaptive_samples: int,
        verbose: VerboseLevel,
        initialize_fn: InitializeFn,
        max_init_retries: int,
        chain_id: int,
        seed: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Run a single chain of inference. Return a list of samples (in the same order as
        the queries) and a list of log likelihood on observations

        Args:
            queries: A list of queries.
            observations: A dictionary of observations.
            num_samples: Number of samples.
            num_adaptive_samples: Number of adaptive samples.
            verbose: Whether to display the progress bar or not.
            initialize_fn: A callable that takes in a distribution and returns a Tensor.
            max_init_retries: The number of attempts to make to initialize values for an
                inference before throwing an error.
            chain_id: The index of the current chain.
            seed: If provided, the seed will be used to initialize the state of the
            random number generators for the current chain
        """
        if seed is not None:
            set_seed(seed)

            # A hack to fix the issue where tqdm doesn't render progress bar correctly in
            # subprocess in Jupyter notebook (https://github.com/tqdm/tqdm/issues/485)
            if verbose == VerboseLevel.LOAD_BAR and issubclass(tqdm, notebook_tqdm):
                print(" ", end="", flush=True)

        sampler = self.sampler(
            queries,
            observations,
            num_samples,
            num_adaptive_samples,
            initialize_fn,
            max_init_retries,
        )
        samples = [[] for _ in queries]
        log_likelihoods = [[] for _ in observations]

        # Main inference loop
        for world in tqdm(
            sampler,
            total=num_samples + num_adaptive_samples,
            desc="Samples collected",
            disable=verbose == VerboseLevel.OFF,
            position=chain_id,
        ):
            for idx, obs in enumerate(observations):
                log_likelihoods[idx].append(world.log_prob([obs]))
            # Extract samples
            for idx, query in enumerate(queries):
                raw_val = world.call(query)
                if not isinstance(raw_val, torch.Tensor):
                    raise TypeError(
                        "The value returned by a queried function must be a tensor."
                    )
                samples[idx].append(raw_val)

        samples = [torch.stack(val) for val in samples]
        log_likelihoods = [torch.stack(val) for val in log_likelihoods]
        return samples, log_likelihoods

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: int,
        num_chains: int = 4,
        num_adaptive_samples: Optional[int] = None,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_fn: InitializeFn = init_to_uniform,
        max_init_retries: int = 100,
        run_in_parallel: bool = False,
        mp_context: Optional[Literal["fork", "spawn", "forkserver"]] = None,
    ) -> MonteCarloSamples:
        """
        Performs inference and returns a ``MonteCarloSamples`` object with samples from the posterior.

        Args:
            queries: List of queries
            observations: Observations as an RVDict keyed by RVIdentifier
            num_samples: Number of samples.
            num_chains: Number of chains to run, defaults to 4.
            num_adaptive_samples: Number of adaptive samples. If not provided, BM will
                fall back to algorithm-specific default value based on num_samples.
            verbose: Whether to display the progress bar or not.
            initialize_fn: A callable that takes in a distribution and returns a Tensor.
                The default behavior is to sample from Uniform(-2, 2) then biject to
                the support of the distribution.
            max_init_retries: The number of attempts to make to initialize values for an
                inference before throwing an error (default to 100).
            run_in_parallel: Whether to run multiple chains in parallel (with multiple
                processes) or not.
            mp_context: The `multiprocessing context <https://docs.python.org/3.8/library/multiprocessing.html#contexts-and-start-methods>`_
                to used for parallel inference.
        """
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        if num_adaptive_samples is None:
            num_adaptive_samples = self._get_default_num_adaptive_samples(num_samples)

        single_chain_infer = partial(
            self._single_chain_infer,
            queries,
            observations,
            num_samples,
            num_adaptive_samples,
            verbose,
            initialize_fn,
            max_init_retries,
        )
        if not run_in_parallel:
            chain_results = map(single_chain_infer, range(num_chains))
        else:
            # pyre-fixme[6]: For 1st param expected `None` but got `Union[str, str,
            #  str, None]`.
            ctx = mp.get_context(mp_context)
            # We'd like to explicitly set a different seed for each process to avoid
            # duplicating the same RNG state for all chains
            first_seed = torch.randint(self._MAX_SEED_VAL, ()).item()
            seeds = [
                (first_seed + 31 * chain_id) % self._MAX_SEED_VAL
                for chain_id in range(num_chains)
            ]
            # run single chain inference in a new thread in subprocesses to avoid
            # forking corrupted internal states
            # (https://github.com/pytorch/pytorch/issues/17199)
            single_chain_infer = partial(_execute_in_new_thread, single_chain_infer)

            with ctx.Pool(
                processes=num_chains, initializer=tqdm.set_lock, initargs=(ctx.Lock(),)
            ) as p:
                chain_results = p.starmap(single_chain_infer, enumerate(seeds))

        all_samples, all_log_liklihoods = zip(*chain_results)
        # the hash of RVIdentifier can change when it is being sent to another process,
        # so we have to rely on the order of the returned list to determine which samples
        # correspond to which RVIdentifier
        all_samples = [dict(zip(queries, samples)) for samples in all_samples]
        # in python the order of keys in a dict is fixed, so we can rely on it
        all_log_liklihoods = [
            dict(zip(observations.keys(), log_likelihoods))
            for log_likelihoods in all_log_liklihoods
        ]

        return MonteCarloSamples(
            all_samples,
            num_adaptive_samples,
            all_log_liklihoods,
            observations,
        )

    def sampler(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: Optional[int] = None,
        num_adaptive_samples: Optional[int] = None,
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
            num_adaptive_samples:  Number of adaptive samples. If not provided, BM will
                fall back to algorithm-specific default value based on num_samples. If
                num_samples is not provided either, then defaults to 0.
            initialize_fn: A callable that takes in a distribution and returns a Tensor.
                The default behavior is to sample from Uniform(-2, 2) then biject to
                the support of the distribution.
            max_init_retries: The number of attempts to make to initialize values for an
                inference before throwing an error (default to 100).
        """
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        if num_adaptive_samples is None:
            if num_samples is None:
                num_adaptive_samples = 0
            else:
                num_adaptive_samples = self._get_default_num_adaptive_samples(
                    num_samples
                )

        world = World.initialize_world(
            queries,
            observations,
            initialize_fn,
            max_init_retries,
        )
        # start inference with a copy of self to ensure that multi-chain or multi
        # inference runs all start with the same pristine state
        kernel = copy.deepcopy(self)
        sampler = Sampler(kernel, world, num_samples, num_adaptive_samples)
        return sampler
