import copy
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Set

import torch
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.sampler import Sampler
from beanmachine.ppl.inference.abstract_infer import (
    VerboseLevel,
    _verify_queries_and_observations,
)
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world import RVDict, World, InitializeFn, init_to_uniform
from tqdm.auto import tqdm


class BaseInference(metaclass=ABCMeta):
    @staticmethod
    def _initialize_world(
        queries: List[RVIdentifier],
        observations: RVDict,
        initialize_fn: InitializeFn = init_to_uniform,
    ) -> World:
        world = World(observations, initialize_fn)
        # recursively add parent nodes to the graph
        for node in queries:
            world.call(node)
        for node in observations:
            world.call(node)
        return world

    @abstractmethod
    def get_proposers(
        self,
        world: World,
        target_rvs: Set[RVIdentifier],
        num_adaptive_sample: int,
    ) -> List[BaseProposer]:
        raise NotImplementedError

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: int,
        num_chains: int = 1,
        num_adaptive_samples: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_fn: InitializeFn = init_to_uniform,
    ) -> MonteCarloSamples:
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        chain_results = []
        for _ in range(num_chains):
            sampler = self.sampler(
                queries,
                observations,
                num_samples,
                num_adaptive_samples,
                initialize_fn,
            )
            samples = {query: [] for query in queries}
            # Main inference loop
            for world in tqdm(
                sampler,
                total=num_samples + num_adaptive_samples,
                desc="Samples collected",
                disable=verbose == VerboseLevel.OFF,
            ):
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
        return MonteCarloSamples(chain_results, num_adaptive_samples)

    def sampler(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: Optional[int] = None,
        num_adaptive_samples: int = 0,
        initialize_fn: InitializeFn = init_to_uniform,
    ) -> Sampler:
        """Returns a generator that returns a new world (represents a new state of the
        graph) each time it is iterated. If num_samples is not provided, this method
        will returns an infinite generator."""
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        world = self._initialize_world(queries, observations, initialize_fn)
        # start inference with a copy of self to ensure that multi-chain or multi
        # inference runs all start with the same pristine state
        kernel = copy.deepcopy(self)
        sampler = Sampler(kernel, world, num_samples, num_adaptive_samples)
        return sampler
