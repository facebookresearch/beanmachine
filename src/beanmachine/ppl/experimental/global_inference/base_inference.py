from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Optional

import torch
import torch.multiprocessing as mp
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.sampler import Sampler
from beanmachine.ppl.experimental.global_inference.simple_world import (
    RVDict,
    SimpleWorld,
)
from beanmachine.ppl.inference.abstract_infer import (
    AbstractMCInference,
    VerboseLevel,
    _verify_queries_and_observations,
)
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from tqdm.auto import trange


class BaseInference(metaclass=ABCMeta):
    @staticmethod
    def _initialize_world(
        queries: List[RVIdentifier],
        observations: RVDict,
        initialize_from_prior: bool,
    ) -> SimpleWorld:
        world = SimpleWorld(observations, initialize_from_prior)
        # recursively add parent nodes to the graph
        for node in queries:
            world.call(node)
        for node in observations:
            world.call(node)
        return world

    @abstractmethod
    def get_proposer(self, world: SimpleWorld) -> BaseProposer:
        raise NotImplementedError

    def _single_chain_infer(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: int,
        num_adaptive_samples: int,
        verbose: VerboseLevel,
        initialize_from_prior: bool,
        seed: int,
        chain_id: int = 0,
    ) -> List[torch.Tensor]:
        AbstractMCInference.set_seed_for_chain(seed, chain_id)
        sampler = self.sampler(
            queries,
            observations,
            num_samples,
            num_adaptive_samples,
            initialize_from_prior,
        )
        samples = {query: [] for query in queries}
        # Main inference loop
        for _ in trange(
            num_samples + num_adaptive_samples,
            desc="Samples collected",
            disable=verbose == VerboseLevel.OFF,
        ):
            world = next(sampler)
            # Extract samples
            for query in queries:
                samples[query].append(world[query].detach().clone())

        # return values in the same order as queries
        return [torch.stack(samples[node]) for node in queries]

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: int,
        num_adaptive_samples: int = 0,
        num_chains: int = 1,
        run_in_parallel: bool = False,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
        initialize_from_prior: bool = False,
        seed: Optional[int] = None,
    ) -> MonteCarloSamples:
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        if seed is None:
            seed = torch.randint(AbstractMCInference._rand_int_max, (1,)).int().item()

        # de-duplicate then fix the order of the queries
        queries = list(set(queries))
        infer_func = partial(
            self._single_chain_infer,
            queries,
            observations,
            num_samples,
            num_adaptive_samples,
            verbose,
            initialize_from_prior,
            seed,
        )

        if run_in_parallel:
            with mp.Pool(num_chains) as pool:
                chain_results = pool.map(infer_func, range(num_chains))
        else:
            chain_results = map(infer_func, range(num_chains))

        # re-map queries to samples
        samples = []
        for chain_result in chain_results:
            samples.append({node: value for node, value in zip(queries, chain_result)})

        return MonteCarloSamples(samples, num_adaptive_samples)

    def sampler(
        self,
        queries: List[RVIdentifier],
        observations: RVDict,
        num_samples: Optional[int] = None,
        num_adaptive_samples: int = 0,
        initialize_from_prior: bool = False,
    ) -> Sampler:
        """Returns a generator that returns a new world (represents a new state of the
        graph) each time it is iterated. If num_samples is not provided, this method
        will returns an infinite generator."""
        _verify_queries_and_observations(
            queries, observations, observations_must_be_rv=True
        )
        world = self._initialize_world(queries, observations, initialize_from_prior)
        proposer = self.get_proposer(world)
        sampler = Sampler(proposer, num_samples, num_adaptive_samples)
        return sampler
