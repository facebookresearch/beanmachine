# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Dict, List

import torch
import torch.multiprocessing as mp
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import RVIdentifier
from torch import Tensor
from torch.multiprocessing import Queue


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

    def __init__(self):
        self.reset()

    @abstractmethod
    def _infer(
        self,
        num_samples: int,
        num_adapt_steps: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
    ) -> Dict[RVIdentifier, Tensor]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractInference.
        """
        raise NotImplementedError("Inference algorithm must implement _infer.")

    @staticmethod
    def set_seed_for_chain(random_seed: int, chain: int):
        torch.manual_seed(random_seed + chain * 31)

    def _parallel_infer(
        self,
        queue: Queue,
        chain: int,
        num_samples: int,
        random_seed: int,
        num_adapt_steps: int,
        verbose: VerboseLevel,
    ):
        try:
            AbstractInference.set_seed_for_chain(random_seed, chain)
            rv_dict = self._infer(num_samples, num_adapt_steps, verbose)
            string_dict = {str(rv): tensor.detach() for rv, tensor in rv_dict.items()}
            queue.put((None, chain, string_dict))
        except BaseException as x:
            queue.put((x, chain, {}))

    def infer(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Tensor],
        num_samples: int,
        num_chains: int = 4,
        run_in_parallel: bool = False,
        num_adapt_steps: int = 0,
        verbose: VerboseLevel = VerboseLevel.LOAD_BAR,
    ) -> MonteCarloSamples:
        """
        Run inference algorithms and reset the world/mode at the end.

        :param queries: random variables to query
        :param observations: observed random variables with their values
        :params num_samples: number of samples excluding adaptation to collect.
        :params num_chains: number of chains to run
        :params num_adapt_steps: number of steps to allow proposer adaptation.
        :param verbose: Integer indicating how much output to print to stdio
        :returns: view of data for chains and samples for query
        """
        try:
            self.queries_ = queries
            self.observations_ = observations
            # pyre-fixme[6]: Expected `Union[bytes, str, typing.SupportsInt]` for
            #  1st param but got `Tensor`.
            random_seed = int(torch.randint(2 ** 62, (1,)))

            if num_chains > 1 and run_in_parallel:
                manager = mp.Manager()
                q = manager.Queue()
                for chain in range(num_chains):
                    p = mp.Process(
                        target=self._parallel_infer,
                        args=(q, chain, num_samples, random_seed),
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
                    AbstractInference.set_seed_for_chain(random_seed, chain)
                    rv_dicts = self._infer(num_samples, num_adapt_steps, verbose)
                    chain_queries.append(rv_dicts)

            monte_carlo_samples = MonteCarloSamples(chain_queries, num_adapt_steps)
        except BaseException as x:
            raise x
        finally:
            self.reset()
        return monte_carlo_samples

    def reset(self):
        """
        Resets world, mode and observation
        """
        self.world_ = StatisticalModel.reset()
        self.queries_ = []
        self.observations_ = {}
