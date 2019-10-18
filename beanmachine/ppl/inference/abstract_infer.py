# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict, List

import torch
import torch.multiprocessing as mp
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import RandomVariable
from torch import Tensor
from torch.multiprocessing import Queue


class AbstractInference(object, metaclass=ABCMeta):
    """
    Abstract inference object that all inference algorithms inherit from.
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def _infer(self, num_samples: int) -> Dict[RandomVariable, Tensor]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractInference.
        """
        raise NotImplementedError("Inference algorithm must implement _infer.")

    def _parallel_infer(self, queue: Queue, chain: int, num_samples: int):
        try:
            torch.seed()
            rv_dict = self._infer(num_samples)
            string_dict = {str(rv): tensor.detach() for rv, tensor in rv_dict.items()}
            queue.put((None, chain, string_dict))
        except BaseException as x:
            queue.put((x, chain, {}))

    def infer(
        self,
        queries: List[RandomVariable],
        observations: Dict[RandomVariable, Tensor],
        num_samples: int,
        num_chains: int = 4,
    ) -> MonteCarloSamples:
        """
        Run inference algorithms and reset the world/mode at the end.

        :param queries: random variables to query
        :param observations: observed random variables with their values
        :params num_samples: number of samples to collect for the query.
        :params num_chains: number of chains to run
        :returns: view of data for chains and samples for query
        """
        try:
            self.queries_ = queries
            self.observations_ = observations

            manager = mp.Manager()
            q = manager.Queue()
            for chain in range(num_chains):
                p = mp.Process(
                    target=self._parallel_infer, args=(q, chain, num_samples)
                )
                p.start()

            chain_queries = [{}] * num_chains
            for _ in range(num_chains):
                (error, chain, string_dict) = q.get()
                if error is not None:
                    raise error
                rv_dict = {rv: string_dict[str(rv)] for rv in queries}
                chain_queries[chain] = rv_dict

            monte_carlo_samples = MonteCarloSamples(chain_queries)
        except BaseException as x:
            raise x
        finally:
            self.reset()
        return monte_carlo_samples

    def reset(self):
        """
        Resets world, mode and observation
        """
        self.stack_, self.world_ = StatisticalModel.reset()
        self.queries_ = []
        self.observations_ = {}
