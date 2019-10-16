# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.statistical_model import StatisticalModel
from beanmachine.ppl.model.utils import RandomVariable
from torch import Tensor


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

            chain_queries = []
            for _ in range(num_chains):
                chain_queries.append(self._infer(num_samples))
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
