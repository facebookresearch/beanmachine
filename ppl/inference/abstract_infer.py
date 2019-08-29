# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod

from beanmachine.ppl.model.statistical_model import StatisticalModel


class AbstractInference(object, metaclass=ABCMeta):
    """
    Abstract inference object that all inference algorithms inherit from.
    """

    def __init__(self, queries, observations):
        self.queries_ = queries
        self.observations_ = observations
        self.reset()

    @abstractmethod
    def _infer(self, num_samples):
        """
        Abstract method to be implemented by classes that inherit from
        AbstractInference.
        """
        raise NotImplementedError("Inference algorithm must implement _infer.")

    def infer(self, num_samples):
        """
        Run inference algorithms and reset the world/mode at the end.

        parameter is:
            num_samples: number of samples to collect for the query.

        returns:
            samples for the query
        """
        try:
            queries_sample = self._infer(num_samples)
        except BaseException as x:
            raise x
        finally:
            self.reset()
        return queries_sample

    def reset(self):
        """
        Resets world, mode and observation
        """
        self.stack_, self.world_ = StatisticalModel.reset()
