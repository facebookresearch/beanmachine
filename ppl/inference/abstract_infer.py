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
        self.stack_, self.world_ = StatisticalModel.initialize()

    @abstractmethod
    def infer(self, num_samples):
        """
        Abstract method to be implemented by classes that inherit from
        AbstractInference.
        """
        raise NotImplementedError("Inference algorithm must implement infer.")
