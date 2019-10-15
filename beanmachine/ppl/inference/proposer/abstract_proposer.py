# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch.tensor as tensor
from beanmachine.ppl.model.utils import RandomVariable
from torch import Tensor


class AbstractProposer(object, metaclass=ABCMeta):
    """
    Abstract proposer object that all proposer algorithms inherit from.
    """

    def __init__(self, world):
        self.world_ = world

    @abstractmethod
    def propose(self, node: RandomVariable) -> Tuple[Tensor, Tensor]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractProposer. This implementation will include how the proposer
        proposes a new value for a given node.

        :param node: the node for which we'll need to propose a new value for.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")

    def post_process(self, node: RandomVariable) -> Tensor:
        """
        To be implemented by proposers that need post-processing after diff is
        created to compute the final proposal log update.

        :param node: the node for which we have already proposed a new value for.
        :returns: the log probability of proposing the old value from this new world.
        """
        return tensor(0.0)
