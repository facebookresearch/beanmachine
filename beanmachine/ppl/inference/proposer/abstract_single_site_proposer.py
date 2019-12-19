# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Tuple

from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import World
from torch import Tensor


class AbstractSingleSiteProposer(object, metaclass=ABCMeta):
    """
    Abstract proposer object that all proposer algorithms inherit from.
    """

    @abstractmethod
    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractProposer. This implementation will include how the proposer
        proposes a new value for a given node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")

    @abstractmethod
    def post_process(self, node: RVIdentifier, world: World) -> Tensor:
        """
        To be implemented by proposers that need post-processing after diff is
        created to compute the final proposal log update.

        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :returns: the log probability of proposing the old value from this new world.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")
