# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import Variable, World
from torch import Tensor


class AbstractSingleSiteProposer(object, metaclass=ABCMeta):
    """
    Abstract proposer object that all proposer algorithms inherit from.
    """

    @abstractmethod
    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        """
        Abstract method to be implemented by classes that inherit from
        AbstractProposer. This implementation will include how the proposer
        proposes a new value for a given node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value and auxiliary variables that needs to be passed
        to post process.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")

    @abstractmethod
    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:
        """
        To be implemented by proposers that need post-processing after diff is
        created to compute the final proposal log update.

        :param node: the node for which we have already proposed a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param auxiliary_variables: Dict of auxiliary variables that is passed
        from propose.
        :returns: the log probability of proposing the old value from this new world.
        """
        raise NotImplementedError("Inference algorithm must implement propose.")

    def do_adaptation(
        self,
        node: RVIdentifier,
        node_var: Variable,
        node_acceptance_results: Tensor,
        iteration_number: int,
        num_adapt_steps: int,
    ) -> None:
        """
        To be implemented by proposers that are capable of adaptation at
        the beginning of the chain.

        :param node: the node for which we have already proposed a new value for.
        :param node_var: the Variable object associated with node.
        :param node_acceptance_results: the boolean values of acceptances for
         values collected so far within _infer().
        :param iteration_number: The current iteration of inference
        :param num_adapt_steps: The number of inference iterations for adaptation.
        :returns: Nothing.
        """
        return

        """ No need for: NotImplementedError(
        "Inference and proposer algorithm must implement do_adaptation.")
        This is because not all inference methods require adaptation.
        """
