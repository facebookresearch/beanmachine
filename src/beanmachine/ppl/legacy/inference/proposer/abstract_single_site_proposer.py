# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

from beanmachine.ppl.legacy.world import TransformType, World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


class AbstractSingleSiteProposer(object, metaclass=ABCMeta):
    """
    Abstract proposer object that all proposer algorithms inherit from.
    """

    def __init__(
        self,
        transform_type: TransformType = TransformType.NONE,
        transforms: Optional[List] = None,
    ):
        if transform_type is TransformType.CUSTOM and transforms is None:
            raise ValueError("Please specify the transform")
        self.transform_type = transform_type
        self.transforms = transforms

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
        raise NotImplementedError("Inference algorithm must implement post_process.")

    def do_adaptation(
        self,
        node: RVIdentifier,
        world: World,
        acceptance_probability: Tensor,
        iteration_number: int,
        num_adaptive_samples: int,
        is_accepted: bool,
    ) -> None:
        """
        To be implemented by proposers that are capable of adaptation at
        the beginning of the chain.

        :param node: the node in `world` to perform proposer adaptation for
        :param world: the new world if `is_accepted`, or the previous world
        otherwise.
        :param acceptance_probability: the acceptance probability of the previous move.
        :param iteration_number: the current iteration of inference
        :param num_adaptive_samples: the number of inference iterations for adaptation.
        :param is_accepted: bool representing whether the new value was accepted.
        :returns: nothing.
        """
        return

        """ No need for: NotImplementedError(
        "Inference and proposer algorithm must implement do_adaptation.")
        This is because not all inference methods require adaptation.
        """
