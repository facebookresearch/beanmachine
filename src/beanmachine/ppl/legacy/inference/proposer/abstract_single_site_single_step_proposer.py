# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

from beanmachine.ppl.inference.utils import safe_log_prob_sum
from beanmachine.ppl.legacy.inference.proposer.abstract_single_site_proposer import (
    AbstractSingleSiteProposer,
)
from beanmachine.ppl.legacy.world import ProposalDistribution, Variable, World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.model.utils import LogLevel
from torch import Tensor


LOGGER_PROPOSER = logging.getLogger("beanmachine.proposer")


class AbstractSingleSiteSingleStepProposer(
    AbstractSingleSiteProposer, metaclass=ABCMeta
):
    """
    Abstract Single-Site Single-Step Proposer
    """

    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor, Dict]:
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value and dict of auxiliary variables that needs to
        be passed to post process.
        """
        node_var = world.get_node_in_world_raise_error(node, False)

        (
            proposal_distribution_struct,
            auxiliary_variables,
        ) = self.get_proposal_distribution(node, node_var, world, {})
        node_var.proposal_distribution = proposal_distribution_struct
        proposal_distribution = proposal_distribution_struct.proposal_distribution
        requires_transform = proposal_distribution_struct.requires_transform
        requires_reshape = proposal_distribution_struct.requires_reshape

        LOGGER_PROPOSER.log(
            LogLevel.DEBUG_PROPOSER.value,
            "- Distribution: {pt}\n".format(pt=str(proposal_distribution_struct))
            + "- Auxiliary params: {pa}\n".format(pa=str(auxiliary_variables)),
        )

        new_value = proposal_distribution.sample()
        negative_proposal_log_update = (
            -1.0 * proposal_distribution.log_prob(new_value).sum()
        )

        if requires_reshape:
            new_value = new_value.reshape(node_var.transformed_value.shape)

        if requires_transform:
            new_value = node_var.inverse_transform_value(new_value)
            negative_proposal_log_update = (
                negative_proposal_log_update - node_var.jacobian
            )

        # pyre-fixme[7]: Expected Tuple[Tensor, Tensor, Dict[typing.Any,
        # typing.Any]] but got Tuple[typing.Any, typing.Union[Tensor, float],
        # Dict[typing.Any, typing.Any]].
        return (
            new_value,
            negative_proposal_log_update,
            auxiliary_variables,
        )

    def post_process(
        self, node: RVIdentifier, world: World, auxiliary_variables: Dict
    ) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :param auxiliary_variables: Dict of auxiliary variables that is passed
        from propose.
        :returns: the log probability of proposing the old value from this new world.
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        proposal_distribution_struct, _ = self.get_proposal_distribution(
            node, node_var, world, auxiliary_variables
        )
        node_var.proposal_distribution = proposal_distribution_struct

        proposal_distribution = proposal_distribution_struct.proposal_distribution
        requires_transform = proposal_distribution_struct.requires_transform
        requires_reshape = proposal_distribution_struct.requires_reshape

        if requires_transform:
            old_value = world.get_old_transformed_value(node)
        else:
            old_value = world.get_old_value(node)
        if old_value is None:
            raise ValueError("old value is not available in world")

        if requires_reshape:
            old_value = old_value.reshape(-1)
        positive_log_update = safe_log_prob_sum(
            proposal_distribution,
            old_value.to(proposal_distribution.sample().device),
        ).to(old_value.device)

        if requires_transform:
            positive_log_update = positive_log_update + node_var.jacobian

        return positive_log_update

    @abstractmethod
    def get_proposal_distribution(
        self,
        node: RVIdentifier,
        node_var: Variable,
        world: World,
        auxiliary_variables: Dict,
    ) -> Tuple[ProposalDistribution, Dict]:
        """
        Returns the proposal distribution of the node.

        :param node: the node for which we're proposing a new value for
        :param node_var: the Variable of the node
        :param world: the world in which we're proposing a new value for node
        :param auxiliary_variables: additional auxiliary variables that may be
        required to find a proposal distribution
        :returns: the tuple of proposal distribution of the node and arguments
        that was used or needs to be used to find the proposal distribution
        """
        raise NotImplementedError("get_proposal_distribution needs to be implemented")
