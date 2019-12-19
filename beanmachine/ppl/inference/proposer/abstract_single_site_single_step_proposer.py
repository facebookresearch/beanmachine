# Copyright (c) Facebook, Inc. and its affiliates
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch.distributions as dist
from beanmachine.ppl.inference.proposer.abstract_single_site_proposer import (
    AbstractSingleSiteProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World
from torch import Tensor


class AbstractSingleSiteSingleStepProposer(
    AbstractSingleSiteProposer, metaclass=ABCMeta
):
    """
    Abstract Single-Site Single-Step Proposer
    """

    def propose(self, node: RVIdentifier, world: World) -> Tuple[Tensor, Tensor]:
        """
        Proposes a new value for the node.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we'll propose a new value for node.
        :returns: a new proposed value for the node and the -ve log probability of
        proposing this new value.
        """
        node_var = world.get_node_in_world_raise_error(node, False)
        number_of_variables = world.get_number_of_variables()

        # if the number of variables in the world is 1 and proposal distribution
        # has already been computed, we can use the old proposal distribution
        # and skip re-computing the gradient, since there are no other variable
        # in the world that may change the gradient and the old one is still
        # correct.
        if node_var.proposal_distribution is None or number_of_variables != 1:
            proposal_distribution_struct = self.get_proposal_distribution(
                node, node_var, world
            )
            node_var.proposal_distribution = proposal_distribution_struct
        proposal_distribution_struct = node_var.proposal_distribution
        proposal_distribution = proposal_distribution_struct.proposal_distribution
        requires_transform = proposal_distribution_struct.requires_transform
        requires_reshape = proposal_distribution_struct.requires_reshape

        # pyre-fixme
        new_value = proposal_distribution.sample()
        negative_proposal_log_update = (
            -1
            # pyre-fixme
            * proposal_distribution.log_prob(new_value).sum()
        )
        if requires_reshape:
            if isinstance(node_var.distribution, dist.Beta) and isinstance(
                proposal_distribution, dist.Dirichlet
            ):
                new_value = new_value.transpose(-1, 0)[0].T.reshape(
                    node_var.value.shape
                )

            new_value = new_value.reshape(node_var.unconstrained_value.shape)
        if requires_transform:
            new_value = node_var.transform_from_unconstrained_to_constrained(new_value)
            negative_proposal_log_update = (
                negative_proposal_log_update + node_var.jacobian
            )

        return (new_value, negative_proposal_log_update)

    def post_process(self, node: RVIdentifier, world: World) -> Tensor:
        """
        Computes the log probability of going back to the old value.

        :param node: the node for which we'll need to propose a new value for.
        :param world: the world in which we have already proposed a new value
        for node.
        :returns: the log probability of proposing the old value from this new world.
        """
        old_unconstrained_value = world.get_old_unconstrained_value(node)

        if old_unconstrained_value is None:
            raise ValueError("old unconstrained value is not available in world")

        node_var = world.get_node_in_world_raise_error(node, False)
        number_of_variables = world.get_number_of_variables()

        # if the number of variables in the world is 1 and proposal distribution
        # has already been computed, we can use the old proposal distribution
        # and skip re-computing the gradient, since there are no other variable
        # in the world that may change the gradient and the old one is still
        # correct.
        if node_var.proposal_distribution is None or number_of_variables != 1:
            proposal_distribution_struct = self.get_proposal_distribution(
                node, node_var, world
            )
            node_var.proposal_distribution = proposal_distribution_struct

        proposal_distribution_struct = node_var.proposal_distribution
        proposal_distribution = proposal_distribution_struct.proposal_distribution
        requires_transform = proposal_distribution_struct.requires_transform
        requires_reshape = proposal_distribution_struct.requires_reshape

        if (
            requires_reshape
            and not isinstance(node_var.distribution, dist.Beta)
            and not isinstance(node_var.distribution, dist.Gamma)
        ):
            old_unconstrained_value = old_unconstrained_value.reshape(-1)

        # pyre-fixme
        positive_log_update = proposal_distribution.log_prob(
            old_unconstrained_value
        ).sum()

        if requires_transform:
            positive_log_update = positive_log_update - node_var.jacobian
        return positive_log_update

    @abstractmethod
    def get_proposal_distribution(
        self, node: RVIdentifier, node_var: Variable, world: World
    ) -> ProposalDistribution:
        """
        Returns the proposal distribution of the node.

        :param node: the node for which we're proposing a new value for
        :param node_var: the Variable of the node
        :param world: the world in which we're proposing a new value for node
        :returns: the proposal distribution of the node
        """
        raise NotImplementedError("get_proposal_distribution needs to be implemented")
