# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils import (
    is_valid,
    zero_grad,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.world import ProposalDistribution, Variable, World
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils import tensorops
from torch import Tensor, tensor


LOGGER = logging.getLogger("beanmachine")


class SingleSiteHalfSpaceNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Half Space Newtonian Monte Carlo Proposers
    """

    def compute_alpha_beta(
        self, node_var: Variable, world: World
    ) -> Tuple[bool, Tensor, Tensor]:
        """
        Computes alpha and beta of the Gamma proposal given the node.
            alpha = 1 - hessian_diag * x^2
            beta = -1 * x * hessian_diag - first_grad

        :param node_var: the node Variable we're proposing a new value for
        :returns: alpha and beta of the Gamma distribution as proposal
        distribution
        """
        node_val = node_var.transformed_value
        score = world.compute_score(node_var)
        zero_grad(node_val)
        first_gradient, hessian_diag = tensorops.halfspace_gradients(score, node_val)
        zero_grad(node_val)
        if not is_valid(first_gradient) or not is_valid(hessian_diag):
            LOGGER.warning(
                "Gradient or Hessian is invalid at node {n}.\n".format(n=str(node_var))
            )
            return False, tensor(0.0), tensor(0.0)
        node_val_reshaped = node_val.reshape(-1)
        # pyre-fixme
        predicted_alpha = (1 - hessian_diag * (node_val_reshaped * node_val_reshaped)).T
        predicted_beta = -1 * node_val_reshaped * hessian_diag - first_gradient
        # pyre-fixme[58]: `&` is not supported for operand types `bool` and
        #  `ByteTensor`.
        condition = (predicted_alpha > 0) & (predicted_beta > 0)
        predicted_alpha = torch.where(
            condition, predicted_alpha, tensor(1.0).to(dtype=predicted_beta.dtype)
        )
        mean = (
            node_var.distribution.mean.reshape(-1)
            if is_valid(node_var.distribution.mean)
            else torch.ones(predicted_beta.shape).to(dtype=predicted_beta.dtype)
        )
        predicted_beta = torch.where(condition, predicted_beta, mean)
        predicted_alpha = predicted_alpha.reshape(node_val.shape)
        predicted_beta = predicted_beta.reshape(node_val.shape)
        return True, predicted_alpha, predicted_beta

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
        :param auxiliary_variables: additional auxiliaryvariables that may
        be required to find
        a proposal distribution
        :returns: the tuple of proposal distribution of the node and arguments
        that was used or needs to be used to find the proposal distribution
        """
        # if the number of variables in the world is 1 and proposal distribution
        # has already been computed, we can use the old proposal distribution
        # and skip re-computing the gradient, since there are no other variable
        # in the world that may change the gradient and the old one is still
        # correct.
        number_of_variables = world.get_number_of_variables()
        proposal_distribution = node_var.proposal_distribution
        if proposal_distribution is not None and number_of_variables == 1:
            return (proposal_distribution, {})

        is_valid, alpha, beta = self.compute_alpha_beta(node_var, world)
        if not is_valid:
            LOGGER.warning(
                "Node {n} has invalid proposal solution. ".format(n=node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(
                node, node_var, world, auxiliary_variables
            )
        return (
            ProposalDistribution(
                proposal_distribution=dist.Gamma(alpha, beta),
                requires_transform=True,
                requires_reshape=False,
                arguments={},
            ),
            {},
        )
