# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    compute_first_gradient,
    is_valid,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World
from torch import Tensor
from torch.autograd import grad


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
        node_val = node_var.value
        score = world.compute_score(node_var)
        zero_grad(node_val)
        is_valid_gradient, gradient = compute_first_gradient(score, node_val)
        if not is_valid_gradient:
            zero_grad(node_val)
            return False, tensor(0.0), tensor(0.0)
        first_gradient = gradient.reshape(-1).clone()
        size = first_gradient.shape[0]
        hessian_diag = torch.zeros(size)
        for i in range(size):
            second_gradient = (
                grad(
                    # pyre-fixme
                    first_gradient.index_select(0, tensor([i])),
                    node_val,
                    create_graph=True,
                )[0]
            ).reshape(-1)

            if not is_valid(second_gradient):
                return False, tensor(0.0), tensor(0.0)

            hessian_diag[i] = second_gradient[i]

        zero_grad(node_val)
        node_val.detach()
        node_val_reshaped = node_val.reshape(-1)
        # pyre-fixme
        predicted_alpha = (1 - hessian_diag * (node_val_reshaped * node_val_reshaped)).T
        predicted_beta = -1 * node_val_reshaped * hessian_diag - first_gradient
        condition = (predicted_alpha > 0) & (predicted_beta > 0)
        predicted_alpha = torch.where(
            condition, predicted_alpha, tensor(1.0).to(dtype=predicted_beta.dtype)
        )
        mean = (
            # pyre-fixme
            node_var.distribution.mean.reshape(-1)
            if is_valid(node_var.distribution.mean)
            else torch.ones(predicted_beta.shape).to(dtype=predicted_beta.dtype)
        )
        predicted_beta = torch.where(condition, predicted_beta, mean)
        predicted_alpha.reshape(node_val.shape)
        predicted_beta.reshape(node_val.shape)
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
        if node_var.proposal_distribution is not None and number_of_variables == 1:
            return (node_var.proposal_distribution, {})

        is_valid, alpha, beta = self.compute_alpha_beta(node_var, world)
        if not is_valid:
            return super().get_proposal_distribution(
                node, node_var, world, auxiliary_variables
            )
        return (
            ProposalDistribution(
                proposal_distribution=dist.Gamma(alpha, beta),
                requires_transform=False,
                requires_reshape=True,
                arguments={},
            ),
            {},
        )
