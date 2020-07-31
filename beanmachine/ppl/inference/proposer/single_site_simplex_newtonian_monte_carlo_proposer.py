# Copyright (c) Facebook, Inc. and its affiliates
import logging
from typing import Dict, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    is_valid,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.utils import tensorops  # pyre-ignore
from beanmachine.ppl.world import ProposalDistribution, Variable, World
from torch import Tensor


LOGGER_WARNING = logging.getLogger("beanmachine.warning")


class SingleSiteSimplexNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Simplex Newtonian Monte Carlo Proposers
    """

    def __init__(self):
        self.reshape_untransformed_beta_to_dirichlet = True

    def compute_alpha(
        self, node_var: Variable, world: World, min_alpha_value: float = 1e-3
    ) -> Tuple[bool, Tensor]:
        """
        Computes alpha of the Dirichlet proposal given the node.
            alpha = 1 - (x^2) (hessian[i, i] - max(hessian[i]))
                where max(hessian[i]) is maximum of the hessian at ith row
                excluding the diagonal values.

        :param node_var: the node Variable we're proposing a new value for
        :returns: alpha of the Dirichlet distribution as proposal distribution
        """
        node_val = node_var.transformed_value
        score = world.compute_score(node_var)
        # ensures gradient is zero at the end of each proposals.
        zero_grad(node_val)
        # pyre-fixme
        first_gradient, hessian_diag_minus_max = tensorops.simplex_gradients(
            score, node_val
        )
        zero_grad(node_val)
        if not is_valid(first_gradient) or not is_valid(hessian_diag_minus_max):
            LOGGER_WARNING.warning(
                "Gradient or Hessian is invalid at node {n}.\n".format(n=str(node_var))
            )
            return False, tensor(0.0)
        node_val_reshaped = node_val.clone().reshape(-1)
        predicted_alpha = (
            1 - ((node_val_reshaped * node_val_reshaped) * (hessian_diag_minus_max))
        ).reshape(node_val.shape)

        # pyre-fixme
        mean = node_var.distribution.mean

        predicted_alpha = torch.where(
            predicted_alpha < -1 * min_alpha_value, mean, predicted_alpha
        )

        predicted_alpha = torch.where(
            predicted_alpha > 0, predicted_alpha, tensor(min_alpha_value)
        )
        return True, predicted_alpha

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
        # if the number of variables in the world is 1 and proposal distribution
        # has already been computed, we can use the old proposal distribution
        # and skip re-computing the gradient, since there are no other variable
        # in the world that may change the gradient and the old one is still
        # correct.
        number_of_variables = world.get_number_of_variables()
        if node_var.proposal_distribution is not None and number_of_variables == 1:
            return (node_var.proposal_distribution, {})

        is_valid, alpha = self.compute_alpha(node_var, world)
        if not is_valid:
            LOGGER_WARNING.warning(
                "Node {n} has invalid proposal solution. ".format(n=node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(
                node, node_var, world, auxiliary_variables
            )
        return (
            ProposalDistribution(
                proposal_distribution=dist.Dirichlet(alpha),
                requires_transform=True,
                requires_reshape=False,
                arguments={},
            ),
            {},
        )
