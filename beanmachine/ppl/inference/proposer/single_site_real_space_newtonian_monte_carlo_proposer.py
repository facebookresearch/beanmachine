# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    compute_first_gradient,
    compute_neg_hessian_invserse,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World
from torch import Tensor


class SingleSiteRealSpaceNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Half Space Newtonian Monte Carlo Proposers
    """

    def compute_normal_mean_covar(
        self, node_var: Variable, world: World
    ) -> Tuple[bool, Tensor, Tensor]:
        """
        Computes mean and covariance of the MultivariateNormal given the node.
            mean = x - first_grad * hessian_inverse
            covariance = -1 * hessian_inverse

        :param node_var: the node Variable we're proposing a new value for
        :returns: mean and covariance
        """
        node_val = node_var.unconstrained_value
        score = world.compute_score(node_var)
        zero_grad(node_val)
        is_valid_gradient, gradient = compute_first_gradient(score, node_val)

        if not is_valid_gradient:
            zero_grad(node_val)
            return False, tensor(0.0), tensor(0.0)

        first_gradient = gradient.reshape(-1).clone()
        is_valid_neg_invserse_hessian, neg_hessian_inverse = compute_neg_hessian_invserse(
            first_gradient, node_val
        )
        zero_grad(node_val)
        node_val.detach()
        if not is_valid_neg_invserse_hessian:
            return False, tensor(0.0), tensor(0.0)

        # node value may of any arbitrary shape, so here, we transform it into a
        # 1D vector using reshape(-1) and with unsqueeze(0), we change 1D vector
        # of size (N) to (1 x N) matrix.
        node_reshaped = node_val.reshape(-1).unsqueeze(0)
        # here we again call unsqueeze(0) on first_gradient to transform it into
        # a matrix in order to be able to perform matrix multiplication.
        mean = (
            node_reshaped
            # pyre-fixme
            + first_gradient.unsqueeze(0).mm(neg_hessian_inverse)
        ).squeeze(0)
        covariance = neg_hessian_inverse
        return True, mean, covariance

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
        is_valid, mean, covariance = self.compute_normal_mean_covar(node_var, world)
        if not is_valid:
            return super().get_proposal_distribution(node, node_var, world)

        requires_transform = False
        # pyre-fixme
        if not isinstance(node_var.distribution.support, dist.constraints._Real):
            requires_transform = True
        return ProposalDistribution(
            proposal_distribution=dist.MultivariateNormal(mean, covariance),
            requires_transform=requires_transform,
            requires_reshape=True,
        )
