# Copyright (c) Facebook, Inc. and its affiliates
from typing import Tuple

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


class SingleSiteSimplexNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Simplex Newtonian Monte Carlo Proposers
    """

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
        node_val = (
            node_var.value if node_var.extended_val is None else node_var.extended_val
        )
        score = world.compute_score(node_var)
        is_valid_gradient, gradient = compute_first_gradient(score, node_val)
        if not is_valid_gradient:
            zero_grad(node_val)
            return False, tensor(0.0)

        first_gradient = gradient.clone().reshape(-1)
        size = first_gradient.shape[0]
        hessian_diag_minus_max = torch.zeros(size)
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
                return False, tensor(0.0)

            hessian_diag_minus_max[i] = second_gradient[i]
            second_gradient[i] = 0
            hessian_diag_minus_max[i] -= second_gradient.max()

        # ensures gradient is zero at the end of each proposals.
        zero_grad(node_val)
        node_val.detach()
        node_val_reshaped = node_val.clone().reshape(-1)
        predicted_alpha = (
            1 - ((node_val_reshaped * node_val_reshaped) * (hessian_diag_minus_max))
        ).reshape(node_val.shape)

        # pyre-fixme
        mean = node_var.distribution.mean

        if isinstance(node_var.distribution, dist.Beta):
            # pyre-fixme
            mean = torch.cat((mean.unsqueeze(-1), (1 - mean).unsqueeze(-1)), -1)

        predicted_alpha = torch.where(
            predicted_alpha < -1 * min_alpha_value, mean, predicted_alpha
        )

        predicted_alpha = torch.where(
            predicted_alpha > 0, predicted_alpha, tensor(min_alpha_value)
        )
        return True, predicted_alpha

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
        is_valid, alpha = self.compute_alpha(node_var, world)
        if not is_valid:
            return super().get_proposal_distribution(node, node_var, world)
        return ProposalDistribution(
            proposal_distribution=dist.Dirichlet(alpha),
            requires_transform=False,
            requires_reshape=True,
        )
