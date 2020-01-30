# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    compute_first_gradient,
    compute_neg_hessian_invserse_eigvals_eigvecs,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.normal_eig import NormalEig
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

    def __init__(self, alpha: float = 10.0, beta: float = 1.0):
        self.alpha_ = alpha
        self.beta_ = beta

    def compute_normal_mean_covar(
        self, node_var: Variable, world: World
    ) -> Tuple[bool, Tensor, Tensor, Tensor, Tensor]:
        """
        Computes mean and covariance of the MultivariateNormal given the node.
            mean = x - first_grad * hessian_inverse
            covariance = -1 * hessian_inverse

        :param node_var: the node Variable we're proposing a new value for
        :returns: is_valid, eigen values and eigen vectors, diff (distance
        between node value and mean) and node value
        """
        node_val = node_var.unconstrained_value
        score = world.compute_score(node_var)
        zero_grad(node_val)
        is_valid_gradient, gradient = compute_first_gradient(score, node_val)

        if not is_valid_gradient:
            zero_grad(node_val)
            return False, tensor(0.0), tensor(0.0), tensor(0.0), tensor(0.0)

        first_gradient = gradient.reshape(-1).clone()
        is_valid_neg_invserse_hessian, eig_vecs, eig_vals = compute_neg_hessian_invserse_eigvals_eigvecs(
            first_gradient, node_val
        )
        zero_grad(node_val)
        node_val.detach()
        if not is_valid_neg_invserse_hessian:
            return False, tensor(0.0), tensor(0.0), tensor(0.0), tensor(0.0)

        # node value may of any arbitrary shape, so here, we transform it into a
        # 1D vector using reshape(-1) and with unsqueeze(0), we change 1D vector
        # of size (N) to (1 x N) matrix.
        node_reshaped = node_val.reshape(-1).unsqueeze(0)
        # here we again call unsqueeze(0) on first_gradient to transform it into
        # a matrix in order to be able to perform matrix multiplication.
        # pyre-fixme
        distance = (
            eig_vecs
            @ (torch.eye(len(eig_vals)) * eig_vals)
            @ (eig_vecs.T @ first_gradient.unsqueeze(0).T)
        ).T
        return True, eig_vals, eig_vecs, distance, node_reshaped

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
        required to find a proposal distribution which in this case is the
        fraction of distance between the current value and NMC mean that we're
        going to pick as our proposer mean.
        :returns: the tuple of proposal distribution of the node and arguments
        that was used or needs to be used to find the proposal distribution
        """
        # if the number of variables in the world is 1 and proposal distribution
        # has already been computed, we can use the old proposal distribution
        # and skip re-computing the gradient, since there are no other variable
        # in the world that may change the gradient and the old one is still
        # correct.
        aux_vars = {}
        if "frac_dist" not in auxiliary_variables:
            beta_ = dist.Beta(tensor(self.alpha_), tensor(self.beta_))
            frac_dist = beta_.sample()
            aux_vars["frac_dist"] = frac_dist
        else:
            frac_dist = auxiliary_variables["frac_dist"]
        number_of_variables = world.get_number_of_variables()
        if node_var.proposal_distribution is not None and number_of_variables == 1:
            distance = node_var.proposal_distribution.arguments["distance"]
            node_val_reshaped = node_var.proposal_distribution.arguments[
                "node_val_reshaped"
            ]
            eig_vals, eig_vecs = (
                # pyre-fixme
                node_var.proposal_distribution.proposal_distribution.eig_decompositions
            )

            mean = (node_val_reshaped + distance * frac_dist).squeeze(0)
            return (
                ProposalDistribution(
                    proposal_distribution=NormalEig(mean, eig_vals, eig_vecs),
                    requires_transform=node_var.proposal_distribution.requires_transform,
                    requires_reshape=True,
                    arguments={
                        "distance": distance,
                        "node_val_reshaped": node_val_reshaped,
                    },
                ),
                aux_vars,
            )

        is_valid, eig_vals, eig_vecs, distance, node_val_reshaped = self.compute_normal_mean_covar(
            node_var, world
        )
        if not is_valid:
            return super().get_proposal_distribution(node, node_var, world, {})

        mean = (node_val_reshaped + distance * frac_dist).squeeze(0)
        requires_transform = False
        # pyre-fixme
        if not isinstance(node_var.distribution.support, dist.constraints._Real):
            requires_transform = True
        return (
            ProposalDistribution(
                proposal_distribution=NormalEig(mean, eig_vals, eig_vecs),
                requires_transform=requires_transform,
                requires_reshape=True,
                arguments={
                    "distance": distance,
                    "node_val_reshaped": node_val_reshaped,
                },
            ),
            aux_vars,
        )
