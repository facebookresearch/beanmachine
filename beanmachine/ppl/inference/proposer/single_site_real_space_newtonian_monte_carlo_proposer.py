# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    is_valid,
    symmetric_inverse,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.normal_eig import NormalEig
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.utils import tensorops
from beanmachine.ppl.world import ProposalDistribution, Variable, World


class SingleSiteRealSpaceNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Half Space Newtonian Monte Carlo Proposers
    """

    def __init__(self, alpha: float = 10.0, beta: float = 1.0):
        self.alpha_ = alpha
        self.beta_ = beta

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
            _arguments = node_var.proposal_distribution.arguments
            distance = _arguments["distance"]
            node_val_reshaped = _arguments["node_val_reshaped"]
            mean = (node_val_reshaped + distance * frac_dist).squeeze(0)
            if "covar" in _arguments:
                _proposer = dist.MultivariateNormal(mean, _arguments["covar"])
            else:
                (eig_vals, eig_vecs) = _arguments["eig_decomp"]
                _proposer = NormalEig(mean, eig_vecs, eig_vecs)
            return (
                ProposalDistribution(
                    proposal_distribution=_proposer,
                    requires_transform=node_var.proposal_distribution.requires_transform,
                    requires_reshape=True,
                    arguments=_arguments,
                ),
                aux_vars,
            )

        node_val = node_var.unconstrained_value
        score = world.compute_score(node_var)
        zero_grad(node_val)
        # pyre-fixme
        first_gradient, hessian = tensorops.gradients(score, node_val)
        zero_grad(node_val)
        node_val.detach()
        if not is_valid(first_gradient) or not is_valid(hessian):
            return super().get_proposal_distribution(node, node_var, world, {})
        # node value may of any arbitrary shape, so here, we transform it into a
        # 1D vector using reshape(-1) and with unsqueeze(0), we change 1D vector
        # of size (N) to (1 x N) matrix.
        node_val_reshaped = node_val.reshape(-1).unsqueeze(0)
        neg_hessian = -1 * hessian.detach()
        _arguments = {"node_val_reshaped": node_val_reshaped}
        # we will first attempt a covariance-inverse-based proposer
        try:
            # pyre-fixme
            covar = neg_hessian.inverse()
            distance = (covar @ first_gradient.unsqueeze(1)).T
            _arguments["distance"] = distance
            mean = (node_val_reshaped + distance * frac_dist).squeeze(0)
            _proposer = dist.MultivariateNormal(mean, covar)
            _arguments["covar"] = covar
        except RuntimeError:
            # pyre-fixme
            eig_vecs, eig_vals = symmetric_inverse(neg_hessian)
            # pyre-fixme
            distance = (
                eig_vecs
                @ (torch.eye(len(eig_vals)) * eig_vals)
                @ (eig_vecs.T @ first_gradient.unsqueeze(1))
            ).T
            _arguments["distance"] = distance
            mean = (node_val_reshaped + distance * frac_dist).squeeze(0)
            _proposer = NormalEig(mean, eig_vals, eig_vecs)
            # pyre-fixme
            _arguments["eig_decomp"] = (eig_vals, eig_vecs)

        requires_transform = False
        # pyre-fixme
        if not isinstance(node_var.distribution.support, dist.constraints._Real):
            requires_transform = True
        return (
            ProposalDistribution(
                proposal_distribution=_proposer,
                requires_transform=requires_transform,
                requires_reshape=True,
                arguments=_arguments,
            ),
            aux_vars,
        )
