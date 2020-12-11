# Copyright (c) Facebook, Inc. and its affiliates
import logging
from typing import Dict, Tuple

import numpy
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.newtonian_monte_carlo_utils import (
    is_scalar,
    is_valid,
    symmetric_inverse,
    zero_grad,
)
from beanmachine.ppl.inference.proposer.normal_eig import NormalEig
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils import tensorops  # pyre-ignore
from beanmachine.ppl.world import ProposalDistribution, Variable, World
from torch import Tensor, tensor


LOGGER = logging.getLogger("beanmachine")


class SingleSiteRealSpaceNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Half Space Newtonian Monte Carlo Proposers
    """

    def __init__(self, alpha: float = 10.0, beta: float = 1.0):
        self.alpha_ = alpha
        self.beta_ = beta
        self.learning_rate_ = tensor(0.0)
        self.running_mean_, self.running_var_ = tensor(0.0), tensor(0.0)
        self.accepted_samples_ = 0
        super().__init__()

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
            # If any of alpha or beta are scalar, we have to reshape them
            # random variable shape to allow for per-index learning rate.
            if is_scalar(self.alpha_) or is_scalar(self.beta_):
                # pyre-fixme[8]: Attribute has type `float`; used as `Tensor`.
                self.alpha_ = self.alpha_ * torch.ones_like(
                    node_var.transformed_value
                ).reshape(-1)

                # pyre-fixme[8]: Attribute has type `float`; used as `Tensor`.
                self.beta_ = self.beta_ * torch.ones_like(
                    node_var.transformed_value
                ).reshape(-1)

            beta_ = dist.Beta(self.alpha_, self.beta_)
            frac_dist = beta_.sample()
            aux_vars["frac_dist"] = frac_dist
        else:
            frac_dist = auxiliary_variables["frac_dist"]
        self.learning_rate_ = frac_dist.detach()
        number_of_variables = world.get_number_of_variables()
        if node_var.proposal_distribution is not None and number_of_variables == 1:
            _arguments = node_var.proposal_distribution.arguments
            distance = _arguments["distance"]
            node_val_reshaped = _arguments["node_val_reshaped"]
            mean = (node_val_reshaped + distance * frac_dist).squeeze(0)
            if "scale_tril" in _arguments:
                _proposer = dist.MultivariateNormal(
                    mean, scale_tril=_arguments["scale_tril"]
                )
            else:
                (eig_vals, eig_vecs) = _arguments["eig_decomp"]
                _proposer = NormalEig(mean, eig_vals=eig_vals, eig_vecs=eig_vecs)
            return (
                ProposalDistribution(
                    proposal_distribution=_proposer,
                    requires_transform=True,
                    requires_reshape=True,
                    arguments=_arguments,
                ),
                aux_vars,
            )

        node_val = node_var.transformed_value
        score = world.compute_score(node_var)
        zero_grad(node_val)
        # pyre-fixme
        first_gradient, hessian = tensorops.gradients(score, node_val)
        zero_grad(node_val)
        if not is_valid(first_gradient) or not is_valid(hessian):
            LOGGER.warning(
                "Gradient or Hessian is invalid at node {nv}.\n".format(
                    nv=str(node_var)
                )
                + "Node {n} has invalid proposal solution. ".format(n=node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(node, node_var, world, {})
        first_gradient = first_gradient.detach()
        hessian = hessian.detach()
        # node value may of any arbitrary shape, so here, we transform it into a
        # 1D vector using reshape(-1) and with unsqueeze(0), we change 1D vector
        # of size (N) to (1 x N) matrix.
        node_val_reshaped = node_val.reshape(-1).unsqueeze(0).detach()
        neg_hessian = -1 * hessian
        _arguments = {"node_val_reshaped": node_val_reshaped}
        # we will first attempt a covariance-inverse-based proposer
        try:
            # TODO(nazaninkt): Change this back to torch once the issue with
            # cholesky runtime is fixed.
            # pyre-fixme
            L = torch.from_numpy(numpy.linalg.cholesky(neg_hessian.numpy())).to(
                # pyre-fixme
                dtype=neg_hessian.dtype
            )
            # H^{-1} = (L L^T)^{-1}
            #        = L^{-T} L^{-1}
            # so L^{-1} is (lower) cholesky factor of H^{-1}
            L_inv = torch.triangular_solve(
                torch.eye(L.size(-1)).to(dtype=neg_hessian.dtype), L, upper=False
            ).solution.t()
            distance = torch.cholesky_solve(first_gradient.unsqueeze(1), L).t()
            _arguments["distance"] = distance
            mean = (
                node_val_reshaped
                + distance * frac_dist.to(dtype=node_val_reshaped.dtype)
            ).squeeze(0)
            _proposer = dist.MultivariateNormal(mean, scale_tril=L_inv)
            _arguments["scale_tril"] = L_inv
        except numpy.linalg.LinAlgError:
            LOGGER.warning(
                "Error: Cholesky decomposition failed. "
                + "Falls back to Eigen decomposition."
            )
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

        return (
            ProposalDistribution(
                proposal_distribution=_proposer,
                requires_transform=True,
                requires_reshape=True,
                arguments=_arguments,
            ),
            aux_vars,
        )

    def compute_beta_priors_from_accepted_lr(
        self, max_lr_num: int = 5
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Alpha and Beta using Method of Moments.

        :returns: the alpha and beta of the Beta prior to learning rate.
        """
        # Running mean and variance are computed following the link below:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        old_mu = self.running_mean_
        old_var = self.running_var_

        n = self.accepted_samples_
        xn = self.learning_rate_

        new_mu = old_mu + (xn - old_mu) / n
        new_var = old_var + ((xn - old_mu) * (xn - new_mu) - old_var) / n
        self.running_var_ = new_var
        self.running_mean_ = new_mu
        if n < max_lr_num:
            return (
                tensor(1.0, dtype=self.learning_rate_.dtype),
                tensor(1.0, dtype=self.learning_rate_.dtype),
            )
        # alpha and beta are calculated following the link below.
        # https://stats.stackexchange.com/questions/12232/calculating-the-
        # parameters-of-a-beta-distribution-using-the-mean-and-variance
        alpha = ((1.0 - new_mu) / new_var - (1.0 / new_mu)) * (new_mu ** 2)
        beta = alpha * (1.0 - new_mu) / new_mu
        alpha = torch.where(alpha <= 0, torch.ones_like(alpha), alpha)
        beta = torch.where(beta <= 0, torch.ones_like(beta), beta)
        return alpha, beta

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
        Do adaption based on the learning rates.

        :param node: the node for which we have already proposed a new value for.
        :param node_var: the Variable object associated with node.
        :param node_acceptance_results: the boolean values of acceptances for
         values collected so far within _infer().
        :param iteration_number: The current iteration of inference
        :param num_adaptive_samples: The number of inference iterations for adaptation.
        :param is_accepted: bool representing whether the new value was accepted.
        """
        if not is_accepted:
            if self.accepted_samples_ == 0:
                self.alpha_, self.beta_ = (
                    tensor(1.0, dtype=self.learning_rate_.dtype),
                    tensor(1.0, dtype=self.learning_rate_.dtype),
                )
        else:
            self.accepted_samples_ += 1
            # pyre-fixme[8]: Attribute has type `float`; used as `Tensor`.
            # pyre-fixme[8]: Attribute has type `float`; used as `Tensor`.
            self.alpha_, self.beta_ = self.compute_beta_priors_from_accepted_lr()
