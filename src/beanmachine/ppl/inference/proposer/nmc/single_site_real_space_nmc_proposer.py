# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple, NamedTuple, Union, Optional

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils import (
    is_scalar,
    is_valid,
    soft_abs_inverse,
    hessian_of_log_prob,
)
from beanmachine.ppl.legacy.inference.proposer.normal_eig import NormalEig
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils import tensorops
from beanmachine.ppl.world import World


LOGGER = logging.getLogger("beanmachine")


class _ProposalArgs(NamedTuple):
    distance: torch.Tensor
    node_val_reshaped: torch.Tensor
    scale_tril: Optional[torch.Tensor] = None
    eig_vals: Optional[torch.Tensor] = None
    eig_vecs: Optional[torch.Tensor] = None


class SingleSiteRealSpaceNMCProposer(SingleSiteAncestralProposer):
    """
    Single-Site Real Space Newtonian Monte Carlo Proposer
    See sec. 3.1 of [1]

    [1] Arora, Nim, et al. `Newtonian Monte Carlo: single-site MCMC meets second-order gradient methods`
    """

    def __init__(self, node: RVIdentifier, alpha: float = 10.0, beta: float = 1.0):
        super().__init__(node)
        self.alpha_: Union[float, torch.Tensor] = alpha
        self.beta_: Union[float, torch.Tensor] = beta
        self.learning_rate_ = torch.tensor(0.0)
        self.running_mean_, self.running_var_ = torch.tensor(0.0), torch.tensor(0.0)
        self.accepted_samples_ = 0
        # cached proposal args
        self._proposal_args: Optional[_ProposalArgs] = None

    def _sample_frac_dist(self, world: World) -> torch.Tensor:
        node_val_flatten = world[self.node].flatten()
        # If any of alpha or beta are scalar, we have to reshape them
        # random variable shape to allow for per-index learning rate.
        if is_scalar(self.alpha_) or is_scalar(self.beta_):
            self.alpha_ = self.alpha_ * torch.ones_like(node_val_flatten)
            self.beta_ = self.beta_ * torch.ones_like(node_val_flatten)
        beta_ = dist.Beta(self.alpha_, self.beta_)
        return beta_.sample()

    def _get_proposal_distribution_from_args(
        self, world: World, frac_dist: torch.Tensor, args: _ProposalArgs
    ) -> dist.Distribution:
        node_val = world[self.node]
        mean = (args.node_val_reshaped + args.distance * frac_dist).squeeze(0)
        if args.scale_tril is not None:
            proposal_dist = dist.MultivariateNormal(mean, scale_tril=args.scale_tril)
        else:
            assert args.eig_vals is not None and args.eig_vecs is not None
            proposal_dist = NormalEig(
                mean, eig_vals=args.eig_vals, eig_vecs=args.eig_vecs
            )
        # reshape to match the original sample shape
        reshape_transform = dist.ReshapeTransform(
            node_val.flatten().size(), node_val.size()
        )
        return dist.TransformedDistribution(proposal_dist, reshape_transform)

    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        """
        Returns the proposal distribution of the node.

        Args:
            world: the world in which we're proposing a new value for node
                required to find a proposal distribution which in this case is the
                fraction of distance between the current value and NMC mean that we're
                going to pick as our proposer mean.
        Returns:
            The proposal distribution.
        """
        frac_dist = self._sample_frac_dist(world)
        self.learning_rate_ = frac_dist

        if self._proposal_args is not None and len(world.latent_nodes) == 1:
            return self._get_proposal_distribution_from_args(
                world, frac_dist, self._proposal_args
            )

        node_var = world.get_variable(self.node)
        node_val = node_var.value
        node_device = node_val.device
        first_gradient, hessian = hessian_of_log_prob(
            world, self.node, node_val, tensorops.gradients
        )
        if not is_valid(first_gradient) or not is_valid(hessian):
            LOGGER.warning(
                "Gradient or Hessian is invalid at node {nv}.\n".format(
                    nv=str(node_var)
                )
                + "Node {n} has invalid proposal solution. ".format(n=self.node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(world)

        # node value may of any arbitrary shape, so here, we use reshape to convert a
        # 1D vector of size (N) to (1 x N) matrix.
        node_val_reshaped = node_val.reshape(1, -1)
        neg_hessian = -1 * hessian
        # we will first attempt a covariance-inverse-based proposer

        # Using chelesky_ex because error propagation is slow in PyTorch (see N1136967)
        L, info = torch.linalg.cholesky_ex(neg_hessian.flip([0, 1]), check_errors=False)
        if info == 0:  # info > 0 means the matrix isn't positive-definite
            # See: https://math.stackexchange.com/questions/1434899/is-there-a-decomposition-u-ut
            # Let, flip(H) = L @ L' (`flip` flips the x, y axes of X: torch.flip(X, (0, 1)))
            # equiv. to applying W @ X @ W'; where W is the permutation matrix
            # [[0 ... 1], [0 ... 1 0], ..., [1 ... 0]]
            # Note: flip(A @ B) = flip(A) @ flip(B) and flip(A^-1) = (flip(A))^-1

            # (flip(H))^-1 = (L @ L')^-1 = L'^-1 @  L^-1
            # flip(H^-1) = (L^-1)' @ (L^-1)
            # Note that L^-1 is lower triangular and isn't the cholesky factor for (flip(H))^-1.
            # flip(flip(H^-1)) = flip((L^-1)') @ flip(L^-1)
            # H^-1 = flip(L^-1)' @ flip(L^-1)
            # flip(L^-1)' is the lower triangular cholesky factor for H^-1.
            L_inv = torch.triangular_solve(
                torch.eye(L.size(-1)).to(dtype=neg_hessian.dtype, device=node_device),
                L,
                upper=False,
            ).solution
            L_chol = L_inv.flip([0, 1]).T
            distance = torch.cholesky_solve(first_gradient.unsqueeze(1), L).t()
            proposal_args = _ProposalArgs(
                distance=distance,
                node_val_reshaped=node_val_reshaped,
                scale_tril=L_chol,
            )
        else:
            LOGGER.warning(
                "Error: Cholesky decomposition failed. "
                + "Falls back to Eigen decomposition."
            )
            eig_vecs, eig_vals = soft_abs_inverse(neg_hessian)
            distance = (
                eig_vecs
                @ (torch.eye(len(eig_vals)) * eig_vals)
                @ (eig_vecs.t() @ first_gradient.unsqueeze(1))
            ).t()
            proposal_args = _ProposalArgs(
                distance=distance,
                node_val_reshaped=node_val_reshaped,
                eig_vals=eig_vals,
                eig_vecs=eig_vecs,
            )
        self._proposal_args = proposal_args
        return self._get_proposal_distribution_from_args(
            world, frac_dist, proposal_args
        )

    def compute_beta_priors_from_accepted_lr(
        self, max_lr_num: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Alpha and Beta using Method of Moments.
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
                torch.tensor(1.0, dtype=self.learning_rate_.dtype),
                torch.tensor(1.0, dtype=self.learning_rate_.dtype),
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
        world: World,
        accept_log_prob: torch.Tensor,
        is_accepted: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Do adaption based on the learning rates.

        Args:
            world: the world in which we're operating in.
            accept_log_prob: Current accepted log prob (Not used in this particular proposer).
            is_accepted: bool representing whether the new value was accepted.
        """
        if not is_accepted:
            if self.accepted_samples_ == 0:
                self.alpha_ = 1.0
                self.beta_ = 1.0
        else:
            self.accepted_samples_ += 1
            self.alpha_, self.beta_ = self.compute_beta_priors_from_accepted_lr()
