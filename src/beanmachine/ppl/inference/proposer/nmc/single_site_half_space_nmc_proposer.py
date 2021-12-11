# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.inference.proposer.newtonian_monte_carlo_utils import (
    is_valid,
    hessian_of_log_prob,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils import tensorops
from beanmachine.ppl.world import World


LOGGER = logging.getLogger("beanmachine")


class SingleSiteHalfSpaceNMCProposer(SingleSiteAncestralProposer):
    """
    Single-Site Half Space Newtonian Monte Carlo Proposers.
    See sec. 3.2 of [1]

    [1] Arora, Nim, et al. `Newtonian Monte Carlo: single-site MCMC meets second-order gradient methods`
    """

    def __init__(self, node: RVIdentifier):
        super().__init__(node)
        self._proposal_distribution = None

    def compute_alpha_beta(
        self, world: World
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        """
        Computes alpha and beta of the Gamma proposal given the node.
            alpha = 1 - hessian_diag * x^2
            beta = -1 * x * hessian_diag - first_grad
        """
        node_val = world[self.node]
        first_gradient, hessian_diag = hessian_of_log_prob(
            world, self.node, node_val, tensorops.halfspace_gradients
        )
        if not is_valid(first_gradient) or not is_valid(hessian_diag):
            LOGGER.warning(
                "Gradient or Hessian is invalid at node {n}.\n".format(n=str(self.node))
            )
            return False, torch.tensor(0.0), torch.tensor(0.0)

        node_val_reshaped = node_val.reshape(-1)
        predicted_alpha = (
            1 - hessian_diag * (node_val_reshaped * node_val_reshaped)
        ).t()
        predicted_beta = -1 * node_val_reshaped * hessian_diag - first_gradient
        condition = (predicted_alpha > 0) & (predicted_beta > 0)
        predicted_alpha = torch.where(
            condition, predicted_alpha, torch.tensor(1.0).to(dtype=predicted_beta.dtype)
        )
        node_var = world.get_variable(self.node)
        mean = (
            node_var.distribution.mean.reshape(-1)
            if is_valid(node_var.distribution.mean)
            else torch.ones_like(predicted_beta)
        )
        predicted_beta = torch.where(condition, predicted_beta, mean)
        predicted_alpha = predicted_alpha.reshape(node_val.shape)
        predicted_beta = predicted_beta.reshape(node_val.shape)
        return True, predicted_alpha, predicted_beta

    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        """
        Returns the proposal distribution of the node.

        Args:
            world: the world in which we're proposing a new value for node.
        Returns:
            The proposal distribution.
        """
        # if the number of variables in the world is 1 and proposal distribution
        # has already been computed, we can use the old proposal distribution
        # and skip re-computing the gradient, since there are no other variable
        # in the world that may change the gradient and the old one is still
        # correct.
        if self._proposal_distribution is not None and len(world.latent_nodes) == 1:
            return self._proposal_distribution

        is_valid, alpha, beta = self.compute_alpha_beta(world)
        if not is_valid:
            LOGGER.warning(
                "Node {n} has invalid proposal solution. ".format(n=self.node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(world)

        self._proposal_distribution = dist.Gamma(alpha, beta)
        return self._proposal_distribution
