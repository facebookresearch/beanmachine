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


class SingleSiteSimplexSpaceNMCProposer(SingleSiteAncestralProposer):
    """
    Single-Site Simplex Newtonian Monte Carlo Proposer
    See sec. 3.2 of [1]

    [1] Arora, Nim, et al. `Newtonian Monte Carlo: single-site MCMC meets second-order gradient methods`
    """

    def __init__(
        self, node: RVIdentifier, transform: dist.Transform = dist.identity_transform
    ):
        super().__init__(node)
        self._transform = transform
        self._proposal_distribution = None

    def compute_alpha(
        self, world: World, min_alpha_value: float = 1e-3
    ) -> Tuple[bool, torch.Tensor]:
        """
        Computes alpha of the Dirichlet proposal given the node.
            alpha = 1 - (x^2) (hessian[i, i] - max(hessian[i]))
                where max(hessian[i]) is maximum of the hessian at ith row
                excluding the diagonal values.

        :param node_var: the node Variable we're proposing a new value for
        :returns: alpha of the Dirichlet distribution as proposal distribution
        """
        node_val = self._transform(world[self.node])
        first_gradient, hessian_diag_minus_max = hessian_of_log_prob(
            world, self.node, node_val, tensorops.simplex_gradients, self._transform
        )
        if not is_valid(first_gradient) or not is_valid(hessian_diag_minus_max):
            LOGGER.warning(
                "Gradient or Hessian is invalid at node {n}.\n".format(n=str(self.node))
            )
            return False, torch.tensor(0.0)

        node_val_reshaped = node_val.reshape(-1)
        predicted_alpha = (
            1 - ((node_val_reshaped * node_val_reshaped) * (hessian_diag_minus_max))
        ).reshape(node_val.shape)

        mean = world.get_variable(self.node).distribution.mean

        predicted_alpha = torch.where(
            predicted_alpha < -1 * min_alpha_value, mean, predicted_alpha
        )

        predicted_alpha = torch.where(
            predicted_alpha > 0, predicted_alpha, torch.tensor(min_alpha_value)
        )
        return True, predicted_alpha

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

        is_valid, alpha = self.compute_alpha(world)
        if not is_valid:
            LOGGER.warning(
                "Node {n} has invalid proposal solution. ".format(n=self.node)
                + "Proposer falls back to SingleSiteAncestralProposer.\n"
            )
            return super().get_proposal_distribution(world)
        self._proposal_distribution = dist.TransformedDistribution(
            dist.Dirichlet(alpha), self._transform.inv
        )
        return self._proposal_distribution
