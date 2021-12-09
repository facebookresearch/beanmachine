# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.world import World
from beanmachine.ppl.world.utils import is_constraint_eq


class SingleSiteRandomWalkProposer(SingleSiteAncestralProposer):
    def __init__(
        self,
        node,
        step_size: float,
    ):
        self.step_size = step_size
        self.target_acc_rate = {False: torch.tensor(0.44), True: torch.tensor(0.234)}
        self._iter = 0
        super().__init__(node)

    def do_adaptation(self, world, accept_log_prob, *args, **kwargs) -> None:
        if torch.isnan(accept_log_prob):
            return
        accept_prob = accept_log_prob.exp()
        if world[self.node].shape[0] == 1:
            target_acc_rate = self.target_acc_rate[False]
            c = torch.reciprocal(target_acc_rate)
        else:
            target_acc_rate = self.target_acc_rate[True]
            c = torch.reciprocal(1.0 - target_acc_rate)

        new_step_size = self.step_size * torch.exp(
            (accept_prob - target_acc_rate) * c / (self._iter + 1.0)
        )
        self._iter += 1

        self.step_size = new_step_size.item()

    def get_proposal_distribution(self, world: World) -> dist.Distribution:
        """Propose a new value for self.node using the prior distribution."""
        node = world.get_variable(self.node)
        node_support = node.distribution.support  # pyre-ignore [16]

        if is_constraint_eq(node_support, dist.constraints.real):
            return dist.Normal(node.value, self.step_size)
        elif is_constraint_eq(node_support, dist.constraints.greater_than):
            lower_bound = node_support.lower_bound
            proposal_distribution = self.gamma_dist_from_moments(
                node.value - lower_bound, self.step_size ** 2
            )
            transform = dist.AffineTransform(loc=lower_bound, scale=1.0)
            transformed_proposal = dist.TransformedDistribution(
                proposal_distribution, transform
            )
            return transformed_proposal
        elif is_constraint_eq(node_support, dist.constraints.interval):
            lower_bound = node_support.lower_bound
            width = node_support.upper_bound - lower_bound
            mu = (node.value - lower_bound) / width
            sigma = (
                torch.ones(node.value.shape, device=node.value.device)
                * self.step_size
                / width
            )
            proposal_distribution = self.beta_dist_from_moments(mu, sigma)
            transform = dist.AffineTransform(loc=lower_bound, scale=width)
            transformed_proposal = dist.TransformedDistribution(
                proposal_distribution, transform
            )
            return transformed_proposal
        elif is_constraint_eq(node_support, dist.constraints.simplex):
            proposal_distribution = self.dirichlet_dist_from_moments(
                node.value, self.step_size
            )
            return proposal_distribution
        else:
            # default to ancestral
            return super().get_proposal_distribution(world)

    def gamma_dist_from_moments(self, expectation, sigma):
        """
        Returns a Gamma distribution.

        :param expectation: expectation value
        :param sigma: sigma value
        :returns: returns the Beta distribution given mu and sigma.
        """
        beta = expectation / (sigma ** 2)
        beta = torch.clamp(beta, min=1e-3)
        alpha = expectation * beta
        alpha = torch.clamp(alpha, min=1e-3)
        distribution = dist.Gamma(concentration=alpha, rate=beta)
        return distribution

    def beta_dist_from_moments(self, mu, sigma):
        """
        Returns a Beta distribution.

        :param mu: mu value
        :param sigma: sigma value
        :returns: returns the Beta distribution given mu and sigma.
        """
        mu = torch.clamp(mu, 1e-3, 1 - 1e-3)
        sigma = torch.clamp(sigma, 1e-3, (mu * (1 - mu)).min().item())
        """
        https://stats.stackexchange.com/questions/12232/calculating-the-
        parameters-of-a-beta-distribution-using-the-mean-and-variance
        """
        alpha = ((1.0 - mu) / (sigma ** 2) - (1.0 / mu)) * (mu ** 2)
        beta = alpha * (1.0 / mu - 1.0)
        distribution = dist.Beta(concentration1=alpha, concentration0=beta)
        return distribution

    def dirichlet_dist_from_moments(self, mu, sigma):
        """
        Returns a Dirichlet distribution. The variances of a Dirichlet
        distribution are inversely proportional to the norm of the concentration
        vector. However, variance is only set as a scalar, not as a vector.
        So the individual variances of the Dirichlet are not tuned, only the
        magnitude of the entire vector.

        :param mu: mu value
        :param sigma: sigma value
        :returns: returns the Dirichlet distribution given mu and sigma.
        """
        alpha = mu / (torch.norm(mu) * sigma ** 2)
        return dist.Dirichlet(concentration=alpha)
