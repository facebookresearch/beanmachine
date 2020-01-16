# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World
from torch.distributions.transforms import AffineTransform


class SingleSiteRandomWalkProposer(SingleSiteAncestralProposer):
    """
    Single-Site Random Walk Metropolis Hastings Implementations

    For random variables with continuous support distributions, returns a
    sample with a perturbation added to the current variable.
    For the rest of the random variables, it returns ancestral metropolis hastings
    proposal.
    """

    def __init__(self, step_size: float = 1.0):
        """
        Initialize object.

        :param step_size: Standard Deviation of the noise used for Diff proposals.
        """

        self.step_size = step_size
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
        required to find a proposal distribution
        :returns: the tuple of proposal distribution of the node and arguments
        that was used or needs to be used to find the proposal distribution
        """
        node_distribution = node_var.distribution

        # pyre-fixme
        if isinstance(node_distribution.support, dist.constraints._Real):
            return (
                ProposalDistribution(
                    proposal_distribution=dist.Normal(
                        node_var.value,
                        torch.ones(node_var.value.shape) * self.step_size,
                    ),
                    requires_transform=False,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )
        if isinstance(
            node_distribution.support, dist.constraints._GreaterThan
        ) or isinstance(node_distribution.support, dist.constraints._GreaterThan):
            lower_bound = node_distribution.support.lower_bound
            node_var.set_transform([AffineTransform(loc=lower_bound, scale=1.0)])
            proposal_distribution = self.gamma_distbn_from_moments(
                node_var.value - lower_bound, self.step_size ** 2
            )

            return (
                ProposalDistribution(
                    proposal_distribution=proposal_distribution,
                    requires_transform=True,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )
        if isinstance(node_distribution.support, dist.constraints._Interval):
            lower_bound = node_distribution.support.lower_bound
            width = node_distribution.support.upper_bound - lower_bound
            # Compute first and second moments of the perturbation distribution
            # by rescaling the support
            mu = (node_var.value - lower_bound) / width
            sigma = torch.ones(node_var.value.shape) * self.step_size / width
            proposal_distribution = self.beta_distbn_from_moments(mu, sigma)
            if lower_bound != 0 and width != 1.0:
                node_var.set_transform([AffineTransform(loc=lower_bound, scale=width)])
                requires_transform = True
            else:
                requires_transform = False

            return (
                ProposalDistribution(
                    proposal_distribution=proposal_distribution,
                    requires_transform=requires_transform,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )
        if isinstance(node_distribution.support, dist.constraints._Simplex):
            proposal_distribution = self.dirichlet_distbn_from_moments(
                node_var.value, self.step_size
            )
            return (
                ProposalDistribution(
                    proposal_distribution=proposal_distribution,
                    requires_transform=False,
                    requires_reshape=False,
                    arguments={},
                ),
                {},
            )
        return super().get_proposal_distribution(
            node, node_var, world, auxiliary_variables
        )

    def gamma_distbn_from_moments(self, expectation, sigma):
        """
        Returns a Gamma distribution.

        :param expectation: expectation value
        :param sigma: sigma value
        :returns: returns the Beta distribution given mu and sigma.
        """
        beta = expectation / (sigma ** 2)
        alpha = expectation * beta
        distribution = dist.Gamma(concentration=alpha, rate=beta)
        return distribution

    def beta_distbn_from_moments(self, mu, sigma):
        """
        Returns a Beta distribution.

        :param mu: mu value
        :param sigma: sigma value
        :returns: returns the Beta distribution given mu and sigma.
        """
        mu = torch.clamp(mu, 1e-3, 1 - 1e-3)
        """
        https://stats.stackexchange.com/questions/12232/calculating-the-
        parameters-of-a-beta-distribution-using-the-mean-and-variance
        """
        alpha = ((1.0 - mu) / (sigma ** 2) - (1.0 / mu)) * (mu ** 2)
        beta = alpha * (1.0 - mu) / mu
        distribution = dist.Beta(concentration1=alpha, concentration0=beta)
        return distribution

    def dirichlet_distbn_from_moments(self, mu, sigma):
        """
        Returns a Dirichlet distribution. The variances of a Dirichlet
        distribution are inversely proportional to the norm of the concentration
        vector

        :param mu: mu value
        :param sigma: sigma value
        :returns: returns the Dirichlet distribution given mu and sigma.
        """
        alpha = mu / (torch.norm(mu) * sigma ** 2)
        return dist.Dirichlet(concentration=alpha)
