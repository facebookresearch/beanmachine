# Copyright (c) Facebook, Inc. and its affiliates
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.inference.proposer.single_site_half_space_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_real_space_newtonian_monte_carlo_proposer import (
    SingleSiteRealSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.inference.proposer.single_site_simplex_newtonian_monte_carlo_proposer import (
    SingleSiteSimplexNewtonianMonteCarloProposer,
)
from beanmachine.ppl.model.utils import RVIdentifier
from beanmachine.ppl.world import ProposalDistribution, Variable, World


class SingleSiteNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Newtonian Monte Carlo Implementations

    In this implementation, we draw a new sample from a proposal that is a
    MultivariateNormal with followings specifications:
        mean = sampled_value - gradient * hessian_inversed
        covariance = - hessian_inversed

    Where the gradient and hessian are computed over the log probability of
    the node being resampled and log probabilities of its immediate children

    To compute, the proposal log update, we go through the following steps:
        1) Draw a new sample from MultivariateNormal(node.mean, node.covariance)
        and compute the log probability of the new draw coming from this
        proposal distribution log(P(X->X'))
        2) Construct the new diff given the new value
        3) Compute new gradient and hessian and hence, new mean and
        covariance and compute the log probability of the old value coming
        from MutlivariateNormal(new mean, new covariance) log(P(X'->X))
        4) Compute the final proposal log update: log(P(X'->X)) - log(P(X->X'))
    """

    def __init__(self):
        self.real_space_proposer_ = SingleSiteRealSpaceNewtonianMonteCarloProposer()
        self.half_space_proposer_ = SingleSiteHalfSpaceNewtonianMonteCarloProposer()
        self.simplex_proposer_ = SingleSiteSimplexNewtonianMonteCarloProposer()
        super().__init__()

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
        # pyre-fixme
        node_distribution_support = node_var.distribution.support
        if world.get_transform(node) or isinstance(
            node_distribution_support, dist.constraints._Real
        ):
            return self.real_space_proposer_.get_proposal_distribution(
                node, node_var, world
            )
        elif isinstance(node_distribution_support, dist.constraints._GreaterThan):
            return self.half_space_proposer_.get_proposal_distribution(
                node, node_var, world
            )
        elif isinstance(
            node_distribution_support, dist.constraints._Simplex
        ) or isinstance(node_var.distribution, dist.Beta):
            return self.simplex_proposer_.get_proposal_distribution(
                node, node_var, world
            )
        return super().get_proposal_distribution(node, node_var, world)
