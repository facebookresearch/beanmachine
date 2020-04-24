# Copyright (c) Facebook, Inc. and its affiliates
from typing import Dict, Tuple

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
from torch import Tensor


class SingleSiteNewtonianMonteCarloProposer(SingleSiteAncestralProposer):
    """
    Single-Site Newtonian Monte Carlo Implementations

    In this implementation, we draw a new sample from a proposal that is a
    MultivariateNormal with followings specifications with distance being
    sampled from Beta(nmc_alpha, nmc_beta)
        mean = sampled_value - distance * (gradient * hessian_inversed)
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

    def __init__(self, nmc_alpha: float = 10.0, nmc_beta: float = 1.0):
        self.real_space_proposer_ = SingleSiteRealSpaceNewtonianMonteCarloProposer(
            nmc_alpha, nmc_beta
        )
        self.half_space_proposer_ = SingleSiteHalfSpaceNewtonianMonteCarloProposer()
        self.simplex_proposer_ = SingleSiteSimplexNewtonianMonteCarloProposer()
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
        # pyre-fixme
        node_distribution_support = node_var.distribution.support
        if world.get_transform(node) or isinstance(
            node_distribution_support, dist.constraints._Real
        ):
            return self.real_space_proposer_.get_proposal_distribution(
                node, node_var, world, auxiliary_variables
            )
        elif isinstance(node_distribution_support, dist.constraints._GreaterThan):
            return self.half_space_proposer_.get_proposal_distribution(
                node, node_var, world, {}
            )
        elif isinstance(
            node_distribution_support, dist.constraints._Simplex
        ) or isinstance(node_var.distribution, dist.Beta):
            return self.simplex_proposer_.get_proposal_distribution(
                node, node_var, world, {}
            )
        return super().get_proposal_distribution(
            node, node_var, world, auxiliary_variables
        )

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
        proposer = None
        node_var = world.get_node_in_world_raise_error(node, False)
        # pyre-fixme
        node_distribution_support = node_var.distribution.support
        if world.get_transform(node) or isinstance(
            node_distribution_support, dist.constraints._Real
        ):
            proposer = self.real_space_proposer_
        elif isinstance(node_distribution_support, dist.constraints._GreaterThan):
            proposer = self.half_space_proposer_
        elif isinstance(
            node_distribution_support, dist.constraints._Simplex
        ) or isinstance(node_var.distribution, dist.Beta):
            proposer = self.simplex_proposer_

        if proposer is not None:
            # pyre-fixme[16]: `None` has no attribute `do_adaptation`.
            return proposer.do_adaptation(
                node,
                world,
                acceptance_probability,
                iteration_number,
                num_adaptive_samples,
                is_accepted,
            )
        return super().do_adaptation(
            node,
            world,
            acceptance_probability,
            iteration_number,
            num_adaptive_samples,
            is_accepted,
        )
