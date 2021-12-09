# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple

import torch.distributions as dist
from beanmachine.ppl.legacy.inference.proposer.single_site_ancestral_proposer import (
    SingleSiteAncestralProposer,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_half_space_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_real_space_newtonian_monte_carlo_proposer import (
    SingleSiteRealSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.inference.proposer.single_site_simplex_newtonian_monte_carlo_proposer import (
    SingleSiteSimplexNewtonianMonteCarloProposer,
)
from beanmachine.ppl.legacy.world import (
    ProposalDistribution,
    Variable,
    World,
    TransformType,
)
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.world.utils import is_constraint_eq
from torch import Tensor


LOGGER = logging.getLogger("beanmachine")


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

    def __init__(
        self,
        nmc_alpha: float = 10.0,
        nmc_beta: float = 1.0,
        transform_type: TransformType = TransformType.DEFAULT,
        transforms: Optional[List] = None,
    ):
        self.proposers_ = {}
        self.alpha_ = nmc_alpha
        self.beta_ = nmc_beta
        super().__init__(transform_type, transforms)
        self.reshape_untransformed_beta_to_dirichlet = True

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
        if node not in self.proposers_:
            # pyre-fixme
            node_distribution_support = node_var.distribution.support
            if world.get_transforms_for_node(
                node
            ).transform_type != TransformType.NONE or is_constraint_eq(
                node_distribution_support, dist.constraints.real
            ):
                self.proposers_[node] = SingleSiteRealSpaceNewtonianMonteCarloProposer(
                    self.alpha_, self.beta_
                )

            elif is_constraint_eq(
                node_distribution_support, dist.constraints.greater_than
            ):
                self.proposers_[node] = SingleSiteHalfSpaceNewtonianMonteCarloProposer()

            elif (
                is_constraint_eq(node_distribution_support, dist.constraints.simplex)
                or isinstance(node_var.distribution, dist.Beta)
                or (
                    isinstance(node_distribution_support, dist.constraints.independent)
                    and (
                        node_distribution_support.base_constraint
                        == dist.constraints.unit_interval
                    )
                )
            ):
                self.proposers_[node] = SingleSiteSimplexNewtonianMonteCarloProposer()
            else:
                LOGGER.warning(
                    "Node {n} has unsupported constraints. ".format(n=node)
                    + "Proposer falls back to SingleSiteAncestralProposer.\n"
                )
                self.proposers_[node] = super()
        return self.proposers_[node].get_proposal_distribution(
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
        if node not in self.proposers_:
            return super().do_adaptation(
                node,
                world,
                acceptance_probability,
                iteration_number,
                num_adaptive_samples,
                is_accepted,
            )

        return self.proposers_[node].do_adaptation(
            node,
            world,
            acceptance_probability,
            iteration_number,
            num_adaptive_samples,
            is_accepted,
        )
