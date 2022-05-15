# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)


def _theta_is_beta_with_real_params(n: bn.SampleNode) -> bool:
    # TODO: For now we support conjugate prior transformation on
    # priors with constant parameter values.
    beta_node = n.inputs[0]
    if not isinstance(beta_node, bn.BetaNode):
        return False
    alpha = beta_node.inputs[0]
    beta = beta_node.inputs[1]
    return isinstance(alpha, bn.ConstantNode) and isinstance(beta, bn.ConstantNode)


def _theta_is_queried(n: bn.SampleNode) -> bool:
    # TODO: This check can be removed if it is not a necessary condition.
    return any(isinstance(i, bn.Query) for i in n.outputs.items)


def _sample_contains_obs(n: bn.SampleNode) -> bool:
    return any(isinstance(o, bn.Observation) for o in n.outputs.items)


def _get_likelihood_obs_samples(
    n: bn.BMGNode,
) -> Tuple[List[bn.Observation], List[bn.SampleNode]]:
    obs = []
    samples = []
    for o in n.outputs.items:
        if isinstance(o, bn.SampleNode) and _sample_contains_obs(o):
            obs.append(next(iter(o.outputs.items.keys())))
            samples.append(o)
    return obs, samples


def _liklihood_is_observed(n: bn.BMGNode) -> bool:
    return any(_sample_contains_obs(i) for i in n.outputs.items)


def beta_bernoulli_conjugate_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    """This fixer transforms graphs with Bernoulli likelihood and Beta prior.
    Since this is a conjugate pair, we analytically update the prior
    parameters Beta(alpha, beta) using observations to get the posterior
    parameters Beta(alpha', beta'). Once we update the parameters,
    we delete the observed samples from the graph. This greatly decreases
    the number of nodes, the number of edges in the graph, and the Bayesian
    update is reduced to parameter update which can lead to performance
    wins during inference."""

    def fixer(n: bn.BMGNode) -> NodeFixerResult:

        # A graph is beta-bernoulli conjugate fixable if:
        #
        # There is a bernoulli node with theta that is sampled
        # from a beta distribution. Further, the beta is queried and
        # the bernoulli node has n observations.
        #
        # That is we are looking for stuff like:
        #
        #       alpha            beta
        #         \              /
        #               Beta
        #                |
        #              Sample
        #          /                \
        #  Bernoulli             Query
        #      |
        #    Sample
        #      |          \
        #  Observation True ...
        #
        #  to turn it into
        #
        #  alpha'     beta'
        #     \       /
        #        Beta
        #         |
        #       Sample
        #         |
        #       Query

        if not isinstance(n, bn.BernoulliNode):
            return Inapplicable
        beta_sample = n.inputs[0]
        if not (
            isinstance(beta_sample, bn.SampleNode)
            and _theta_is_beta_with_real_params(beta_sample)
            and _theta_is_queried(beta_sample)
            and _liklihood_is_observed(n)
        ):
            return Inapplicable

        beta_node = beta_sample.inputs[0]
        assert isinstance(beta_node, bn.BetaNode)

        obs, samples_to_remove = _get_likelihood_obs_samples(n)

        alpha = beta_node.inputs[0]
        assert isinstance(alpha, bn.ConstantNode)
        obs_sum = sum(o.value for o in obs)
        transformed_alpha = bmg.add_pos_real(alpha.value + obs_sum)

        beta = beta_node.inputs[1]
        assert isinstance(beta, bn.ConstantNode)

        # Update: beta' = beta + n - obs_sum
        transformed_beta = bmg.add_pos_real(beta.value + len(obs) - obs_sum)

        beta_node.inputs[0] = transformed_alpha
        beta_node.inputs[1] = transformed_beta

        # We need to remove both the sample and the observation node.
        for o in obs:
            bmg.remove_leaf(o)

        for s in samples_to_remove:
            if len(s.outputs.items) == 0:
                bmg.remove_node(s)

        return n

    return fixer


def beta_binomial_conjugate_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    """This fixer transforms graphs with Binomial likelihood and Beta prior.
    Since this is a conjugate pair, we analytically update the prior
    parameters Beta(alpha, beta) using observations to get the posterior
    parameters Beta(alpha', beta'). Once we update the parameters,
    we delete the observed samples from the graph. This greatly decreases
    the number of nodes, the number of edges in the graph, and the Bayesian
    update is reduced to parameter update which can lead to performance
    wins during inference."""

    def fixer(n: bn.BMGNode) -> NodeFixerResult:
        # A graph is beta-binomial conjugate fixable if:
        #
        # There is a binomial node with theta that is sampled
        # from a beta distribution. Further, the beta is queried and
        # the binomial node has n observations.
        #
        # That is we are looking for stuff like:
        #
        #      alpha            beta
        #        \              /
        #             Beta
        #               |
        # Count       Sample
        #   \       /       \
        #   Binomial       Query
        #      |
        #    Sample
        #      |          \
        #  Observation 3.0 ...
        #
        #  to turn it into
        #
        #  alpha'     beta'
        #     \       /
        #        Beta
        #         |
        #       Sample
        #         |
        #       Query

        if not isinstance(n, bn.BinomialNode):
            return Inapplicable
        beta_sample = n.inputs[1]

        if not (
            isinstance(beta_sample, bn.SampleNode)
            and _theta_is_beta_with_real_params(beta_sample)
            and _theta_is_queried(beta_sample)
            and _liklihood_is_observed(n)
        ):
            return Inapplicable

        count = n.inputs[0]
        assert isinstance(count, bn.UntypedConstantNode)
        beta_node = beta_sample.inputs[0]
        assert isinstance(beta_node, bn.BetaNode)

        obs, samples_to_remove = _get_likelihood_obs_samples(n)

        alpha = beta_node.inputs[0]
        assert isinstance(alpha, bn.ConstantNode)
        obs_sum = sum(o.value for o in obs)
        transformed_alpha = bmg.add_pos_real(alpha.value + obs_sum)

        # Update: beta' = beta + sum count - obs_sum
        beta = beta_node.inputs[1]
        assert isinstance(beta, bn.ConstantNode)
        updated_count = len(obs) * count.value
        transformed_beta = bmg.add_pos_real(beta.value + updated_count - obs_sum)

        beta_node.inputs[0] = transformed_alpha
        beta_node.inputs[1] = transformed_beta

        # We need to remove both the sample and the observation node.
        for o in obs:
            bmg.remove_leaf(o)

        for s in samples_to_remove:
            if len(s.outputs.items) == 0:
                bmg.remove_node(s)

        return n

    return fixer
