# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)


def _mu_is_normal_with_real_params(n: bn.SampleNode) -> bool:
    # TODO: For now we support conjugate prior transformation on
    # priors with constant parameter values. We need to modify this for
    # cascading normals e.g. normal(normal(normal...))
    normal_node = n.inputs[0]
    if not isinstance(normal_node, bn.NormalNode):
        return False
    mu = normal_node.inputs[0]
    sigma = normal_node.inputs[1]
    return isinstance(mu, bn.ConstantNode) and isinstance(sigma, bn.ConstantNode)


def _mu_is_queried(n: bn.SampleNode) -> bool:
    # TODO: This check can be removed if it is not a necessary condition.
    return any(isinstance(i, bn.Query) for i in n.outputs.items)


def _sample_contains_obs(n: bn.SampleNode) -> bool:
    return any(isinstance(o, bn.Observation) for o in n.outputs.items)


def _normal_is_observed(n: bn.BMGNode) -> bool:
    return any(_sample_contains_obs(i) for i in n.outputs.items)


def normal_normal_conjugate_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    """This fixer transforms graphs with Normal likelihood with fixed sigma
    and Normal prior for mu. Since this is a conjugate pair, we analytically
    update the prior parameters Normal(mu, sigma) using observations to get
    the posterior parameters Normal(mu', sigma'). Once we update the parameters,
    we delete the observed samples from the graph. This greatly decreases
    the number of nodes, the number of edges in the graph, and the Bayesian
    update is reduced to parameter update which can lead to performance
    wins during inference."""

    def _transform_mu(
        mu: bn.ConstantNode,
        std: bn.ConstantNode,
        sigma: bn.ConstantNode,
        obs: List[bn.Observation],
    ) -> bn.BMGNode:
        precision_prior = pow(std.value, -2.0)
        precision_data = len(obs) * pow(sigma.value, -2.0)
        precision_inv = pow((precision_prior + precision_data), -1.0)
        data_sum = sum(o.value for o in obs)
        transformed_mu = precision_inv * (
            (mu.value * pow(std.value, -2.0)) + (data_sum * pow(sigma.value, -2.0))
        )
        return bmg.add_constant(transformed_mu)

    def _transform_std(
        std: bn.ConstantNode,
        sigma: bn.ConstantNode,
        obs: List[bn.Observation],
    ) -> bn.BMGNode:
        precision_prior = 1 / pow(std.value, 2)
        precision_data = len(obs) / pow(sigma.value, 2)
        transformed_std = math.sqrt(1 / (precision_prior + precision_data))
        return bmg.add_constant(transformed_std)

    def fixer(n: bn.BMGNode) -> NodeFixerResult:
        # A graph is normal-normal conjugate fixable if:
        #
        # There is a Normal node with mu that is sampled
        # from a Normal distribution. Further, the Normal prior
        # is queried and the Normal likelihood has n observations.
        #
        #
        # That is we are looking for stuff like:
        #
        #     mu       std
        #      \       /
        #        Normal
        #          |
        #  sigma Sample
        #    \  /       \
        #   Normal     Query
        #      |
        #    Sample
        #      |            \
        #  Observation 15.9 ...
        #
        #  to turn it into
        #
        #   mu'       std'
        #     \       /
        #      Normal
        #         |
        #       Sample
        #         |
        #       Query

        if not isinstance(n, bn.NormalNode):
            return Inapplicable
        mu_normal_sample = n.inputs[0]
        if not (
            isinstance(mu_normal_sample, bn.SampleNode)
            and _mu_is_normal_with_real_params(mu_normal_sample)
            and _mu_is_queried(mu_normal_sample)
            and _normal_is_observed(n)
        ):
            return Inapplicable

        sigma = n.inputs[1]
        assert isinstance(sigma, bn.UntypedConstantNode)
        mu_normal_node = mu_normal_sample.inputs[0]
        assert isinstance(mu_normal_node, bn.NormalNode)

        obs = []
        samples_to_remove = []
        for o in n.outputs.items:
            if isinstance(o, bn.SampleNode) and _sample_contains_obs(o):
                obs.append(next(iter(o.outputs.items.keys())))
                samples_to_remove.append(o)

        mu = mu_normal_node.inputs[0]
        std = mu_normal_node.inputs[1]
        assert isinstance(mu, bn.ConstantNode)
        assert isinstance(std, bn.ConstantNode)

        transformed_mu = _transform_mu(mu, std, sigma, obs)
        transformed_std = _transform_std(std, sigma, obs)

        mu_normal_node.inputs[0] = transformed_mu
        mu_normal_node.inputs[1] = transformed_std

        # We need to remove both the sample and the observation node.
        for o in obs:
            bmg.remove_leaf(o)

        for s in samples_to_remove:
            if len(s.outputs.items) == 0:
                bmg.remove_node(s)

        return n

    return fixer
