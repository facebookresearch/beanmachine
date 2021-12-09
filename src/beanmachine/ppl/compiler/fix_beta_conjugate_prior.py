# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import List, Optional, Tuple

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.typer_base import TyperBase


class BetaPriorFixer(ProblemFixerBase):
    """Beta distribution is a conjugate prior to Bernoulli, Binomial, Negative
    Binomial and Geometric distributions. Graph pattern check for the Beta prior
    is same for all the conjugate pairs. This fixer checks if the prior satisfies
    required transformation conditions.
    """

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _theta_is_beta_with_real_params(self, n: bn.SampleNode) -> bool:
        # TODO: For now we support conjugate prior transformation on
        # priors with constant parameter values.
        beta_node = n.inputs[0]
        if not isinstance(beta_node, bn.BetaNode):
            return False
        alpha = beta_node.inputs[0]
        beta = beta_node.inputs[1]
        return isinstance(alpha, bn.ConstantNode) and isinstance(beta, bn.ConstantNode)

    def _theta_is_queried(self, n: bn.SampleNode) -> bool:
        # TODO: This check can be removed if it is not a necessary condition.
        return any(isinstance(i, bn.Query) for i in n.outputs.items)

    def _sample_contains_obs(self, n: bn.SampleNode) -> bool:
        return any(isinstance(o, bn.Observation) for o in n.outputs.items)

    def _liklihood_is_observed(self, n: bn.BMGNode) -> bool:
        return any(self._sample_contains_obs(i) for i in n.outputs.items)

    def _get_likelihood_obs_samples(
        self, n: bn.BMGNode
    ) -> Tuple[List[bn.Observation], List[bn.SampleNode]]:
        obs = []
        samples = []
        for o in n.outputs.items:
            if isinstance(o, bn.SampleNode) and self._sample_contains_obs(o):
                obs.append(next(iter(o.outputs.items.keys())))
                samples.append(o)
        return obs, samples

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        if not isinstance(n, bn.SampleNode):
            return False
        return self._theta_is_beta_with_real_params(n) and self._theta_is_queried(n)

    def _transform_alpha(
        self, alpha: bn.ConstantNode, obs: List[bn.Observation]
    ) -> bn.BMGNode:
        # Update: alpha' = alpha + obs_sum
        obs_sum = sum(o.value for o in obs)
        return self._bmg.add_pos_real(alpha.value + obs_sum)

    @abstractmethod
    def _transform_beta(
        self, beta: bn.ConstantNode, obs: List[bn.Observation]
    ) -> bn.BMGNode:
        pass


class BetaBernoulliConjugateFixer(BetaPriorFixer):
    """This fixer transforms graphs with Bernoulli likelihood and Beta prior.
    Since this is a conjugate pair, we analytically update the prior
    parameters Beta(alpha, beta) using observations to get the posterior
    parameters Beta(alpha', beta'). Once we update the parameters,
    we delete the observed samples from the graph. This greatly decreases
    the number of nodes, the number of edges in the graph, and the Bayesian
    update is reduced to parameter update which can lead to performance
    wins during inference."""

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        BetaPriorFixer.__init__(self, bmg, typer)

    def _transform_beta(
        self, beta: bn.ConstantNode, obs: List[bn.Observation]
    ) -> bn.BMGNode:
        # Update: beta' = beta + n - obs_sum
        obs_sum = sum(o.value for o in obs)
        n = len(obs)
        return self._bmg.add_pos_real(beta.value + n - obs_sum)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
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
            return False
        sample = n.inputs[0]
        return super()._needs_fixing(sample) and self._liklihood_is_observed(n)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        assert isinstance(n, bn.BernoulliNode)
        beta_sample = n.inputs[0]
        assert isinstance(beta_sample, bn.SampleNode)
        beta_node = beta_sample.inputs[0]
        assert isinstance(beta_node, bn.BetaNode)

        obs, samples_to_remove = self._get_likelihood_obs_samples(n)

        alpha = beta_node.inputs[0]
        assert isinstance(alpha, bn.ConstantNode)
        transformed_alpha = self._transform_alpha(alpha, obs)

        beta = beta_node.inputs[1]
        assert isinstance(beta, bn.ConstantNode)
        transformed_beta = self._transform_beta(beta, obs)

        beta_node.inputs[0] = transformed_alpha
        beta_node.inputs[1] = transformed_beta

        # We need to remove both the sample and the observation node.
        for o in obs:
            self._bmg.remove_leaf(o)

        for s in samples_to_remove:
            if len(s.outputs.items) == 0:
                self._bmg.remove_node(s)

        return n


class BetaBinomialConjugateFixer(BetaPriorFixer):
    """This fixer transforms graphs with Binomial likelihood and Beta prior.
    Since this is a conjugate pair, we analytically update the prior
    parameters Beta(alpha, beta) using observations to get the posterior
    parameters Beta(alpha', beta'). Once we update the parameters,
    we delete the observed samples from the graph. This greatly decreases
    the number of nodes, the number of edges in the graph, and the Bayesian
    update is reduced to parameter update which can lead to performance
    wins during inference."""

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        BetaPriorFixer.__init__(self, bmg, typer)

    def _transform_beta(
        self, beta: bn.ConstantNode, obs: List[bn.Observation], count: float
    ) -> bn.BMGNode:
        # Update: beta' = beta + sum count - obs_sum
        obs_sum = sum(o.value for o in obs)
        n = len(obs)
        updated_count = n * count
        return self._bmg.add_pos_real(beta.value + updated_count - obs_sum)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
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
            return False
        sample = n.inputs[1]
        return super()._needs_fixing(sample) and self._liklihood_is_observed(n)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        assert isinstance(n, bn.BinomialNode)
        count = n.inputs[0]
        assert isinstance(count, bn.UntypedConstantNode)
        beta_sample = n.inputs[1]
        assert isinstance(beta_sample, bn.SampleNode)
        beta_node = beta_sample.inputs[0]
        assert isinstance(beta_node, bn.BetaNode)

        obs, samples_to_remove = self._get_likelihood_obs_samples(n)

        alpha = beta_node.inputs[0]
        assert isinstance(alpha, bn.ConstantNode)
        transformed_alpha = self._transform_alpha(alpha, obs)

        beta = beta_node.inputs[1]
        assert isinstance(beta, bn.ConstantNode)
        transformed_beta = self._transform_beta(beta, obs, count.value)

        beta_node.inputs[0] = transformed_alpha
        beta_node.inputs[1] = transformed_beta

        # We need to remove both the sample and the observation node.
        for o in obs:
            self._bmg.remove_leaf(o)

        for s in samples_to_remove:
            if len(s.outputs.items) == 0:
                self._bmg.remove_node(s)

        return n
