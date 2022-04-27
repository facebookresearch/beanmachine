# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Type, Callable

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_problem import (
    ancestors_first_graph_fixer,
    node_fixer_first_match,
    GraphFixer,
    GraphFixerResult,
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)
from beanmachine.ppl.compiler.sizer import Sizer

# TODO Move this to a utils module
from beanmachine.ppl.compiler.support import _prod
from torch import Size, tensor


# These graph fixers turn vectorized models into unvectorized models.
# For example, the model
#
# @rv def flip():
#   return Bernoulli(tensor([0.25, 0.75]))
#
# which we cannot represent in BMG is rewritten into the model:
#
# p = tensor([0.25, 0.75])
# @rv def f0:
#   return Bernoulli(p[0])
# @rv def f1:
#   return Bernoulli(p[1])
# @functional def flip():
#   return tensor([f0()), f1())])
#
# which we can represent in BMG.
#
# TODO: Consider optimizing distributions where the tensor elements are all
# the same; if we have Bernoulli([[0.5, 0.5], [0.5, 0.5]]) then that can be represented in
# BMG as an IID_SAMPLE(2,2) from Bernoulli(0.5). We could write another fixer
# which makes this transformation, or we could modify this fixer.


def _is_fixable_size(s: Size) -> bool:
    dim = len(s)
    if dim == 1:
        return s[0] > 1
    if dim == 2:
        return s[0] > 1 or s[1] > 1
    return False


def _node_to_index_list(
    bmg: BMGraphBuilder, sizer: Sizer, n: bn.BMGNode
) -> List[bn.BMGNode]:
    size = sizer[n]
    dim = len(size)
    index_list = []
    if dim == 0:
        # If we have just a single value then there's no indexing required.
        index_list.append(n)
    elif dim == 1:
        for i in range(0, size[0]):
            ci = bmg.add_constant(i)
            ni = bmg.add_index(n, ci)
            index_list.append(ni)
    else:
        assert dim == 2
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                ci = bmg.add_constant(i)
                cj = bmg.add_constant(j)
                ni = bmg.add_index(n, ci)
                nij = bmg.add_index(ni, cj)
                index_list.append(nij)
    return index_list


def _generate_arglists(bmg: BMGraphBuilder, sizer: Sizer, node: bn.BMGNode):
    # This code is a bit tricky to understand so lets work an example.
    # Suppose node has two inputs, call them X and Y. X has size [3], Y has
    # size [2, 3], and node has size [2, 3].
    final_size = sizer[node]  # Size([2, 3])
    final_length = _prod(final_size)  # 2 x 3 = 6
    input_nodes = [_node_to_index_list(bmg, sizer, n) for n in node.inputs]
    # input_nodes is [
    #   [ Index(X, 0), Index(X, 1), Index(X, 2)],
    #   [ Index(Index(Y, 0), 0), Index(Index(Y, 0), 1), ...]
    # ]
    index_lists = []
    # Let's now look at what happens on the FIRST loop iteration:
    for i in range(len(input_nodes)):
        input_node = input_nodes[i]
        # First time through the loop input_node is [Index(X, 0), Index(X, 1), Index(X, 2)]
        input_length = len(input_node)  # 3
        input_size = sizer[node.inputs[i]]  # Size([3])
        t = (
            tensor(range(input_length))  # tensor([0, 1, 2])
            .reshape(input_size)  # tensor([0, 1, 2])
            .broadcast_to(final_size)  # tensor([[0, 1, 2], [0, 1, 2]])
            .reshape(final_length)  # tensor([0, 1, 2, 0, 1, 2])
            .tolist()  # [0, 1, 2, 0, 1, 2]
        )
        index_lists.append(t)

    # When we're done both iterations we have two lists of the same length:
    # [0, 1, 2, 0, 1, 2]
    # [0, 1, 2, 3, 4, 5]
    #
    # Now make tuples out of each column.
    #
    # [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 5)]
    index_tuples = list(zip(*index_lists))
    # These pairs give the elements of X and Y needed to build devectorized nodes.

    # Now make actual argument lists for each tuple.
    return [
        [input_nodes[i][index_tuple[i]] for i in range(len(index_tuple))]
        for index_tuple in index_tuples
    ]


def _distribution_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    # These are all the distributions that we know how to devectorize,
    # and the factory methods we need to use to generate a new node
    # of the appropriate type.

    # TODO: categorical
    # TODO: categorical logit
    # TODO: dirichlet
    return {
        bn.BernoulliLogitNode: bmg.add_bernoulli_logit,
        bn.BernoulliNode: bmg.add_bernoulli,
        bn.BetaNode: bmg.add_beta,
        bn.BinomialNode: bmg.add_binomial,
        bn.BinomialLogitNode: bmg.add_binomial_logit,
        bn.Chi2Node: bmg.add_chi2,
        bn.GammaNode: bmg.add_gamma,
        bn.HalfCauchyNode: bmg.add_halfcauchy,
        bn.HalfNormalNode: bmg.add_halfnormal,
        bn.NormalNode: bmg.add_normal,
        bn.PoissonNode: bmg.add_poisson,
        bn.StudentTNode: bmg.add_studentt,
        bn.UniformNode: bmg.add_uniform,
    }


_distribution_types = list(_distribution_factories(BMGraphBuilder()).keys())


def _is_fixable_sample(sizer: Sizer, n: bn.BMGNode) -> bool:
    if not isinstance(n, bn.SampleNode):
        return False
    dist = n.operand
    if type(dist) not in _distribution_types:
        return False
    return _is_fixable_size(sizer[dist])


def _vectorized_distribution_node_fixer(bmg: BMGraphBuilder, sizer: Sizer) -> NodeFixer:
    distribution_factories = _distribution_factories(bmg)

    def fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not _is_fixable_sample(sizer, node):
            return Inapplicable
        assert isinstance(node, bn.SampleNode)
        dist = node.operand
        arglists = _generate_arglists(bmg, sizer, dist)
        samples = []
        for arglist in arglists:
            b = distribution_factories[type(dist)](*arglist)
            s = bmg.add_sample(b)
            samples.append(s)
        size = sizer[dist]
        t = bmg.add_tensor(size, *samples)
        return t

    return fixer


def _operator_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    return {
        # TODO: Do addition and multiply need to be the multi- versions?
        bn.AdditionNode: bmg.add_addition,
        bn.DivisionNode: bmg.add_division,
        bn.ExpNode: bmg.add_exp,
        bn.LogisticNode: bmg.add_logistic,
        bn.LogNode: bmg.add_log,
        bn.MultiplicationNode: bmg.add_multiplication,
        bn.NegateNode: bmg.add_negate,
        bn.PhiNode: bmg.add_phi,
        bn.PowerNode: bmg.add_power,
    }
    # TODO soon: LogSumExp, ExpM1, Logm1exp
    # TODO later: all comparisons, all bitwise, floordiv,
    # shifts, mod, not, invert,


def _vectorized_operator_node_fixer(bmg: BMGraphBuilder, sizer: Sizer) -> NodeFixer:

    operator_factories = _operator_factories(bmg)

    def node_fixer(n: bn.BMGNode) -> NodeFixerResult:
        if type(n) not in operator_factories:
            return Inapplicable
        # We do not rewrite multiplications of matrices by scalars; that's
        # handled in a later rewriter.
        if (
            isinstance(n, bn.MultiplicationNode)
            and len(n.inputs) == 2
            and (len(sizer[n.inputs[0]]) <= 1 or len(sizer[n.inputs[1]]) <= 1)
        ):
            return Inapplicable
        if not _is_fixable_size(sizer[n]):
            return Inapplicable
        assert isinstance(n, bn.OperatorNode)
        arglists = _generate_arglists(bmg, sizer, n)
        results = []
        for arglist in arglists:
            r = operator_factories[type(n)](*arglist)
            results.append(r)
        size = sizer[n]
        t = bmg.add_tensor(size, *results)
        return t

    return node_fixer


def vectorized_operator_fixer(bmg: BMGraphBuilder) -> GraphFixer:
    def fixer() -> GraphFixerResult:
        sizer = Sizer()

        dist_fixer = _vectorized_distribution_node_fixer(bmg, sizer)
        oper_fixer = _vectorized_operator_node_fixer(bmg, sizer)
        node_fixer = node_fixer_first_match([dist_fixer, oper_fixer])
        vof = ancestors_first_graph_fixer(bmg, sizer, node_fixer)
        made_progress, errors = vof()

        # If we changed something then we might have a leaf sample node;
        # we can remove it.
        if made_progress:
            for n in bmg.all_nodes():
                if _is_fixable_sample(sizer, n):
                    assert n.is_leaf
                    bmg.remove_leaf(n)
        return made_progress, errors

    return fixer


def vectorized_observation_fixer(bmg: BMGraphBuilder) -> GraphFixer:
    def fixer() -> GraphFixerResult:
        made_change = False
        # We might have an illegal observation. Fix it.
        for o in bmg.all_observations():
            observed = o.observed
            if not isinstance(observed, bn.TensorNode):
                continue
            if not _is_fixable_size(observed._size):
                continue
            # TODO: What if the observation is of a different size than the
            # tensor node we've just generated? That should be an error, but instead
            # we just crash here. Figure out where to put an error detection pass
            # which prevents this crash and reports the error.
            dim = len(observed._size)
            if dim == 1:
                for i in range(0, observed._size[0]):
                    s = observed.inputs[i]
                    assert isinstance(s, bn.SampleNode)
                    bmg.add_observation(s, o.value[i])
            else:
                assert dim == 2
                for i in range(0, observed._size[0]):
                    for j in range(0, observed._size[1]):
                        s = observed.inputs[i * observed._size[1] + j]
                        assert isinstance(s, bn.SampleNode)
                        bmg.add_observation(s, o.value[i][j])
            bmg.remove_leaf(o)
            made_change = True
        return made_change, ErrorReport()

    return fixer
