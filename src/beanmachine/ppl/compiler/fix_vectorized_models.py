# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Type

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_matrix_scale import matrix_scale_fixer
from beanmachine.ppl.compiler.fix_problem import (
    ancestors_first_graph_fixer,
    fixpoint_graph_fixer,
    GraphFixer,
    GraphFixerResult,
    Inapplicable,
    node_fixer_first_match,
    NodeFixer,
    NodeFixerResult,
    sequential_graph_fixer,
)
from beanmachine.ppl.compiler.sizer import is_scalar, Sizer

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


def _is_fixable_size(s: Size) -> bool:
    dim = len(s)
    if dim == 1:
        return s[0] > 1
    if dim == 2:
        return s[0] > 1 or s[1] > 1
    return False


def _is_indexable_node(sizer: Sizer, n: bn.BMGNode) -> bool:
    if type(n) not in _indexable_node_types:
        return False
    return _is_fixable_size(sizer[n])


def _inputs_are_devectorizable(sizer: Sizer, node: bn.BMGNode) -> bool:
    # For a node to be devectorizable:
    # * All its inputs must be either indexable or scalars.
    # * At least one input must be indexable.
    return all(
        _is_indexable_node(sizer, i) or is_scalar(sizer[i]) for i in node.inputs
    ) and any(_is_indexable_node(sizer, i) for i in node.inputs)


def _node_to_index_list(
    bmg: BMGraphBuilder, sizer: Sizer, n: bn.BMGNode
) -> List[bn.BMGNode]:

    size = sizer[n]
    dim = len(size)
    index_list = []
    # This code is a little confusing because BMG uses column-major matrices
    # and torch uses row-major tensors.  The Sizer always gives the size
    # that a graph node would be in *torch*, so if we have a Size([2, 3])
    # matrix node, that has two rows and three columns in torch, and would
    # be indexed first by row and then by column. But in BMG, that would
    # be two columns, three rows, and indexed by column first, then row.
    #
    # The practical upshot is: if we have, say, Size([3]) OR Size([1, 3])
    # then either way, we will have a one-column, three row BMG node, and
    # therefore we only need a single level of indexing.

    if dim == 0:
        # If we have just a single value then there's no indexing required.
        index_list.append(n)
    elif dim == 1:

        for i in range(0, size[0]):
            ci = bmg.add_constant(i)
            ni = bmg.add_index(n, ci)
            index_list.append(ni)
    elif size[0] == 1:
        assert dim == 2
        for i in range(0, size[1]):
            ci = bmg.add_constant(i)
            ni = bmg.add_index(n, ci)
            index_list.append(ni)
    else:
        # We need two levels of indexing.
        assert dim == 2
        for i in range(0, size[0]):
            ci = bmg.add_constant(i)
            ni = bmg.add_index(n, ci)
            for j in range(0, size[1]):
                cj = bmg.add_constant(j)
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
    if not _is_fixable_size(sizer[dist]):
        return False
    # Every input must be either a scalar or indexable,
    # and at least one input must be indexable.
    if not _inputs_are_devectorizable(sizer, dist):
        return False
    return True


_indexable_node_types = [
    bn.ColumnIndexNode,
    bn.ConstantTensorNode,
    bn.IndexNode,
    bn.MatrixMultiplicationNode,
    bn.MatrixScaleNode,
    bn.SampleNode,
    bn.TensorNode,
    bn.ToMatrixNode,
    bn.UntypedConstantNode,
]


def _vectorized_distribution_node_fixer(bmg: BMGraphBuilder, sizer: Sizer) -> NodeFixer:
    distribution_factories = _distribution_factories(bmg)

    def vect_dist_fixer(node: bn.BMGNode) -> NodeFixerResult:
        # The graph transformation we're doing here takes graphs of the form:
        #
        # indexable  -->   dist  -->  sample  -->  consumer
        #
        # where the "indexable" produces a matrix, the consumer takes a matrix,
        # but the distribution requires scalar inputs and produces a scalar
        # output.
        #
        # We transform it into the graph:
        #
        #           --> index[0]  -->  dist  --> sample -->
        # indexable                                        to_matrix  --> consumer
        #           --> index[1]  -->  dist  --> sample  -->
        #           ...
        #
        # And now everyone is happy; the operators get scalars and the
        # consumer gets a matrix.
        #
        #
        # TODO: Consider optimizing distributions where the tensor elements are all
        # the same; if we have Bernoulli([[0.5, 0.5], [0.5, 0.5]]) then that can be
        # represented in BMG as an IID_SAMPLE(2,2) from Bernoulli(0.5). We could
        # write another fixer which makes this transformation, or we could modify
        # this fixer.  NOTE that not all inference algorithms might support
        # IID_SAMPLE nodes; look into this before attempting the optimization.

        if not _is_fixable_sample(sizer, node):
            return Inapplicable
        assert isinstance(node, bn.SampleNode)
        dist = node.operand
        # We need to generate n new distribution and sample nodes, each of
        # which takes some scalar indexed from its inputs. The factory method that
        # builds the distribution is in the distribution factories list.
        # _generate_arglists constructs the arguments to that factory method.
        arglists = _generate_arglists(bmg, sizer, dist)
        samples = []
        factory = distribution_factories[type(dist)]
        for arglist in arglists:
            b = factory(*arglist)
            s = bmg.add_sample(b)
            samples.append(s)
        size = sizer[dist]
        # We now have n new operator nodes; stick them into a tensor.  We then
        # return that tensor. The caller will retarget the input edge of the
        # consumer from the original operator to the tensor, and the graph is
        # rewritten.
        t = bmg.add_tensor(size, *samples)
        return t

    return vect_dist_fixer


def _operator_factories(bmg: BMGraphBuilder) -> Dict[Type, Callable]:
    return {
        # Note that we expect devectorization to run *before* multiary
        # addition/multiplication rewriting, so we can assume that
        # all additions and multiplications are binary.
        bn.AdditionNode: bmg.add_addition,
        bn.DivisionNode: bmg.add_division,
        bn.Exp2Node: bmg.add_exp2,
        bn.ExpNode: bmg.add_exp,
        bn.ExpM1Node: bmg.add_expm1,
        bn.LogisticNode: bmg.add_logistic,
        bn.Log10Node: bmg.add_log10,
        bn.Log1pNode: bmg.add_log1p,
        bn.Log2Node: bmg.add_log2,
        bn.Log1mexpNode: bmg.add_log1mexp,
        bn.LogNode: bmg.add_log,
        bn.MultiplicationNode: bmg.add_multiplication,
        bn.NegateNode: bmg.add_negate,
        bn.PhiNode: bmg.add_phi,
        bn.PowerNode: bmg.add_power,
        bn.SquareRootNode: bmg.add_squareroot,
    }
    # TODO: LogSumExp, all comparisons, all bitwise, floordiv,
    # shifts, mod, invert.  Should we devectorize "not"?


def _vectorized_operator_node_fixer(bmg: BMGraphBuilder, sizer: Sizer) -> NodeFixer:

    operator_factories = _operator_factories(bmg)

    def _is_fixable_operator(sizer: Sizer, operator: bn.BMGNode) -> bool:
        # * The operator must be on the list of devectorizable operators
        #   in operator_factories above.
        # * The sizer must judge that the operator in its current
        #   place in the graph produces a 1-d or 2-d tensor, not a scalar.
        # * Every input must be either a scalar or indexable,
        # * At least one input must be indexable.
        # * All inputs of a multiplication must be non-scalars.
        #   (We rewrite scalar-matrix multiplications in a different fixer.)

        if type(operator) not in operator_factories:
            return False
        if not _is_fixable_size(sizer[operator]):
            return False
        if not _inputs_are_devectorizable(sizer, operator):
            return False
        if isinstance(operator, bn.MultiplicationNode) and not all(
            _is_indexable_node(sizer, i) for i in operator.inputs
        ):
            return False
        return True

    def vect_op_node_fixer(operator: bn.BMGNode) -> NodeFixerResult:

        # The graph transformation we're doing here takes graphs of the form:
        #
        # indexable  -->   operator  -->  consumer
        #
        # where the "indexable" produces a matrix, the consumer takes a matrix,
        # but the BMG operator only operates on scalars.
        #
        # We transform it into the graph:
        #
        #           --> index[0]  -->  operator -->
        # indexable                                to_matrix  --> consumer
        #           --> index[1]  -->  operator -->
        #           ...
        #
        # And now everyone is happy; the operators get scalars and the
        # consumer gets a matrix.
        #
        # Obviously this increases the number of nodes in the graph by O(n) in
        # the size of the indexible matrix but until we have more vectorized BMG
        # operators we cannot do much better.  (Also, we can often optimize away
        # some of the indexing operations in the arithmetic graph rewriter.)
        #
        if not _is_fixable_operator(sizer, operator):
            return Inapplicable

        # We need to generate n new operator nodes, each of which takes
        # some scalar indexed from its operands. The factory method that
        # builds those operator nodes is in the operator factories list;
        # _generate_arglists constructs the arguments to that factory method.
        arglists = _generate_arglists(bmg, sizer, operator)
        results = []
        factory = operator_factories[type(operator)]
        for arglist in arglists:
            r = factory(*arglist)
            results.append(r)
        size = sizer[operator]
        # We now have n new operator nodes; stick them into a tensor.  We then
        # return that tensor. The caller will retarget the input edge of the
        # consumer from the original operator to the tensor, and the graph is
        # rewritten.
        t = bmg.add_tensor(size, *results)
        return t

    return vect_op_node_fixer


def vectorized_operator_fixer(bmg: BMGraphBuilder) -> GraphFixer:
    def vop_fixer() -> GraphFixerResult:
        sizer = Sizer()

        dist_fixer = _vectorized_distribution_node_fixer(bmg, sizer)
        oper_fixer = _vectorized_operator_node_fixer(bmg, sizer)
        scale_fixer = matrix_scale_fixer(bmg, sizer)
        node_fixer = node_fixer_first_match([dist_fixer, oper_fixer, scale_fixer])
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

    return vop_fixer


def vectorized_observation_fixer(bmg: BMGraphBuilder) -> GraphFixer:
    def vobs_fixer() -> GraphFixerResult:
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

    return vobs_fixer


def vectorized_model_fixer(bmg: BMGraphBuilder) -> GraphFixer:
    vector_ops = vectorized_operator_fixer(bmg)
    vector_obs = vectorized_observation_fixer(bmg)
    return fixpoint_graph_fixer(sequential_graph_fixer([vector_ops, vector_obs]))
