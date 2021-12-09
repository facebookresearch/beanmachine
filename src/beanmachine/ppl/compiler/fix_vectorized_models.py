# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Dict, Type, Callable

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.sizer import Sizer

# TODO Move this to a utils module
from beanmachine.ppl.compiler.support import _prod
from beanmachine.ppl.compiler.typer_base import TyperBase
from torch import Size, tensor


# This class turns vectorized models into unvectorized models.
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


class VectorizedOperatorFixer(ProblemFixerBase):

    fixed_one: bool
    _distribution_factories: Dict[Type, Callable]
    _operator_factories: Dict[Type, Callable]

    def __init__(self, bmg: BMGraphBuilder, typer: Sizer) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)
        self.fixed_one = False
        # TODO: categorical
        # TODO: categorical logit
        # TODO: dirichlet
        self._distribution_factories = {
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
        self._operator_factories = {
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

    def _node_to_index_list(self, n: bn.BMGNode) -> List[bn.BMGNode]:
        size = self._typer[n]
        dim = len(size)
        index_list = []
        if dim == 0:
            # If we have just a single value then there's no indexing required.
            index_list.append(n)
        elif dim == 1:
            for i in range(0, size[0]):
                ci = self._bmg.add_constant(i)
                ni = self._bmg.add_index(n, ci)
                index_list.append(ni)
        else:
            assert dim == 2
            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    ci = self._bmg.add_constant(i)
                    cj = self._bmg.add_constant(j)
                    ni = self._bmg.add_index(n, ci)
                    nij = self._bmg.add_index(ni, cj)
                    index_list.append(nij)
        return index_list

    def _generate_arglists(self, node: bn.BMGNode):
        # This code is a bit tricky to understand so lets work an example.
        # Suppose node has two inputs, call them X and Y. X has size [3], Y has
        # size [2, 3], and node has size [2, 3].
        final_size = self._typer[node]  # Size([2, 3])
        final_length = _prod(final_size)  # 2 x 3 = 6
        input_nodes = [self._node_to_index_list(n) for n in node.inputs]
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
            input_size = self._typer[node.inputs[i]]  # Size([3])
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

    def _replace_sample(self, node: bn.SampleNode) -> bn.BMGNode:
        dist = node.operand
        arglists = self._generate_arglists(dist)
        samples = []
        for arglist in arglists:
            b = self._distribution_factories[type(dist)](*arglist)
            s = self._bmg.add_sample(b)
            samples.append(s)
        size = self._typer[dist]
        t = self._bmg.add_tensor(size, *samples)
        return t

    def _replace_operator(self, node: bn.OperatorNode) -> bn.BMGNode:
        arglists = self._generate_arglists(node)
        results = []
        for arglist in arglists:
            r = self._operator_factories[type(node)](*arglist)
            results.append(r)
        size = self._typer[node]
        t = self._bmg.add_tensor(size, *results)
        return t

    def _is_fixable_sample(self, n: bn.BMGNode) -> bool:
        if not isinstance(n, bn.SampleNode):
            return False
        dist = n.operand
        if type(dist) not in self._distribution_factories:
            return False
        return _is_fixable_size(self._typer[dist])

    def _is_fixable_operator(self, n: bn.BMGNode) -> bool:
        if type(n) not in self._operator_factories:
            return False
        # We do not rewrite multiplications of matrices by scalars; that's
        # handled in a later rewriter.
        if (
            isinstance(n, bn.MultiplicationNode)
            and len(n.inputs) == 2
            and (
                len(self._typer[n.inputs[0]]) <= 1 or len(self._typer[n.inputs[1]]) <= 1
            )
        ):
            return False
        return _is_fixable_size(self._typer[n])

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return self._is_fixable_sample(n) or self._is_fixable_operator(n)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        self.fixed_one = True
        if isinstance(n, bn.SampleNode):
            return self._replace_sample(n)
        assert isinstance(n, bn.OperatorNode)
        return self._replace_operator(n)


class VectorizedModelFixer:

    _bmg: BMGraphBuilder
    errors: ErrorReport

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        # We don't need the passed-in typer.
        self._bmg = bmg
        self.errors = ErrorReport()

    def _fix_observations(self) -> None:
        for o in self._bmg.all_observations():
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
                    self._bmg.add_observation(s, o.value[i])
            else:
                assert dim == 2
                for i in range(0, observed._size[0]):
                    for j in range(0, observed._size[1]):
                        s = observed.inputs[i * observed._size[1] + j]
                        assert isinstance(s, bn.SampleNode)
                        self._bmg.add_observation(s, o.value[i][j])
            self._bmg.remove_leaf(o)

    def fix_problems(self) -> None:
        vf = VectorizedOperatorFixer(self._bmg, Sizer())
        vf.fix_problems()
        assert not vf.errors.any()

        if not vf.fixed_one:
            # We changed nothing so there is nothing more to do.
            return

        # We changed something. We might have a leaf sample node; we can remove it.
        for n in self._bmg.all_nodes():
            if vf._is_fixable_sample(n):
                assert n.is_leaf
                self._bmg.remove_leaf(n)

        # We might have an illegal observation. Fix it.
        self._fix_observations()
