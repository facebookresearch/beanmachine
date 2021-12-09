# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# In lattice_typer.py we assign a type to every node in a graph.
# If the node or any ancestor node is unsupported in BMG then the node
# is said to be "untypable"; otherwise, it computes (and caches) the smallest
# possible lattice type for that node.
#
# We can similarly put "requirements" on edges. The rules are as follows:
#
# * Every node in a graph has n >= 0 inputs.  For every node we can compute
#   a list of n requirements, one corresponding to each edge.
#
# * If a node is untypable then every edge requirement is "Any" -- that is,
#   there is no requirement placed on the edge. The node cannot be represented
#   in BMG, so there is no reason to report a requirement violation on any of
#   its edges.
#
# * Requirements of input edges of typable nodes are computed for each node.
#   Since this computation is cheap and requires no traversal of the graph
#   once the lattice type is known, we do not attempt to cache this information.
#

import typing
from typing import Callable, Dict, List

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


# These are the nodes which always have the same requirements no matter
# what their inputs are.

_known_requirements: Dict[type, List[bt.Requirement]] = {
    bn.Observation: [bt.AnyRequirement()],
    bn.Query: [bt.AnyRequirement()],
    # Distributions
    bn.BernoulliLogitNode: [bt.Real],
    bn.BernoulliNode: [bt.Probability],
    bn.BetaNode: [bt.PositiveReal, bt.PositiveReal],
    bn.BinomialNode: [bt.Natural, bt.Probability],
    bn.GammaNode: [bt.PositiveReal, bt.PositiveReal],
    bn.HalfCauchyNode: [bt.PositiveReal],
    bn.NormalNode: [bt.Real, bt.PositiveReal],
    bn.HalfNormalNode: [bt.PositiveReal],
    bn.StudentTNode: [bt.PositiveReal, bt.Real, bt.PositiveReal],
    # Operators
    bn.LogisticNode: [bt.Real],
    bn.Log1mexpNode: [bt.NegativeReal],
    bn.PhiNode: [bt.Real],
    bn.ToIntNode: [bt.upper_bound(bt.Real)],
    bn.ToNegativeRealNode: [bt.Real],
    bn.ToRealNode: [bt.upper_bound(bt.Real)],
    bn.ToPositiveRealNode: [bt.upper_bound(bt.PositiveReal)],
    bn.ToProbabilityNode: [bt.upper_bound(bt.Real)],
}


def _requirements_valid(node: bn.BMGNode, reqs: List[bt.Requirement]):
    return len(reqs) == len(node.inputs) and not any(
        r in bt._invalid_requirement_types for r in reqs
    )


class EdgeRequirements:

    _dispatch: Dict[type, Callable]

    typer: LatticeTyper

    def __init__(self, typer: LatticeTyper) -> None:
        self.typer = typer
        self._dispatch = {
            # Factors
            bn.ExpProductFactorNode: self._requirements_expproduct,
            # Distributions
            bn.CategoricalNode: self._requirements_categorical,
            bn.DirichletNode: self._requirements_dirichlet,
            # Operators
            bn.AdditionNode: self._requirements_addition,
            bn.ChoiceNode: self._requirements_choice,
            bn.ColumnIndexNode: self._requirements_column_index,
            bn.ComplementNode: self._same_as_output,
            bn.ExpM1Node: self._same_as_output,
            bn.ExpNode: self._requirements_exp_neg,
            bn.IfThenElseNode: self._requirements_if,
            bn.LogNode: self._requirements_log,
            bn.LogSumExpNode: self._requirements_logsumexp,
            bn.LogSumExpVectorNode: self._requirements_logsumexp_vector,
            bn.MultiplicationNode: self._requirements_multiplication,
            bn.MatrixScaleNode: self._requirements_matrix_scale,
            bn.NegateNode: self._requirements_exp_neg,
            bn.PowerNode: self._requirements_power,
            bn.SampleNode: self._same_as_output,
            bn.ToMatrixNode: self._requirements_to_matrix,
            bn.VectorIndexNode: self._requirements_index,
        }

    def _requirements_expproduct(
        self, node: bn.ExpProductFactorNode
    ) -> List[bt.Requirement]:
        # Each input to an exp-power is required to be a
        # real, negative real, positive real or probability.
        return [
            self.typer[i]
            if self.typer[i]
            in {bt.Real, bt.NegativeReal, bt.PositiveReal, bt.Probability}
            else bt.Real
            for i in node.inputs
        ]

    def _requirements_dirichlet(self, node: bn.DirichletNode) -> List[bt.Requirement]:
        # BMG's Dirichlet node requires that the input be a column
        # vector of positive reals, and the row count of the vector is
        # the number of elements in the simplex we produce. We can
        # express that restriction as a positive real matrix with
        # column count equal to 1 and row count equal to the final
        # dimension of the size.
        #
        # A degenerate case here is Dirichlet(tensor([1.0])); we would
        # normally generate the input as a positive real constant, but
        # we require that it be a positive real constant 1x1 *matrix*,
        # which is a different node. The "always matrix" requirement
        # forces the problem fixer to ensure that the input node is
        # always considered by BMG to be a matrix.
        #
        # TODO: BMG requires it to be a *broadcast* matrix; what happens
        # if we feed one Dirichlet into another?  That would be a simplex,
        # not a broadcast matrix. Do some research here; do we actually
        # need the semantics of "always a broadcast matrix" ?

        # To compute the requirement we get the matrix column size of the input.

        input_type = self.typer[node]
        required_columns = 1
        required_rows = 1

        # TODO: This will produce a bad error message experience. What we want to
        # say in the case where the input is not a matrix at all is "we require a
        # one-column positive real matrix" and not "we require a 1x1 positive real
        # matrix".

        if isinstance(input_type, bt.BMGMatrixType):
            required_rows = input_type.rows

        return [
            bt.always_matrix(
                bt.PositiveReal.with_dimensions(required_rows, required_columns)
            )
        ]

    def _requirements_categorical(
        self, node: bn.CategoricalNode
    ) -> List[bt.Requirement]:
        # A categorical node requires that its input be a simplex with a single column.

        it = self.typer[node.probability]

        # If we have a matrix input then its lattice type is one hot matrix, simplex,
        # or broadcast matrix.
        #
        # Let's list all the possible cases:
        #
        # * If we have an r x c one hot or simplex then require a r x 1 simplex;
        #   This will succeed if c == 1 and fail with a good error message otherwise.
        #
        # * If we have an r x c other matrix then require a r x 1 simplex;
        #   This will always fail with a good error message.
        #
        # In all cases we require an r x 1 simplex:

        if isinstance(it, bt.BMGMatrixType):
            return [bt.SimplexMatrix(it.rows, 1)]

        # We do not have a matrix input. There are two possibilities:
        #
        # * The input is 1.  We require a 1 x 1 simplex, and this will succeed.
        # * The input is any other atomic value, and we need to fail.  But what
        #   unmet requirement should we put on this edge?

        # TODO: The current code produces a bad error message experience because at
        # present we have no way to represent the stated requirement: "a simplex with
        # one column and ANY number of rows".  We only have a way to represent the
        # requirement "a simplex with one column and a SPECIFIC number of rows".
        # We should create a requirement object that represents the real requirement,
        # and then error reporting can give a sensible error message.
        #
        # Note that once we have this requirement object we can simply return it
        # for *every* case and delete all this code. The requirements fixer
        # would then be responsible for determining if the requirement can be met.

        return [bt.SimplexMatrix(1, 1)]

    def _requirements_addition(self, node: bn.BMGNode) -> List[bt.Requirement]:
        it = self.typer[node]
        assert it in {bt.Real, bt.NegativeReal, bt.PositiveReal}
        return [it] * len(node.inputs)  # pyre-ignore

    def _requirements_column_index(
        self, node: bn.ColumnIndexNode
    ) -> List[bt.Requirement]:
        # See notes in _requirements_index.  If we have an index into
        # a one-hot matrix or an all-zero matrix, assume that we want
        # it to be a column of bools.
        lt = self.typer[node.left]
        assert isinstance(lt, bt.BMGMatrixType)
        result = lt
        if isinstance(lt, bt.OneHotMatrix) or isinstance(lt, bt.ZeroMatrix):
            result = bt.Boolean.with_dimensions(lt.rows, lt.columns)
        return [bt.always_matrix(result), bt.Natural]

    def _requirements_exp_neg(self, node: bn.UnaryOperatorNode) -> List[bt.Requirement]:
        # Same logic for both exp and negate operators
        ot = self.typer[node.operand]
        if bt.supremum(ot, bt.NegativeReal) == bt.NegativeReal:
            return [bt.NegativeReal]
        if bt.supremum(ot, bt.PositiveReal) == bt.PositiveReal:
            return [bt.PositiveReal]
        return [bt.Real]

    def _same_as_output(self, node: bn.BMGNode) -> List[bt.Requirement]:
        # Input type must be same as output type
        return [self.typer[node]]

    def _requirements_if(self, node: bn.IfThenElseNode) -> List[bt.Requirement]:
        # The condition has to be Boolean; the consequence and alternative need
        # to be the same.
        # TODO: Consider what to do if the node type is Tensor.
        it = self.typer[node]
        return [bt.Boolean, it, it]

    def _requirements_choice(self, node: bn.ChoiceNode) -> List[bt.Requirement]:
        # The condition has to be Natural; the values must all be of the same type.
        # TODO: Consider what to do if the node type is Tensor.
        node_type = self.typer[node]
        c: List[bt.Requirement] = [bt.Natural]
        v: List[bt.Requirement] = [node_type]
        return c + v * (len(node.inputs) - 1)

    def _requirements_index(self, node: bn.VectorIndexNode) -> List[bt.Requirement]:
        # The index operator introduces an interesting wrinkle into the
        # "requirements" computation. Until now we have always had the property
        # that queries and observations are "sinks" of the graph, and the transitive
        # closure of the inputs to the sinks can have their requirements checked in
        # order going from the nodes farthest from the sinks down to the sinks.
        # That is, each node can meet its input requirements *before* its output
        # nodes meet their requirements. We now have a case where doing so creates
        # potential inefficiencies.
        #
        # B is the constant 3x1 vector [[0], [1], [1]]
        # N is any node of type natural.
        # I is an index
        # F is Bernoulli.
        #
        #   B N
        #   | |
        #    I
        #    |
        #    F
        #
        # The requirement on edge I->F is Probability
        # The requirement on edge N->I is Natural.
        # What is the requirement on the B->I edge?
        #
        # If we say that it is Boolean[3, 1], its inf type, then the graph we end up
        # generating is
        #
        # b = const_bool_matrix([[0], [1], [1]])  # 3x1 bool matrix
        # n = whatever                      # natural
        # i = index(b, i)                   # bool
        # z = const_prob(0)                 # prob
        # o = const_prob(1)                 # prob
        # c = if_then_else(i, o, z)         # prob
        # f = Bernoulli(c)                  # bool
        #
        # But it would be arguably better to produce
        #
        # b = const_prob_matrix([[0.], [1.], [1.]])  # 3x1 prob matrix
        # n = whatever                      # natural
        # i = index(b, i)                   # prob
        # f = Bernoulli(i)                  # bool
        #
        # TODO: We might consider an optimization pass which does so.
        #
        # However there is an even worse situation. Suppose we have
        # this unlikely-but-legal graph:
        #
        # Z is [[0], [0], [0]]
        # N is any natural
        # I is an index
        # C requires a Boolean input
        # L requires a NegativeReal input
        #
        #    Z   N
        #     | |
        #      I
        #     | |
        #    C   L
        #
        # The inf type of Z is Zero[3, 1].
        # The I->C edge requirement is Boolean
        # The I->L edge requirement is NegativeReal
        #
        # Now what requirement do we impose on the Z->I edge? We have our choice
        # of "matrix of negative reals" or "matrix of bools", and whichever we
        # pick will disappoint someone.
        #
        # Fortunately for us, this situation is unlikely; a model writer who
        # contrives a situation where they are making a stochastic choice where
        # all choices are all zero AND that zero needs to be used as both
        # false and a negative number is not writing realistic models.
        #
        # What we will do in this unlikely situation is decide that the intended
        # output type is Boolean and therefore the vector is a vector of bools.
        #
        # -----
        #
        # We require:
        # * the vector must be one row
        # * the vector must be a matrix, not a single value
        # * the vector must be either a simplex, or a matrix where the element
        #   type is the output type of the indexing operation
        # * the index must be a natural
        #

        lt = self.typer[node.left]

        # If we have a tensor that has more than two dimensions, who can
        # say what the column count should be?

        # TODO: We need a better error message for that scenario.
        # It will be common for people to use tensors that are too high
        # dimension for BMG to handle and we should say that clearly.

        required_rows = lt.rows if isinstance(lt, bt.BMGMatrixType) else 1
        required_columns = 1

        if isinstance(lt, bt.SimplexMatrix):
            vector_req = lt.with_dimensions(required_rows, required_columns)
        else:
            it = self.typer[node]
            assert isinstance(it, bt.BMGMatrixType)
            vector_req = it.with_dimensions(required_rows, required_columns)

        return [bt.always_matrix(vector_req), bt.Natural]

    def _requirements_log(self, node: bn.LogNode) -> List[bt.Requirement]:
        # Input must be probability or positive real; choose the smaller.
        ot = bt.supremum(self.typer[node.operand], bt.Probability)
        if ot == bt.Probability:
            return [bt.Probability]
        return [bt.PositiveReal]

    def _requirements_logsumexp(self, node: bn.LogSumExpNode) -> List[bt.Requirement]:
        s = bt.supremum(*[self.typer[i] for i in node.inputs])
        if s not in {bt.Real, bt.NegativeReal, bt.PositiveReal}:
            s = bt.Real
        return [s] * len(node.inputs)  # pyre-ignore

    def _requirements_logsumexp_vector(
        self, node: bn.LogSumExpVectorNode
    ) -> List[bt.Requirement]:
        # LOGSUMEXP_VECTOR requires:
        # * a broadcast matrix with one column and any number of rows
        # * The matrix has real, positive real, or negative real elements.
        #
        # TODO: What about simplex matrices? It should be legal to do
        # a logsumexp of a sample from a Dirichlet, right? Can we convert
        # a simplex to a broadcast matrix of positive reals?
        t = self.typer[node.operand]
        required_rows = t.rows if isinstance(t, bt.BMGMatrixType) else 1
        required_columns = 1
        if isinstance(t, bt.PositiveRealMatrix) or isinstance(t, bt.NegativeRealMatrix):
            t = t.with_dimensions(required_rows, required_columns)
        else:
            t = bt.Real.with_dimensions(required_rows, required_columns)
        return [bt.always_matrix(t)]

    def _requirements_multiplication(
        self, node: bn.MultiplicationNode
    ) -> List[bt.Requirement]:
        it = self.typer[node]
        assert it in {bt.Probability, bt.PositiveReal, bt.Real}
        return [it] * len(node.inputs)  # pyre-ignore

    def _requirements_matrix_scale(
        self, node: bn.MatrixScaleNode
    ) -> List[bt.Requirement]:
        it = self.typer[node]
        assert isinstance(it, bt.BMGMatrixType)
        it = typing.cast(bt.BroadcastMatrixType, it)
        it_scalar = it.with_dimensions(1, 1)
        # TODO[Walid]: Not entirely sure if we need this asserted requirement.
        #              It was included only for analogy with multiplication
        assert it_scalar in {bt.Probability, bt.PositiveReal, bt.Real}
        return [it_scalar, it]

    def _requirements_power(self, node: bn.PowerNode) -> List[bt.Requirement]:
        # BMG supports a power node that has these possible combinations of
        # base and exponent type:
        #
        # P ** R+  --> P
        # P ** R   --> R+
        # R+ ** R+ --> R+
        # R+ ** R  --> R+
        # R ** R+  --> R
        # R ** R   --> R
        req_base = bt.supremum(self.typer[node.left], bt.Probability)
        if req_base not in {bt.Probability, bt.PositiveReal, bt.Real}:
            req_base = bt.Real
        req_exp = bt.supremum(self.typer[node.right], bt.PositiveReal)
        if req_exp not in {bt.PositiveReal, bt.Real}:
            req_exp = bt.Real
        return [req_base, req_exp]

    def _requirements_to_matrix(self, node: bn.ToMatrixNode) -> List[bt.Requirement]:
        node_type = self.typer[node]
        assert isinstance(node_type, bt.BMGMatrixType)
        item_type = node_type.with_dimensions(1, 1)
        rc: List[bt.Requirement] = [bt.Natural, bt.Natural]
        its: List[bt.Requirement] = [item_type]
        return rc + its * (len(node.inputs) - 2)

    def requirements(self, node: bn.BMGNode) -> List[bt.Requirement]:
        input_count = len(node.inputs)
        if input_count == 0:
            result = []
        elif self.typer[node] == bt.Untypable:
            result = [bt.AnyRequirement()] * input_count
        else:
            t = type(node)
            if t in _known_requirements:
                result = _known_requirements[t]
            elif t in self._dispatch:
                result = self._dispatch[t](node)
            else:
                result = [bt.AnyRequirement()] * input_count

        assert _requirements_valid(node, result)
        return result
