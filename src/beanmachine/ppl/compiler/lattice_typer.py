# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# See notes in typer_base.py for how the type computation logic works.
# See notes in bmg_types.py for the type lattice documentation.
#
# This typer identifies which nodes in the graph are "typable", and of the
# typable nodes, determines the *smallest* lattice type possible for that
# node.
#
# A node is "typable" if (1) it is either a constant or valid BMG node,
# and (2) all of its ancestors are typable. If either requirement is not
# met then a node is untypable.
#
# The purpose of this restriction is to avoid doing work to guess at what
# the types of nodes are in graphs where there is no possibility of this
# graph being legal. We also wish to avoid reporting confusing cascading
# errors based on incorrect guesses as to what the type of the node "should"
# be. Descendents of untypable nodes are also untyped; this is a clear
# and easily implemented rule.
#
# Suppose then we have a node where all of its ancestors are typeable. What
# is the "smallest lattice type" computed here?
#
# For example, suppose we have an addition with two inputs: a sample
# from a beta and a sample from a half Cauchy. The types of the samples
# cannot be smaller than Probability and PositiveReal, respectively.
# An addition of two dissimilarly-typed nodes is not legal, but we could
# make it legal by converting both nodes to PositiveReal OR to Real,
# and then outputting that type.  The smallest of those two possibilities
# is PositiveReal, so this is the lattice type we associate with such
# an addition node.
#
# This class implements rules for each typable BMG node. When adding new
# logic for nodes, keep the following in mind:
#
# * The logic in the base class ensures that types of all ancestors are computed
#   first. We automatically mark all nodes with untypable ancestors as untypable.
#   Therefore we can assume that the type of every ancestor node is both computed
#   and it is a valid type.
#
# * "Untyped" constant node types are computed solely from their *values*.
#   For example, a constant tensor(1.0) can have the lattice type One
#   because we can use that value in a context where a Boolean, Natural,
#   Probability, and so on, are needed, just by creating a typed constant node
#   of the appropriate type.
#
# * "Typed" constant nodes have the type associated with that node regardless
#   of the type of the value; a constant real node with value 2.0 has type
#   real, even though it could be a natural or positive real.
#
# * For non-constant nodes, the lattice type associated with a node should always
#   be an actual BMG type that the node could have. For example: the BMG addition
#   operator requires that its output be PostiveReal, NegativeReal or Real, so
#   the lattice type must be one of those three. We never say "this is a sum of
#   naturals therefore the sum node is also a natural".  And we never say "this
#   is the sum of two 1x3 simplexes, so therefore the result is a 1x3 positive real
#   matrix", and so on.
#
# By following these rules we will be able to more easily compute what the edge
# requirements are, what conversion nodes must be inserted, and what errors must
# be reported when a graph cannot be transformed as required.


import typing
from typing import Callable, Dict, Set

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.typer_base import TyperBase


# Node types which always have the same lattice type
# no matter what the types of their inputs are.

_requires_nothing: Dict[type, bt.BMGLatticeType] = {
    # Distributions
    bn.BernoulliLogitNode: bt.Boolean,
    bn.BernoulliNode: bt.Boolean,
    bn.BetaNode: bt.Probability,
    bn.BinomialNode: bt.Natural,
    bn.CategoricalNode: bt.Natural,
    bn.FlatNode: bt.Probability,
    bn.GammaNode: bt.PositiveReal,
    bn.HalfCauchyNode: bt.PositiveReal,
    bn.NormalNode: bt.Real,
    bn.HalfNormalNode: bt.PositiveReal,
    bn.StudentTNode: bt.Real,
    # Factors
    bn.ExpProductFactorNode: bt.Real,
    # Operators
    bn.LogisticNode: bt.Probability,
    bn.LogSumExpNode: bt.Real,
    bn.LogSumExpVectorNode: bt.Real,
    bn.Log1mexpNode: bt.NegativeReal,
    bn.PhiNode: bt.Probability,
    bn.ToNegativeRealNode: bt.NegativeReal,
    bn.ToIntNode: bt.Natural,
    bn.ToRealNode: bt.Real,
    bn.ToPositiveRealNode: bt.PositiveReal,
    bn.ToProbabilityNode: bt.Probability,
    # Typed constants
    bn.ConstantTensorNode: bt.Tensor,
    bn.BooleanNode: bt.Boolean,
    bn.NaturalNode: bt.Natural,
    bn.NegativeRealNode: bt.NegativeReal,
    bn.PositiveRealNode: bt.PositiveReal,
    bn.ProbabilityNode: bt.Probability,
    bn.RealNode: bt.Real,
}


# This maps FROM the *python* type of a node which represents a constant matrix
# TO a canonical instance of the *graph type object*. We need to be able to
# inspect a node's python type and then construct a graph type object that
# has the same dimensionality as the node.
_constant_matrix_graph_types: Dict[type, bt.BMGMatrixType] = {
    bn.ConstantBooleanMatrixNode: bt.Boolean,
    bn.ConstantNaturalMatrixNode: bt.Natural,
    bn.ConstantNegativeRealMatrixNode: bt.NegativeReal,
    bn.ConstantPositiveRealMatrixNode: bt.PositiveReal,
    bn.ConstantProbabilityMatrixNode: bt.Probability,
    bn.ConstantRealMatrixNode: bt.Real,
    bn.ConstantSimplexMatrixNode: bt.SimplexMatrix(1, 1),
}

# These are the node types which always represent a matrix in BMG.
# Even if the node is a 1x1 matrix, it is a matrix and not an atomic value.
_always_matrix_types: Set[type] = {
    bn.ColumnIndexNode,
    bn.ConstantBooleanMatrixNode,
    bn.ConstantNaturalMatrixNode,
    bn.ConstantNegativeRealMatrixNode,
    bn.ConstantPositiveRealMatrixNode,
    bn.ConstantProbabilityMatrixNode,
    bn.ConstantRealMatrixNode,
    bn.ConstantSimplexMatrixNode,
    bn.ToMatrixNode,
    bn.MatrixScaleNode,
}


class LatticeTyper(TyperBase[bt.BMGLatticeType]):

    _dispatch: Dict[type, Callable]

    def __init__(self) -> None:
        TyperBase.__init__(self)
        self._dispatch = {
            bn.Observation: self._type_observation,
            bn.Query: self._type_query,
            bn.DirichletNode: self._type_dirichlet,
            # Operators
            bn.AdditionNode: self._type_addition,
            bn.ChoiceNode: self._type_choice,
            bn.ColumnIndexNode: self._type_column_index,
            bn.ComplementNode: self._type_complement,
            bn.ExpM1Node: self._type_expm1,
            bn.ExpNode: self._type_exp,
            bn.IfThenElseNode: self._type_if,
            bn.LogNode: self._type_log,
            bn.MultiplicationNode: self._type_multiplication,
            bn.MatrixScaleNode: self._type_matrix_scale,
            bn.NegateNode: self._type_negate,
            bn.PowerNode: self._type_power,
            bn.SampleNode: self._type_sample,
            bn.ToMatrixNode: self._type_to_matrix,
            bn.ToPositiveRealMatrixNode: self._type_to_pos_real_matrix,
            bn.ToRealMatrixNode: self._type_to_real_matrix,
            bn.VectorIndexNode: self._type_index,
        }

    def _type_observation(self, node: bn.Observation) -> bt.BMGLatticeType:
        return self[node.observed]

    def _type_query(self, node: bn.Query) -> bt.BMGLatticeType:
        return self[node.operator]

    def _type_dirichlet(self, node: bn.DirichletNode) -> bt.BMGLatticeType:
        # The type of a Dirichlet node is a one-column simplex with as many
        # rows as its input.
        input_type = self[node.concentration]
        rows = 1
        columns = 1
        if isinstance(input_type, bt.BMGMatrixType):
            rows = input_type.rows
        return bt.SimplexMatrix(rows, columns)

    def _type_addition(self, node: bn.BMGNode) -> bt.BMGLatticeType:
        op_type = bt.supremum(*[self[i] for i in node.inputs])
        if bt.supremum(op_type, bt.NegativeReal) == bt.NegativeReal:
            return bt.NegativeReal
        if bt.supremum(op_type, bt.PositiveReal) == bt.PositiveReal:
            return bt.PositiveReal
        return bt.Real

    def _type_column_index(self, node: bn.ColumnIndexNode) -> bt.BMGLatticeType:
        # A stochastic index into a one-hot or all-zero constant matrix
        # is treated as a column of bools.
        lt = self[node.left]
        assert isinstance(lt, bt.BMGMatrixType)
        result = lt
        if isinstance(lt, bt.ZeroMatrix) or isinstance(lt, bt.OneHotMatrix):
            result = bt.Boolean
        return result.with_dimensions(lt.rows, 1)

    def _type_complement(self, node: bn.ComplementNode) -> bt.BMGLatticeType:
        if bt.supremum(self[node.operand], bt.Boolean) == bt.Boolean:
            return bt.Boolean
        return bt.Probability

    def _type_exp(self, node: bn.ExpNode) -> bt.BMGLatticeType:
        ot = self[node.operand]
        if bt.supremum(ot, bt.NegativeReal) == bt.NegativeReal:
            return bt.Probability
        return bt.PositiveReal

    def _type_expm1(self, node: bn.ExpM1Node) -> bt.BMGLatticeType:
        # ExpM1 takes a real, positive real or negative real. Its return has
        # the same type as its input.
        ot = self[node.operand]
        if bt.supremum(ot, bt.PositiveReal) == bt.PositiveReal:
            return bt.PositiveReal
        if bt.supremum(ot, bt.NegativeReal) == bt.NegativeReal:
            return bt.NegativeReal
        return bt.Real

    def _type_if(self, node: bn.IfThenElseNode) -> bt.BMGLatticeType:
        # TODO: Consider adding a pass which optimizes away IF(X, Y, Y) to
        # just plain Y.
        # TODO: What if we have an atomic type on one side and a 1x1 matrix
        # on the other? That has not yet arisen in practice but we might
        # consider putting a matrix constraint on the atomic side and
        # marking IF as producing a matrix in that case.

        # TODO: We need to consider what happens if the consequence and alternative
        # types have no supremum other than Tensor. In bmg_requirements.py we impose
        # the requirement that the consequence and alternative are both of their
        # supremum, but if that is Tensor then we need to give an error.

        result = bt.supremum(self[node.consequence], self[node.alternative])
        if result == bt.Zero or result == bt.One:
            result = bt.Boolean
        return result

    def _type_choice(self, node: bn.ChoiceNode) -> bt.BMGLatticeType:
        # The type of a choice node is the supremum of all its value's types.

        # TODO: We need to consider what happens if the value's types
        # have no supremum other than Tensor. In bmg_requirements.py we
        # impose the requirement that the values are both of their
        # supremum, but if that is Tensor then we need to give an error.
        result = bt.supremum(
            *(self[node.inputs[i]] for i in range(1, len(node.inputs)))
        )
        if result == bt.Zero or result == bt.One:
            result = bt.Boolean
        return result

    def _type_index(self, node: bn.VectorIndexNode) -> bt.BMGLatticeType:
        # The lattice type of an index is derived from the lattice type of
        # the vector, but it's not as straightforward as just
        # shrinking the type down to a 1x1 matrix. The elements of
        # a one-hot vector are bools, for instance, not all one.
        # The elements of a simplex are probabilities.
        lt = self[node.left]
        if isinstance(lt, bt.OneHotMatrix):
            return bt.Boolean
        if isinstance(lt, bt.ZeroMatrix):
            return bt.Boolean
        if isinstance(lt, bt.SimplexMatrix):
            return bt.Probability
        if isinstance(lt, bt.BMGMatrixType):
            return lt.with_dimensions(1, 1)
        # The only other possibility is that we have a tensor, so let's say
        # its elements are reals.
        return bt.Real

    def _type_log(self, node: bn.LogNode) -> bt.BMGLatticeType:
        ot = bt.supremum(self[node.operand], bt.Probability)
        if ot == bt.Probability:
            return bt.NegativeReal
        return bt.Real

    def _type_multiplication(self, node: bn.MultiplicationNode) -> bt.BMGLatticeType:
        ot = bt.supremum(*[self[i] for i in node.inputs])
        it = bt.supremum(ot, bt.Probability)
        if bt.supremum(it, bt.Real) == bt.Real:
            return it
        return bt.Real

    def _type_matrix_scale(self, node: bn.MatrixScaleNode) -> bt.BMGLatticeType:
        assert len(node.inputs) == 2
        lt = self[node.left]
        assert lt is not bt.Untypable
        assert bt.supremum(lt, bt.Real) == bt.Real
        assert isinstance(
            lt, bt.BMGMatrixType
        )  # Beanstalk scalars are single matrix types
        lt = typing.cast(bt.BroadcastMatrixType, lt)
        rt = self[node.right]
        assert rt is not bt.Untypable
        assert isinstance(rt, bt.BMGMatrixType)
        ltm = lt.with_dimensions(rt.rows, rt.columns)
        return bt.supremum(ltm, rt)

    def _type_negate(self, node: bn.NegateNode) -> bt.BMGLatticeType:
        ot = self[node.operand]
        if bt.supremum(ot, bt.PositiveReal) == bt.PositiveReal:
            return bt.NegativeReal
        if bt.supremum(ot, bt.NegativeReal) == bt.NegativeReal:
            return bt.PositiveReal
        return bt.Real

    def _type_power(self, node: bn.PowerNode) -> bt.BMGLatticeType:
        # BMG supports a power node that has these possible combinations of
        # base and exponent type:
        #
        # P ** R+  --> P
        # P ** R   --> R+
        # R+ ** R+ --> R+
        # R+ ** R  --> R+
        # R ** R+  --> R
        # R ** R   --> R
        inf_base = bt.supremum(self[node.left], bt.Probability)
        inf_exp = bt.supremum(self[node.right], bt.PositiveReal)
        if inf_base == bt.Probability and inf_exp == bt.Real:
            return bt.PositiveReal
        if bt.supremum(inf_base, bt.Real) == bt.Real:
            return inf_base
        return bt.Real

    def _type_sample(self, node: bn.SampleNode) -> bt.BMGLatticeType:
        return self[node.operand]

    def _type_to_matrix(self, node: bn.ToMatrixNode) -> bt.BMGLatticeType:
        assert len(node.inputs) >= 3
        rows = node.inputs[0]
        assert isinstance(rows, bn.NaturalNode)
        columns = node.inputs[1]
        assert isinstance(columns, bn.NaturalNode)
        t = bt.supremum(*(self[item] for item in node.inputs.inputs[2:]))
        if bt.supremum(t, bt.Real) != bt.Real:
            t = bt.Real
        elif t == bt.One or t == bt.Zero:
            # This should not happen, but just to be sure we'll make
            # an all-one or all-zero matrix into a matrix of bools.
            # (It should not happen because an all-constant matrix should
            # already be a TensorConstant node.)
            t = bt.Boolean
        assert isinstance(t, bt.BMGMatrixType)
        return t.with_dimensions(rows.value, columns.value)

    def _type_to_real_matrix(self, node: bn.ToRealMatrixNode) -> bt.BMGLatticeType:
        op = node.operand
        t = self[op]
        assert isinstance(t, bt.BMGMatrixType)
        assert self.is_matrix(op)
        return bt.RealMatrix(t.rows, t.columns)

    def _type_to_pos_real_matrix(
        self, node: bn.ToPositiveRealMatrixNode
    ) -> bt.BMGLatticeType:
        op = node.operand
        t = self[op]
        assert isinstance(t, bt.BMGMatrixType)
        assert self.is_matrix(op)
        return bt.PositiveRealMatrix(t.rows, t.columns)

    def _compute_type_inputs_known(self, node: bn.BMGNode) -> bt.BMGLatticeType:
        # If there is any input node whose type cannot be determined, then *none*
        # of its descendents can be determined, even if a descendent node always
        # has the same type regardless of its inputs. This ensures that (1) we only
        # attempt to assign type judgments to graphs that are supported by BMG,
        # and (2) will help us avoid presenting cascading errors to the user in
        # the event that a graph violates a typing rule.
        for i in node.inputs:
            if self[i] == bt.Untypable:
                return bt.Untypable
        if isinstance(node, bn.UntypedConstantNode):
            return bt.type_of_value(node.value)
        t = type(node)
        if t in _requires_nothing:
            result = _requires_nothing[t]
        elif t in _constant_matrix_graph_types:
            assert isinstance(node, bn.ConstantTensorNode)
            r = _constant_matrix_graph_types[t]
            result = r.with_size(node.value.size())
        elif t in self._dispatch:
            result = self._dispatch[t](node)
        else:
            # TODO: Consider asserting that the node is unsupported by BMG.
            result = bt.Untypable
        assert result != bt.Zero and result != bt.One
        return result

    def is_bool(self, node: bn.BMGNode) -> bool:
        t = self[node]
        return t != bt.Untypable and bt.supremum(t, bt.Boolean) == bt.Boolean

    def is_natural(self, node: bn.BMGNode) -> bool:
        t = self[node]
        return t != bt.Untypable and bt.supremum(t, bt.Natural) == bt.Natural

    def is_prob_or_bool(self, node: bn.BMGNode) -> bool:
        t = self[node]
        return t != bt.Untypable and bt.supremum(t, bt.Probability) == bt.Probability

    def is_matrix(self, node: bn.BMGNode) -> bool:
        t = type(node)
        if t in _always_matrix_types:
            return True
        lattice_type = self[node]
        if isinstance(lattice_type, bt.SimplexMatrix):
            return True
        if isinstance(lattice_type, bt.BMGMatrixType) and (
            lattice_type.rows != 1 or lattice_type.columns != 1
        ):
            return True
        return False
