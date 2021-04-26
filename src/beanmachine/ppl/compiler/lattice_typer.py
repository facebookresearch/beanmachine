# Copyright (c) Facebook, Inc. and its affiliates.

# See notes in typer_base.py for how the type computation logic works.
# See notes in bmg_types.py for the type lattice documentation.
#
# This typer identifies which nodes in the graph are "typable", and of the
# typable nodes, determines the *smallest* lattice type possible for that
# node.
#
# A node is "typable" if (1) it is a valid BMG node and (2) all of its
# ancestors are typable. If either requirement is not met then a node is
# untypable.
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
# * For constant nodes, it is acceptable to associate a "fake" lattice type with
#   the node. For example, a constant real 1.0 can have the lattice type One
#   because we can use that value in a context where a Boolean, Natural,
#   Probability, and so on, are needed, just by creating a constant of the
#   appropriate type.
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

from typing import Callable, Dict

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
    bn.FlatNode: bt.Probability,
    bn.GammaNode: bt.PositiveReal,
    bn.HalfCauchyNode: bt.PositiveReal,
    bn.NormalNode: bt.Real,
    bn.StudentTNode: bt.Real,
    # Factors
    bn.ExpProductFactorNode: bt.Real,
    # Operators
    bn.LogisticNode: bt.Probability,
    bn.LogSumExpNode: bt.Real,
    bn.Log1mexpNode: bt.NegativeReal,
    bn.PhiNode: bt.Probability,
    bn.ToRealNode: bt.Real,
    bn.ToPositiveRealNode: bt.PositiveReal,
    bn.ToProbabilityNode: bt.Probability,
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
            bn.ComplementNode: self._type_complement,
            bn.ExpM1Node: self._type_expm1,
            bn.ExpNode: self._type_exp,
            bn.IfThenElseNode: self._type_if,
            bn.IndexNode: self._type_index,
            bn.LogNode: self._type_log,
            bn.MultiAdditionNode: self._type_addition,
            bn.MultiplicationNode: self._type_multiplication,
            bn.NegateNode: self._type_negate,
            bn.PowerNode: self._type_power,
            bn.SampleNode: self._type_sample,
        }

    def _type_observation(self, node: bn.Observation) -> bt.BMGLatticeType:
        return self[node.observed]

    def _type_query(self, node: bn.Query) -> bt.BMGLatticeType:
        return self[node.operator]

    def _type_dirichlet(self, node: bn.DirichletNode) -> bt.BMGLatticeType:
        return bt.SimplexMatrix(1, node._required_columns)

    def _type_addition(self, node: bn.BMGNode) -> bt.BMGLatticeType:
        op_type = bt.supremum(*[self[i] for i in node.inputs])
        if bt.supremum(op_type, bt.NegativeReal) == bt.NegativeReal:
            return bt.NegativeReal
        if bt.supremum(op_type, bt.PositiveReal) == bt.PositiveReal:
            return bt.PositiveReal
        return bt.Real

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
        # TODO: This isn't quite right. Suppose we have a degenerate case
        # such as IF(flip(), 0, 0).  We should conclude that the type
        # of this node is BOOL, not ZERO.
        # TODO: What about IF (flip(), 1, 1) -- is that BOOL or SIMPLEX?
        # TODO: Consider adding a pass which optimizes away IF(X, Y, Y) to
        # just plain Y.
        return bt.supremum(self[node.consequence], self[node.alternative])

    def _type_index(self, node: bn.IndexNode) -> bt.BMGLatticeType:
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
        it = bt.supremum(self[node.left], self[node.right], bt.Probability)
        if bt.supremum(it, bt.Real) == bt.Real:
            return it
        return bt.Real

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
        t = type(node)
        if t in _requires_nothing:
            result = _requires_nothing[t]
        elif isinstance(node, bn.ConstantNode):
            result = bt.type_of_value(node.value)
        elif t in self._dispatch:
            result = self._dispatch[t](node)
        else:
            result = bt.Untypable
        return result
