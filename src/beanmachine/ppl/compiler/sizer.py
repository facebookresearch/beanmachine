# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# See notes in typer_base.py for how the type computation logic works.
#
# This typer identifies the tensor size associated with a graph node.
# For example, if we have a random variable:
#
# @rv def flips():
#   return Bernoulli(coin())
#
# Then we need to know if coin() is a single probability or, if multiple,
# what shape the tensor is. BMG only supports two-dimensional arrays
# and there are restrictions on how we can produce multi-valued samples,
# multiply matrices, and so on.
#
# Every node must have a *consistent* size; consider for example this
# unlikely but legal model:
#
# @rv def weird(n):
#   if n == 0:
#     return Bernoulli(tensor([0.5, 0.5])) # two values
#   else:
#     return Normal(0.0, 1.0) # one value
#
# @rv def flip():
#   return Bernoulli(0.5)
#
# @fun problem():
#   return weird(flip())
#
# What is the size of the node associated with "problem"? It does not have a consistent
# size, so we will mark it as unsized.
#
# The purpose of this is to avoid doing work to guess at what
# the sizes of nodes are in graphs where there is no possibility of this
# graph being legal. We also wish to avoid reporting confusing cascading
# errors based on incorrect guesses as to what the size of the node "should"
# be. Descendents of unsized nodes are also unsized; this is a clear
# and easily implemented rule.
#
# We use this logic in two main places in the compiler. First, when computing
# supports for stochastic control flow. If we have something like:
#
# @rv def flip_two():
#   return Bernoulli([0.5, 0.5])
#
# @rv def normal(n):
#   return Normal(0, 1)
#
# ...
# x = normal(flip_two())
#
# then we need to know that there are four samples from normal() and we are stochastically
# choosing one of them in x.
#
# Second, when doing various rewrites of the graph we need to know what node sizes are
# so that we can either rewrite operations that cannot be represented in BMG into
# unvectorized operations, or produce sensible error messages if we cannot.

from functools import reduce
from typing import Callable, Dict, Set

import beanmachine.ppl.compiler.bmg_nodes as bn
import torch
from beanmachine.ppl.compiler.typer_base import TyperBase
from torch import Size


# We use an impossible value as a marker for unsizable:
Unsized = Size([-1])
Scalar = Size([])

# These nodes are always scalars no matter what their input:
_always_scalar: Set[type] = {
    bn.ExpProductFactorNode,
    bn.FlatNode,
    bn.InNode,
    bn.IsNode,
    bn.IsNotNode,
    bn.ItemNode,
    bn.NotInNode,
    bn.NotNode,
    bn.ToIntNode,
    bn.ToNegativeRealNode,
    bn.ToRealNode,
    bn.ToPositiveRealNode,
    bn.ToProbabilityNode,
}

# The size of these nodes is just the size of broadcasting all their inputs.
_broadcast_the_inputs: Set[type] = {
    bn.AdditionNode,
    bn.BernoulliLogitNode,
    bn.BernoulliNode,
    bn.BetaNode,
    bn.BinomialLogitNode,
    bn.BinomialNode,
    bn.BitAndNode,
    bn.BitOrNode,
    bn.BitXorNode,
    bn.Chi2Node,
    bn.ComplementNode,
    bn.DivisionNode,
    bn.DirichletNode,
    bn.EqualNode,
    bn.ExpM1Node,
    bn.ExpNode,
    bn.FloorDivNode,
    bn.GammaNode,
    bn.GreaterThanEqualNode,
    bn.GreaterThanNode,
    bn.HalfCauchyNode,
    bn.HalfNormalNode,
    bn.InvertNode,
    bn.LessThanEqualNode,
    bn.LessThanNode,
    bn.LogisticNode,
    bn.LogNode,
    bn.LogSumExpNode,
    bn.Log1mexpNode,
    bn.LShiftNode,
    bn.ModNode,
    bn.MultiplicationNode,
    bn.NegateNode,
    bn.NormalNode,
    bn.NotEqualNode,
    bn.Observation,
    bn.PhiNode,
    bn.PoissonNode,
    bn.PowerNode,
    bn.Query,
    bn.RShiftNode,
    bn.SampleNode,
    bn.StudentTNode,
    bn.ToPositiveRealMatrixNode,
    bn.ToRealMatrixNode,
    bn.UniformNode,
}


def _broadcast_two(x: Size, y: Size) -> Size:
    # Given two sizes, what is their broadcast size, if any?  Rather than replicate
    # the logic in torch which does this computation, we simply construct two
    # all-zero tensors of the given sizes and try to add them. If the addition succeeds
    # then the size of the sum is the size we want.
    #
    # TODO: Is there a better way to do this other than try it and see what happens?
    # TODO: Try torch.distributions.utils.broadcast_all

    if x == Unsized or y == Unsized:
        return Unsized
    try:
        return (torch.zeros(x) + torch.zeros(y)).size()
    except Exception:
        return Unsized


def _broadcast(*sizes: Size) -> Size:
    # Many tensor operations in torch have "broadcast" semantics. A brief explanation:
    #
    # If we compute tensor([1, 2]) + tensor([20, 30]) we do pairwise addition to get
    # tensor([21, 32]) as the sum.  But what happens if the dimensions do not match?
    # In this case we "broadcast" the tensors; we find a tensor size greater than or
    # equal to the sizes of both operands and duplicate the data as necessary.
    #
    # For example, if we are adding tensor([1, 2]) + tensor([3]) then the right summand
    # is broadcast to tensor([3, 3]), and we get tensor([4, 5]) as the sum.
    #
    # Note that not all sizes can be broadcast. Summing tensor([1, 2]) + tensor([10, 20, 30])
    # is not legal because there is no obvious way to expand [1, 2] to be the same size as
    # [10, 20, 30].
    #
    # We often need to answer the question "given n sizes, what is the size of the broadcast
    # of all n sizes?" This method does that computation.
    return reduce(_broadcast_two, sizes)


def size_to_str(size: Size) -> str:
    if size == Unsized:
        return "unsized"
    return "[" + ",".join(str(i) for i in size) + "]"


class Sizer(TyperBase[Size]):

    _dispatch: Dict[type, Callable]

    def __init__(self) -> None:
        TyperBase.__init__(self)
        self._dispatch = {
            bn.ChoiceNode: self._size_choice,
            bn.IfThenElseNode: self._size_if,
            bn.IndexNode: self._size_index,
            bn.MatrixMultiplicationNode: self._size_mm,
            bn.SwitchNode: self._size_switch,
            bn.TensorNode: lambda n: n._size,
            bn.ToMatrixNode: self._size_to_matrix,
        }
        # TODO:
        # ColumnIndexNode
        # LogSumExpTorchNode --
        #   note that final parameter affects size
        # VectorIndexNode
        # LogSumExpVectorNode

    def _size_choice(self, node: bn.ChoiceNode) -> Size:
        s = self[node.inputs[1]]
        for i in range(1, len(node.inputs)):
            if self[node.inputs[i]] != s:
                return Unsized
        return s

    def _size_if(self, node: bn.IfThenElseNode) -> Size:
        consequence = self[node.inputs[1]]
        alternative = self[node.inputs[2]]
        if consequence != alternative:
            return Unsized
        return consequence

    def _size_index(self, node: bn.IndexNode) -> Size:
        collection_size = self[node.left]
        if len(collection_size) == 0:
            # This operation is illegal in torch, so let's just say unsized.
            return Unsized
        result_size = collection_size[1:]
        assert isinstance(result_size, Size)
        return result_size

    def _size_mm(self, node: bn.MatrixMultiplicationNode) -> Size:
        # Just do the multiplication and see what size we get.
        # TODO: Torch supports both broadcasting and non-broadcasting versions
        # of matrix multiplication. We might need to track both separately and
        # ensure that we compute size, support, and so on, accordingly.
        left = torch.zeros(self[node.left])
        right = torch.zeros(self[node.right])
        return left.mm(right).size()

    def _size_switch(self, node: bn.SwitchNode) -> Size:
        s = self[node.inputs[2]]
        for i in range(1, (len(node.inputs) - 1) // 2):
            if self[node.inputs[i * 2 + 2]] != s:
                return Unsized
        return s

    def _size_to_matrix(self, node: bn.ToMatrixNode) -> Size:
        rows = node.inputs[0]
        assert isinstance(rows, bn.NaturalNode)
        columns = node.inputs[1]
        assert isinstance(columns, bn.NaturalNode)
        return Size([rows.value, columns.value])

    # This implements the abstract base type method.
    def _compute_type_inputs_known(self, node: bn.BMGNode) -> Size:
        # If there is any input node whose size cannot be determined, then *none*
        # of its descendents can be determined, even if a descendent node always
        # has the same type regardless of its inputs. This ensures that (1) we only
        # attempt to assign sizes to graphs that are supported by the BMG type system,
        # and (2) will help us avoid presenting cascading errors to the user in
        # the event that a graph violates a typing rule.
        for i in node.inputs:
            if self[i] == Unsized:
                return Unsized
        if isinstance(node, bn.ConstantNode):
            if isinstance(node.value, torch.Tensor):
                return node.value.size()
            return Scalar
        if hasattr(node, "_size"):
            return node._size  # pyre-ignore
        t = type(node)
        if t in _always_scalar:
            result = Scalar
        elif t in _broadcast_the_inputs:
            result = _broadcast(*(self[i] for i in node.inputs))
        elif t in self._dispatch:
            result = self._dispatch[t](node)
        else:
            result = Unsized
        return result
