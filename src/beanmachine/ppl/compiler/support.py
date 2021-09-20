# Copyright (c) Facebook, Inc. and its affiliates.

# See notes in typer_base.py for how the type computation logic works.
#
# This typer computes all possible tensor values that a graph node can
# possibly have. For example, if we have a random variable:
#
# @rv def flips(n):
#   return Bernoulli(0.5)
#
# @functional def sumflips():
#    return flips(0) + flips(1)
#
# Then the sample nodes each have a two-value support {0, 1} --
# and the addition node has a support {0, 1, 2}.
#
# Some nodes -- a sample from a normal, for instance -- have infinite
# support; we mark those with a special value. Similarly, some nodes
# have finite but large support, where "large" is a parameter we can
# choose; to keep graphs relatively small we will refuse to compile
# a model where there are thousands of samples associated with a
# particular call. For example, suppose we have K categories:
#
# @rv def cat():
#   return Categorical(tensor([1, 1, 1, 1, 1, ...]))
#
# @rv def norm(n):
#   return Normal(0, 1)
#
# @functional toobig():
#   return whatever(cat())
#
# That model generates K normal samples; we want to restrict that
# to "small" K.


import functools
import itertools
from math import isnan
from typing import Callable, Dict

import beanmachine.ppl.compiler.bmg_nodes as bn
import torch
from beanmachine.ppl.compiler.bmg_nodes import SetOfTensors, positive_infinity
from beanmachine.ppl.compiler.sizer import Sizer
from beanmachine.ppl.compiler.typer_base import TyperBase
from torch import tensor

Infinite = SetOfTensors([])
TooBig = SetOfTensors([])
Unknown = SetOfTensors([])

_limit = 1000

_always_infinite = {
    bn.BetaNode,
    bn.Chi2Node,
    bn.DirichletNode,
    bn.FlatNode,
    bn.GammaNode,
    bn.HalfCauchyNode,
    bn.HalfNormalNode,
    bn.NormalNode,
    bn.PoissonNode,
    bn.StudentTNode,
    bn.UniformNode,
}

# TODO: We could do better for the comparison operators because we known
# that the support is not a function of the inputs; the support is just
# {True, False} for each element.  Handling this correctly would enable
# us to do stochastic control flows of the form some_rv(normal(1) > normal(2))
# even though the normal rvs have infinite supports.
#
# However, since BMG does not yet implement comparison operators, this is a
# moot point; if it ever does so, then revisit this decision.

_product_of_inputs = {
    bn.AdditionNode: torch.Tensor.__add__,
    bn.BitAndNode: torch.Tensor.__and__,
    bn.BitOrNode: torch.Tensor.__or__,
    bn.BitXorNode: torch.Tensor.__xor__,
    bn.DivisionNode: torch.Tensor.div,
    bn.ExpNode: torch.Tensor.exp,  # pyre-ignore
    bn.ExpM1Node: torch.Tensor.expm1,  # pyre-ignore
    bn.EqualNode: torch.Tensor.eq,
    bn.FloorDivNode: torch.Tensor.__floordiv__,
    bn.GreaterThanEqualNode: torch.Tensor.ge,
    bn.GreaterThanNode: torch.Tensor.gt,
    bn.InvertNode: torch.Tensor.__invert__,  # pyre-ignore
    bn.LessThanEqualNode: torch.Tensor.le,
    bn.LessThanNode: torch.Tensor.lt,
    bn.LogisticNode: torch.Tensor.sigmoid,
    bn.LogNode: torch.Tensor.log,
    bn.LShiftNode: torch.Tensor.__lshift__,
    bn.ModNode: torch.Tensor.__mod__,
    bn.MultiplicationNode: torch.Tensor.mul,
    bn.NegateNode: torch.Tensor.neg,
    bn.NotEqualNode: torch.Tensor.ne,
    bn.PhiNode: torch.distributions.Normal(0.0, 1.0).cdf,
    bn.PowerNode: torch.Tensor.pow,
    bn.RShiftNode: torch.Tensor.__rshift__,
}


# TODO:
#
# NotNode -- note that "not t" on a tensor is equivalent to "not Tensor.__bool__(t)"
# and produces either True or False. It is *not* the same as "Tensor.logical_not(t)"
# which executes "not" on each element and returns a tensor of the same size as t.
#
# LogSumExpTorchNode
# SwitchNode
# IndexNode
# MatrixMultiplicationNode
# Log1mexpNode


_nan = float("nan")


def _set_approximate_size(s: SetOfTensors) -> float:
    if s is Infinite:
        return positive_infinity
    if s is Unknown:
        return _nan
    if s is TooBig:
        return _limit
    return len(s)


def _set_product_approximate_size(x: float, y: SetOfTensors) -> float:
    # If either size is unknown (NaN), we return NaN.
    # Otherwise, if either size is infinite, we return infinite.
    # Otherwise, return the product.
    return x * _set_approximate_size(y)


class ComputeSupport(TyperBase[SetOfTensors]):

    _dispatch: Dict[type, Callable]
    _sizer: Sizer

    def __init__(self) -> None:
        TyperBase.__init__(self)
        self._sizer = Sizer()
        self._dispatch = {
            bn.BernoulliLogitNode: self._support_bernoulli,
            bn.BernoulliNode: self._support_bernoulli,
            bn.SampleNode: self._support_sample,
            bn.TensorNode: self._support_tensor,
        }

    def _product(self, f: Callable, *nodes: bn.BMGNode) -> SetOfTensors:
        # * We have a sequence of nodes n[0], n[1], ... n[k-1].
        #
        # * Each of those nodes has possible values t[x][0], t[x][1] ...
        #   for x from 0 to k-1.
        #
        # * We have a function f which takes k tensors and returns a tensor.
        #
        # * We wish to compute the set:
        #
        # {
        #   f(t[0][0], t[1][0], ... t[k-1][0]),
        #   f(t[0][1], t[1][0], ... t[k-1][0]),
        #   ...
        # }
        #
        # That is, the Cartesian product of all possible combinations of
        # values of each node, with each element of the product run through
        # function f.
        #
        # However, we have some problems:
        #
        # * The support of a node might be infinite.
        # * The support of a node might be unknown.
        # * The support of a node might be finite but large.
        # * The size of the product might be finite but large.
        #
        # In those cases we want to return special values Infinite, Unknown
        # or TooBig rather than wasting time and memory to compute the set.
        #
        # ####
        #
        # First thing to do is determine the *approximate* size of the
        # Cartesian product of possible values.
        #
        # TODO: This approximation is an over-estimate; for instance, when
        # multiplying {0 or 1} by { n elements} we assume that the resulting
        # set has up to 2n elements, when in fact it only has n or n+1 elements.
        # Would it be simpler and more accurate to instead make a loop, accumulate
        # the result into a mutable deduplicating set, and if the set ever gets
        # too big, bail out then?

        size = functools.reduce(
            lambda acc, node: _set_product_approximate_size(acc, self[node]), nodes, 1.0
        )

        if size == positive_infinity:
            return Infinite
        if isnan(size):
            return Unknown
        if size >= _limit:
            return TooBig

        # If we've made it here then every node had a small support
        # and the product of the approximate sizes was small too.
        # We can just take the Cartesian product and call f.

        return SetOfTensors(f(c) for c in itertools.product(*(self[n] for n in nodes)))

    def _support_bernoulli(self, node: bn.BernoulliBase) -> SetOfTensors:
        # The support of a Bernoulli only depends on the shape of its input,
        # not on the content of that input. Suppose we have a Bernoulli
        # of shape [1, 2, 3]; there are 1x2x3 = 6 values in the output
        # each of which is either 0 or 1, so that's 64 possibilities. If
        # we have too big a shape then just bail out rather than handling
        # thousands or millions of possibilities.
        s = self._sizer[node]
        p = bn.prod(s)
        if 2.0 ** p >= _limit:
            return TooBig
        return SetOfTensors(
            tensor(i).reshape(s) for i in itertools.product(*([[0.0, 1.0]] * p))
        )

    def _support_sample(self, node: bn.SampleNode) -> SetOfTensors:
        return self[node.operand]

    def _support_tensor(self, node: bn.TensorNode) -> SetOfTensors:
        return self._product(tensor, *node.inputs)

    # This implements the abstract base type method.
    def _compute_type_inputs_known(self, node: bn.BMGNode) -> SetOfTensors:
        if isinstance(node, bn.ConstantNode):
            return SetOfTensors([node.value])
        t = type(node)
        if t in _always_infinite:
            result = Infinite
        elif t in _product_of_inputs:
            result = self._product(lambda x: _product_of_inputs[t](*x), *node.inputs)
        elif t in self._dispatch:
            result = self._dispatch[t](node)
        else:
            result = Unknown

        return result
