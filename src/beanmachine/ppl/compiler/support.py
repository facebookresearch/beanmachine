# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
import operator
from math import isnan
from typing import Callable, Dict

import beanmachine.ppl.compiler.bmg_nodes as bn
import torch
from beanmachine.ppl.compiler.sizer import Sizer
from beanmachine.ppl.compiler.typer_base import TyperBase
from beanmachine.ppl.utils.set_of_tensors import SetOfTensors
from torch import tensor

Infinite = SetOfTensors([])
TooBig = SetOfTensors([])
Unknown = SetOfTensors([])

positive_infinity = float("inf")


def _prod(x):
    """Compute the product of a sequence of values of arbitrary length"""
    return functools.reduce(operator.mul, x, 1)


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
    bn.EqualNode: torch.Tensor.eq,
    bn.ExpM1Node: torch.Tensor.expm1,  # pyre-ignore
    bn.ExpNode: torch.Tensor.exp,  # pyre-ignore
    bn.FloorDivNode: torch.Tensor.__floordiv__,
    bn.GreaterThanEqualNode: torch.Tensor.ge,
    bn.GreaterThanNode: torch.Tensor.gt,
    bn.InvertNode: torch.Tensor.__invert__,  # pyre-ignore
    bn.ItemNode: lambda x: x,  # item() is an identity
    bn.LessThanEqualNode: torch.Tensor.le,
    bn.LessThanNode: torch.Tensor.lt,
    bn.LogisticNode: torch.Tensor.sigmoid,
    bn.LogNode: torch.Tensor.log,
    bn.LShiftNode: torch.Tensor.__lshift__,
    bn.MatrixMultiplicationNode: torch.Tensor.mm,  # pyre-ignore
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
# Log1mexpNode
# IndexNode
# Log1mexpNode
#
# We will need to implement computation of the support
# of an arbitrary binomial distribution because samples are
# discrete values between 0 and count, which is typically small.
# Though implementing support computation if count is non-stochastic
# is straightforward, we do not yet have the gear to implement
# this for stochastic counts. Consider this contrived case:
#
# @bm.random_variable def a(): return Binomial(2, 0.5)
# @bm.random_variable def b(): return Binomial(a() + 1, 0.4)
# @bm.random_variable def c(i): return Normal(0.0, 2.0)
# @bm.random_variable def d(): return Normal(c(b()), 3.0)
#
# The support of a() is 0, 1, 2 -- easy.
#
# We need to know the support of b() in order to build the
# graph for d(). But how do we know the support of b()?
#
# What we must do is compute that the maximum possible value
# for a() + 1 is 3, and so the support of b() is 0, 1, 2, 3,
# and therefore there are four samples of c(i) generated.
#
# There are two basic ways to do this that immediately come to
# mind.
#
# The first is to simply ask the graph for the support of
# a() + 1, which we can generate, and then take the maximum
# value thus generated.
#
# If that turns out to be too expensive for some reason then
# we can write a bit of code that answers the question
# "what is the maximum value of your support?" and have each
# node implement that. However, that then introduces new
# problems; to compute the maximum value of a negation, for
# instance, we then would also need to answer the question
# "what is the minimum value you support?" and so on.


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
            bn.CategoricalLogitNode: self._support_categorical,
            bn.CategoricalNode: self._support_categorical,
            bn.SampleNode: self._support_sample,
            bn.SwitchNode: self._support_switch,
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
        p = _prod(s)
        if 2.0 ** p >= _limit:
            return TooBig
        return SetOfTensors(
            tensor(i).reshape(s) for i in itertools.product(*([[0.0, 1.0]] * p))
        )

    def _support_categorical(self, node: bn.CategoricalNodeBase) -> SetOfTensors:
        # Suppose we have something like Categorical(tensor([0.25, 0.25, 0.5])),
        # with size [3]. The support is 0, 1, 2.
        # If we have a higher dimensional categorical, say size [2, 3] such as
        # [[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]], then the support is
        # [0, 0], [0, 1], ... [2, 2], each is of size [2].
        #
        # That is: the range of values is determined by the last element
        # of the categorical input size, and the size of each member of
        # the support is the truncation of the last member off the categorical
        # input size.

        input_size = self._sizer[node.probability]
        if len(input_size) == 0:
            return Unknown
        max_element = input_size[-1]  # 3, in our example above.
        r = list(range(max_element))  # [0, 1, 2]
        result_size = input_size[:-1]  # [2] in our example above
        # In our example above we would have 3 ** 2 members of the
        # support. Compute how many members we're going to get and
        # bail out if it is too large.

        # TODO: Move this prod helper function out of bmg_nodes.py
        num_result_elements = _prod(result_size)
        if max_element ** num_result_elements >= _limit:
            return TooBig

        return SetOfTensors(
            tensor(i).reshape(result_size)
            for i in itertools.product(*([r] * num_result_elements))
        )

    def _support_sample(self, node: bn.SampleNode) -> SetOfTensors:
        return self[node.operand]

    def _support_switch(self, node: bn.SwitchNode) -> SetOfTensors:

        for i in range((len(node.inputs) - 1) // 2):
            if self[node.inputs[2 + i * 2]] == Infinite:
                return Infinite

        for i in range((len(node.inputs) - 1) // 2):
            if self[node.inputs[2 + i * 2]] == TooBig:
                return TooBig

        for i in range((len(node.inputs) - 1) // 2):
            if self[node.inputs[2 + i * 2]] == Unknown:
                return Unknown

        s = 0
        for i in range((len(node.inputs) - 1) // 2):
            s += len(self[node.inputs[2 + i * 2]])

        if s >= _limit:
            return TooBig

        return SetOfTensors(
            itertools.chain(
                *(
                    self[node.inputs[2 + i * 2]]
                    for i in range((len(node.inputs) - 1) // 2)
                )
            )
        )

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
