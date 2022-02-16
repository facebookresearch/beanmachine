#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# BM -> BMG compiler arithmetic tests

import math
import operator
import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli, Beta, Binomial, HalfCauchy, Normal


@bm.random_variable
def bern():
    return Bernoulli(0.5)


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def norm():
    return Normal(0.0, 1.0)


@bm.random_variable
def hc():
    return HalfCauchy(1.0)


@bm.functional
def expm1_prob():
    return beta().expm1()


@bm.functional
def expm1_real():
    return torch.expm1(norm())


@bm.functional
def expm1_negreal():
    return torch.Tensor.expm1(-hc())


@bm.functional
def logistic_prob():
    return beta().sigmoid()


@bm.functional
def logistic_real():
    return torch.sigmoid(norm())


@bm.functional
def logistic_negreal():
    return torch.Tensor.sigmoid(-hc())


@bm.random_variable
def ordinary_arithmetic(n):
    return Bernoulli(
        probs=torch.tensor(0.5) + torch.log(torch.exp(n * torch.tensor(0.1)))
    )


@bm.random_variable
def stochastic_arithmetic():
    s = 0.0
    # Verify that mutating += works on lists normally:
    items = [0]
    items += [1]
    # Verify that +=, *=, -= all work on graph nodes:
    for n in items:
        p = torch.log(torch.tensor(0.01))
        p *= ordinary_arithmetic(n)
        s += p
    m = 1
    m -= torch.exp(input=torch.log(torch.tensor(0.99)) + s)
    return Bernoulli(m)


@bm.functional
def mutating_assignments():
    # Torch supports mutating tensors in-place, which allows for
    # aliasing. THE COMPILER DOES NOT CORRECTLY DETECT ALIASING
    # WHEN A STOCHASTIC QUANTITY IS INVOLVED!
    x = torch.tensor(1.0)
    y = x  # y is an alias for x
    y += 2.0  # y is now 3, and so is x
    y = y + 4.0  # y is now 7, but x is still 3
    # So far we're all fine; every mutated tensor has been non-stochastic.
    b = beta() * x + y  # b is beta_sample * 3 + 7
    # Now let's see how things go wrong. We'll alias stochastic quantity b:
    c = b
    c *= 5.0
    # In Python Bean Machine, c and b are now both (beta() * 3 + 7) * 5
    # but the compiler does not detect that c and b are aliases, and does
    # not represent tensor mutations in graph nodes. The compiler thinks
    # that c is (beta() * 3 + 7) * 5 but b is still (beta() * 3 + 7):
    return b


@bm.random_variable
def neg_of_neg():
    return Normal(-torch.neg(norm()), 1.0)


@bm.functional
def subtractions():
    # Show that we can handle a bunch of different ways to subtract things
    # Show that unary plus operations are discarded.
    n = +norm()
    b = +beta()
    h = +hc()
    return +torch.sub(+n.sub(+b), +b - h)


@bm.random_variable
def bino():
    return Binomial(total_count=3, probs=0.5)


@bm.functional
def unsupported_invert():
    return ~bino()


@bm.functional
def unsupported_bitxor():
    return bino() ^ bino()


@bm.functional
def unsupported_add():
    # What happens if we use a stochastic quantity in an operation with
    # a non-tensor, non-number?
    return bino() + "foo"


@bm.functional
def log_1():
    # Ordinary constant, math.log. Note that a functional is
    # required to return a tensor. Verify that ordinary
    # arithmetic still works in a model.
    return torch.tensor(math.log(1.0))


@bm.functional
def log_2():
    # Tensor constant, math.log; this is legal.
    # A multi-valued tensor would be an error.
    return torch.tensor(math.log(torch.tensor(2.0)))


@bm.functional
def log_3():
    # Tensor constant, Tensor.log.
    # An ordinary constant would be an error.
    return torch.Tensor.log(torch.tensor(3.0))


@bm.functional
def log_4():
    # Tensor constant, instance log
    return torch.tensor([4.0, 4.0]).log()


@bm.functional
def log_5():
    # Stochastic value, math.log
    return torch.tensor(math.log(beta() + 5.0))


@bm.functional
def log_6():
    # Stochastic value, Tensor.log
    return torch.Tensor.log(beta() + 6.0)


@bm.functional
def log_7():
    # Stochastic value, instance log
    return (beta() + 7.0).log()


@bm.functional
def exp_1():
    # Ordinary constant, math.exp. Note that a functional is
    # required to return a tensor. Verify that ordinary
    # arithmetic still works in a model.
    return torch.tensor(math.exp(1.0))


@bm.functional
def exp_2():
    # Tensor constant, math.exp; this is legal.
    # A multi-valued tensor would be an error.
    return torch.tensor(math.exp(torch.tensor(2.0)))


@bm.functional
def exp_3():
    # Tensor constant, Tensor.exp.
    # An ordinary constant would be an error.
    return torch.Tensor.exp(torch.tensor(3.0))


@bm.functional
def exp_4():
    # Tensor constant, instance exp
    return torch.tensor([4.0, 4.0]).exp()


@bm.functional
def exp_5():
    # Stochastic value, math.exp
    return torch.tensor(math.exp(beta() + 5.0))


@bm.functional
def exp_6():
    # Stochastic value, Tensor.exp
    return torch.Tensor.exp(beta() + 6.0)


@bm.functional
def exp_7():
    # Stochastic value, instance exp
    return (beta() + 7.0).exp()


@bm.functional
def pow_1():
    # Ordinary constant, power operator. Note that a functional is
    # required to return a tensor. Verify that ordinary
    # arithmetic still works in a model.
    return torch.tensor(1.0 ** 10.0)


@bm.functional
def pow_2():
    # Tensor constant, power operator.
    return torch.tensor(2.0) ** 2.0


@bm.functional
def pow_3():
    # Tensor constant, Tensor.pow, named argument.
    return torch.Tensor.pow(torch.tensor(3.0), exponent=torch.tensor(3.0))


@bm.functional
def pow_4():
    # Tensor constant, instance pow, named argument
    return torch.tensor(4.0).pow(exponent=torch.tensor(4.0))


@bm.functional
def pow_5():
    # Stochastic value, power operator
    return beta() ** 5.0


@bm.functional
def pow_6():
    # Stochastic value, Tensor.pow
    return torch.Tensor.pow(torch.tensor(6.0), exponent=beta())


@bm.functional
def pow_7():
    # Stochastic value, instance exp
    return torch.tensor(7.0).pow(exponent=beta())


@bm.functional
def pow_8():
    # Constant values, operator.pow
    return operator.pow(torch.tensor(8.0), torch.tensor(2.0))


@bm.functional
def pow_9():
    # Stochastic values, operator.pow
    return operator.pow(beta(), torch.tensor(9.0))


@bm.functional
def to_real_1():
    # Calling float() causes a TO_REAL node to be emitted into the graph.
    # TODO: Is this actually a good idea? We already automatically insert
    # TO_REAL when necessary to make the type system happy. float() could
    # just be an identity on graph nodes instead of adding TO_REAL.
    #
    # Once again, a functional is required to return a tensor.
    return torch.tensor([float(bern()), 1.0])


@bm.functional
def to_real_2():
    # Similarly for the tensor float() instance method.
    return bern().float()


@bm.functional
def not_1():
    # Ordinary constant, not operator. Note that a functional is
    # required to return a tensor. Verify that ordinary
    # arithmetic still works in a model.
    return torch.tensor(not 1.0)


@bm.functional
def not_2():
    # Tensor constant; not operator. This is legal.
    # A multi-valued tensor would be an error.
    return torch.tensor(not torch.tensor(2.0))


@bm.functional
def not_3():
    # Tensor constant, Tensor.logical_not.
    # An ordinary constant would be an error.
    return torch.Tensor.logical_not(torch.tensor(3.0))


@bm.functional
def not_4():
    # Tensor constant, instance logical_not
    return torch.tensor(4.0).logical_not()


@bm.functional
def not_5():
    # Stochastic value, not operator
    return torch.tensor(not (beta() + 5.0))


@bm.functional
def not_6():
    # Stochastic value, Tensor.logical_not
    return torch.Tensor.logical_not(beta() + 6.0)


@bm.functional
def not_7():
    # Stochastic value, instance logical_not
    return (beta() + 7.0).logical_not()


@bm.functional
def not_8():
    # Constant value, operator.not_
    return torch.tensor(operator.not_(torch.tensor(8.0)))


@bm.functional
def not_9():
    # Stochastic value, operator.not_
    return torch.tensor(operator.not_(beta() + 9.0))


@bm.functional
def neg_1():
    # Ordinary constant, - operator. Note that a functional is
    # required to return a tensor. Verify that ordinary
    # arithmetic still works in a model.
    return torch.tensor(-1.0)


@bm.functional
def neg_2():
    # Tensor constant; - operator.
    return -torch.tensor(2.0)


@bm.functional
def neg_3():
    # Tensor constant, Tensor.neg.
    return torch.Tensor.neg(torch.tensor(3.0))


@bm.functional
def neg_4():
    # Tensor constant, instance neg
    return torch.tensor(4.0).neg()


@bm.functional
def neg_5():
    # Stochastic value, - operator
    return -(beta() + 5.0)


@bm.functional
def neg_6():
    # Stochastic value, Tensor.neg.
    # TODO: "negative" is a synonym; make it work too.
    return torch.Tensor.neg(beta() + 6.0)


@bm.functional
def neg_7():
    # Stochastic value, instance neg
    # TODO: "negative" is a synonym; make it work too.
    return (beta() + 7.0).neg()


@bm.functional
def neg_8():
    # Constant value, operator.neg
    return operator.neg(torch.tensor(8.0))


@bm.functional
def neg_9():
    # Stochastic value, operator.neg
    return operator.neg(beta() + 9.0)


@bm.functional
def add_1():
    # Ordinary arithmetic, + operator
    return torch.tensor(1.0 + 1.0)


@bm.functional
def add_2():
    # Tensor arithmetic, + operator
    return torch.tensor(2.0) + torch.tensor(2.0)


@bm.functional
def add_3():
    # Tensor constants, Tensor.add.
    # TODO: Tensor.add takes an optional third argument with the semantics
    # add(a, b, c) --> a + b * c. Test that as well.
    return torch.Tensor.add(torch.tensor(3.0), torch.tensor(3.0))


@bm.functional
def add_4():
    # Tensor constant, instance add
    # TODO: Tensor.add takes an optional third argument with the semantics
    # a.add(b, c) --> a + b * c. Test that as well.
    return torch.tensor(4.0).add(torch.tensor(4.0))


@bm.functional
def add_5():
    # Stochastic value, + operator
    return beta() + 5.0


@bm.functional
def add_6():
    # Stochastic value, Tensor.add.
    return torch.Tensor.add(beta(), torch.tensor(6.0))


@bm.functional
def add_7():
    # Stochastic value, instance add
    return beta().add(torch.tensor(7.0))


@bm.functional
def add_8():
    # Constant values, operator.add
    return operator.add(torch.tensor(8.0), torch.tensor(8.0))


@bm.functional
def add_9():
    # Stochastic values, operator.add
    return operator.add(beta(), torch.tensor(9.0))


@bm.functional
def and_1():
    # Ordinary arithmetic, & operator
    return torch.tensor(1 & 3)


@bm.functional
def and_2():
    # Tensor arithmetic, & operator
    return torch.tensor(6) & torch.tensor(2)


@bm.functional
def and_3():
    # Tensor constants, Tensor.bitwise_and.
    return torch.Tensor.bitwise_and(torch.tensor(7), torch.tensor(3))


@bm.functional
def and_4():
    # Tensor constant, instance bitwise_and
    return torch.tensor(7).bitwise_and(torch.tensor(4))


@bm.functional
def and_5():
    # Stochastic value, & operator
    return beta() & 2


@bm.functional
def and_6():
    # Stochastic value, Tensor.bitwise_and
    return torch.Tensor.bitwise_and(beta(), torch.tensor(4))


@bm.functional
def and_7():
    # Stochastic value, instance bitwise_and
    return beta().bitwise_and(torch.tensor(8))


@bm.functional
def and_8():
    # Constant values, operator.and_
    return operator.and_(torch.tensor(15), torch.tensor(8))


@bm.functional
def and_9():
    # Stochastic values, operator.and_
    return operator.and_(beta(), torch.tensor(16))


@bm.functional
def div_1():
    # Ordinary arithmetic, / operator
    return torch.tensor(1.0 / 1.0)


@bm.functional
def div_2():
    # Tensor arithmetic, / operator
    return torch.tensor(4.0) / torch.tensor(2.0)


@bm.functional
def div_3():
    # Tensor constants, Tensor.div.
    # TODO: div also takes an optional rounding flag; test that.
    return torch.Tensor.div(torch.tensor(6.0), torch.tensor(2.0))


@bm.functional
def div_4():
    # Tensor constant, instance divide (a synonym).
    return torch.tensor(8.0).divide(torch.tensor(2.0))


@bm.functional
def div_5():
    # Stochastic value, / operator
    return beta() / 2.0


@bm.functional
def div_6():
    # Stochastic value, Tensor.true_divide (a synonym)
    return torch.Tensor.true_divide(beta(), torch.tensor(4.0))


@bm.functional
def div_7():
    # Stochastic value, instance div
    return beta().div(torch.tensor(8.0))


@bm.functional
def div_8():
    # Constant values, operator.truediv
    return operator.truediv(torch.tensor(16.0), torch.tensor(2.0))


@bm.functional
def div_9():
    # Stochastic values, operator.truediv
    return operator.truediv(beta(), torch.tensor(16.0))


@bm.functional
def floordiv_1():
    # Ordinary arithmetic, // operator
    return torch.tensor(1.0 // 1.0)


@bm.functional
def floordiv_2():
    # Tensor arithmetic, // operator
    return torch.tensor(4.0) // torch.tensor(2.0)


@bm.functional
def floordiv_3():
    # Tensor constants, Tensor.floor_divide.
    return torch.Tensor.floor_divide(torch.tensor(6.0), torch.tensor(2.0))


@bm.functional
def floordiv_4():
    # Tensor constant, instance floor_divide
    return torch.tensor(8.0).floor_divide(torch.tensor(2.0))


@bm.functional
def floordiv_5():
    # Stochastic value, // operator
    return beta() // 4.0


@bm.functional
def floordiv_6():
    # Stochastic value, Tensor.floor_divide
    return torch.Tensor.floor_divide(beta(), torch.tensor(8.0))


@bm.functional
def floordiv_7():
    # Stochastic value, instance floor_divide
    return beta().floor_divide(torch.tensor(16.0))


@bm.functional
def floordiv_8():
    # Constant values, operator.floordiv
    return operator.floordiv(torch.tensor(16.0), torch.tensor(2.0))


@bm.functional
def floordiv_9():
    # Stochastic values, operator.floordiv
    return operator.floordiv(beta(), torch.tensor(32.0))


@bm.functional
def inv_1():
    # Ordinary constant, ~ operator.
    return torch.tensor(~1)


@bm.functional
def inv_2():
    # Tensor constant; ~ operator.
    return ~torch.tensor(2)


@bm.functional
def inv_3():
    # Tensor constant, Tensor.bitwise_not.
    return torch.Tensor.bitwise_not(torch.tensor(3))


@bm.functional
def inv_4():
    # Tensor constant, instance bitwise_not
    return torch.tensor(4).bitwise_not()


@bm.functional
def inv_5():
    # Stochastic value, ~ operator
    return ~(beta() + 5.0)


@bm.functional
def inv_6():
    # Stochastic value, Tensor.bitwise_not
    return torch.Tensor.bitwise_not(beta() + 6.0)


@bm.functional
def inv_7():
    # Stochastic value, instance bitwise_not
    return (beta() + 7.0).bitwise_not()


@bm.functional
def inv_8():
    # Constant value, operator.inv
    return operator.inv(torch.tensor(8))


@bm.functional
def inv_9():
    # Stochastic value, operator.inv
    return operator.inv(beta())


@bm.functional
def lshift_1():
    # Ordinary arithmetic, << operator
    return torch.tensor(1 << 1)


@bm.functional
def lshift_2():
    # Tensor arithmetic, << operator
    return torch.tensor(2.0) << torch.tensor(2.0)


@bm.functional
def lshift_3():
    # Tensor constants, Tensor.bitwise_left_shift.
    return torch.Tensor.bitwise_left_shift(torch.tensor(6.0), torch.tensor(2.0))


@bm.functional
def lshift_4():
    # Tensor constant, instance bitwise_left_shift
    return torch.tensor(8.0).bitwise_left_shift(torch.tensor(2.0))


@bm.functional
def lshift_5():
    # Stochastic value, << operator
    return beta() << 4.0


@bm.functional
def lshift_6():
    # Stochastic value, Tensor.bitwise_left_shift
    return torch.Tensor.bitwise_left_shift(beta(), torch.tensor(8.0))


@bm.functional
def lshift_7():
    # Stochastic value, instance bitwise_left_shift
    return beta().bitwise_left_shift(torch.tensor(16.0))


@bm.functional
def lshift_8():
    # Constant values, operator.lshift
    return operator.lshift(torch.tensor(16.0), torch.tensor(2.0))


@bm.functional
def lshift_9():
    # Stochastic values, operator.lshift
    return operator.lshift(beta(), torch.tensor(32.0))


@bm.functional
def mod_1():
    # Ordinary arithmetic, % operator
    return torch.tensor(1 % 1)


@bm.functional
def mod_2():
    # Tensor arithmetic, % operator
    return torch.tensor(5.0) % torch.tensor(3.0)


@bm.functional
def mod_3():
    # Tensor constants, Tensor.fmod.
    return torch.Tensor.fmod(torch.tensor(11.0), torch.tensor(4.0))


@bm.functional
def mod_4():
    # Tensor constant, instance remainder (a near synonym).
    return torch.tensor(9.0).remainder(torch.tensor(5.0))


@bm.functional
def mod_5():
    # Stochastic value, % operator
    return beta() % 5.0


@bm.functional
def mod_6():
    # Stochastic value, Tensor.fmod
    return torch.Tensor.fmod(beta(), torch.tensor(6.0))


@bm.functional
def mod_7():
    # Stochastic value, instance fmod
    return beta().fmod(torch.tensor(7.0))


@bm.functional
def mod_8():
    # Constant values, operator.mod
    return operator.mod(torch.tensor(17.0), torch.tensor(9.0))


@bm.functional
def mod_9():
    # Stochastic values, operator.mod
    return operator.mod(beta(), torch.tensor(9.0))


@bm.functional
def mul_1():
    # Ordinary arithmetic, * operator
    return torch.tensor(1.0 * 1.0)


@bm.functional
def mul_2():
    # Tensor arithmetic, * operator
    return torch.tensor(2.0) * torch.tensor(2.0)


@bm.functional
def mul_3():
    # Tensor constants, Tensor.mul.
    return torch.Tensor.mul(torch.tensor(3.0), torch.tensor(3.0))


@bm.functional
def mul_4():
    # Tensor constant, instance multiply (a synonym).
    return torch.tensor(4.0).multiply(torch.tensor(4.0))


@bm.functional
def mul_5():
    # Stochastic value, * operator
    return beta() * 5.0


@bm.functional
def mul_6():
    # Stochastic value, Tensor.multiply.
    return torch.Tensor.multiply(beta(), torch.tensor(6.0))


@bm.functional
def mul_7():
    # Stochastic value, instance mul
    return beta().mul(torch.tensor(7.0))


@bm.functional
def mul_8():
    # Constant values, operator.mul
    return operator.mul(torch.tensor(8.0), torch.tensor(8.0))


@bm.functional
def mul_9():
    # Stochastic values, operator.mul
    return operator.mul(beta(), torch.tensor(9.0))


@bm.functional
def or_1():
    # Ordinary arithmetic, | operator
    return torch.tensor(1 | 3)


@bm.functional
def or_2():
    # Tensor arithmetic, | operator
    return torch.tensor(6) | torch.tensor(2)


@bm.functional
def or_3():
    # Tensor constants, Tensor.bitwise_or.
    return torch.Tensor.bitwise_or(torch.tensor(7), torch.tensor(3))


@bm.functional
def or_4():
    # Tensor constant, instance bitwise_or
    return torch.tensor(7).bitwise_or(torch.tensor(4))


@bm.functional
def or_5():
    # Stochastic value, | operator
    return beta() | 2


@bm.functional
def or_6():
    # Stochastic value, Tensor.bitwise_or
    return torch.Tensor.bitwise_or(beta(), torch.tensor(4))


@bm.functional
def or_7():
    # Stochastic value, instance bitwise_or
    return beta().bitwise_or(torch.tensor(8))


@bm.functional
def or_8():
    # Constant values, operator.or_
    return operator.or_(torch.tensor(15), torch.tensor(8))


@bm.functional
def or_9():
    # Stochastic values, operator.or_
    return operator.or_(beta(), torch.tensor(16))


@bm.functional
def pos_1():
    # Ordinary constant, + operator.
    return torch.tensor(+1.0)


@bm.functional
def pos_2():
    # Tensor constant; + operator.
    return +torch.tensor(2.0)


@bm.functional
def pos_5():
    # Stochastic value, + operator
    return +(beta() + 5.0)


@bm.functional
def pos_8():
    # Constant value, operator.pos
    return operator.pos(torch.tensor(8.0))


@bm.functional
def pos_9():
    # Stochastic value, operator.pos
    return operator.pos(beta() + 9.0)


@bm.functional
def rshift_1():
    # Ordinary arithmetic, >> operator
    return torch.tensor(2 >> 1)


@bm.functional
def rshift_2():
    # Tensor arithmetic, << operator
    return torch.tensor(4.0) >> torch.tensor(2.0)


@bm.functional
def rshift_3():
    # Tensor constants, Tensor.bitwise_right_shift.
    return torch.Tensor.bitwise_right_shift(torch.tensor(6.0), torch.tensor(2.0))


@bm.functional
def rshift_4():
    # Tensor constant, instance bitwise_right_shift
    return torch.tensor(8.0).bitwise_right_shift(torch.tensor(2.0))


@bm.functional
def rshift_5():
    # Stochastic value, >> operator
    return beta() >> 4.0


@bm.functional
def rshift_6():
    # Stochastic value, Tensor.bitwise_right_shift
    return torch.Tensor.bitwise_right_shift(beta(), torch.tensor(8.0))


@bm.functional
def rshift_7():
    # Stochastic value, instance bitwise_right_shift
    return beta().bitwise_right_shift(torch.tensor(16.0))


@bm.functional
def rshift_8():
    # Constant values, operator.rshift
    return operator.rshift(torch.tensor(16.0), torch.tensor(2.0))


@bm.functional
def rshift_9():
    # Stochastic values, operator.lshift
    return operator.rshift(beta(), torch.tensor(32.0))


@bm.functional
def sub_1():
    # Ordinary arithmetic, - operator
    return torch.tensor(2.0 - 1.0)


@bm.functional
def sub_2():
    # Tensor arithmetic, - operator
    return torch.tensor(5.0) - torch.tensor(2.0)


@bm.functional
def sub_3():
    # Tensor constants, Tensor.sub.
    # TODO: Tensor.sub takes an optional third argument with the semantics
    # sub(a, b, c) --> a - b * c. Test that as well.
    return torch.Tensor.sub(torch.tensor(6.0), torch.tensor(3.0))


@bm.functional
def sub_4():
    # Tensor constant, instance add
    # TODO: Tensor.add takes an optional third argument with the semantics
    # a.sub(b, c) --> a - b * c. Test that as well.
    return torch.tensor(8.0).sub(torch.tensor(4.0))


@bm.functional
def sub_5():
    # Stochastic value, - operator
    return beta() - 5.0


@bm.functional
def sub_6():
    # Stochastic value, Tensor.subtract (a synonym)
    return torch.Tensor.subtract(beta(), torch.tensor(6.0))


@bm.functional
def sub_7():
    # Stochastic value, instance sub
    return beta().sub(torch.tensor(7.0))


@bm.functional
def sub_8():
    # Constant values, operator.sub
    return operator.sub(torch.tensor(16.0), torch.tensor(8.0))


@bm.functional
def sub_9():
    # Stochastic values, operator.sub
    return operator.sub(beta(), torch.tensor(9.0))


@bm.functional
def xor_1():
    # Ordinary arithmetic, ^ operator
    return torch.tensor(1 ^ 3)


@bm.functional
def xor_2():
    # Tensor arithmetic, ^ operator
    return torch.tensor(6) ^ torch.tensor(2)


@bm.functional
def xor_3():
    # Tensor constants, Tensor.bitwise_xor.
    return torch.Tensor.bitwise_xor(torch.tensor(7), torch.tensor(3))


@bm.functional
def xor_4():
    # Tensor constant, instance bitwise_xor
    return torch.tensor(7).bitwise_xor(torch.tensor(4))


@bm.functional
def xor_5():
    # Stochastic value, ^ operator
    return beta() ^ 2


@bm.functional
def xor_6():
    # Stochastic value, Tensor.bitwise_xor
    return torch.Tensor.bitwise_xor(beta(), torch.tensor(4))


@bm.functional
def xor_7():
    # Stochastic value, instance bitwise_xor
    return beta().bitwise_xor(torch.tensor(8))


@bm.functional
def xor_8():
    # Constant values, operator.xor
    return operator.xor(torch.tensor(15), torch.tensor(8))


@bm.functional
def xor_9():
    # Stochastic values, operator.xor
    return operator.xor(beta(), torch.tensor(16))


class BMGArithmeticTest(unittest.TestCase):
    def test_bmg_arithmetic_logical_not(self) -> None:
        self.maxDiff = None

        # "not" operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            not_1(),
            not_2(),
            not_3(),
            not_4(),
            not_5(),
            not_6(),
            not_7(),
            not_8(),
            not_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a not operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a not operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a not operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a not operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_float(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([to_real_1(), to_real_2()], {})
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=2];
  N4[label=1];
  N5[label=ToReal];
  N6[label=1.0];
  N7[label=ToMatrix];
  N8[label=Query];
  N9[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N5;
  N3 -> N7;
  N4 -> N7;
  N5 -> N7;
  N5 -> N9;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_log(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                log_1(),
                log_2(),
                log_3(),
                log_4(),
                log_5(),
                log_6(),
                log_7(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=Query];
  N02[label=0.6931471824645996];
  N03[label=Query];
  N04[label=1.0986123085021973];
  N05[label=Query];
  N06[label="[1.3862943649291992,1.3862943649291992]"];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=5.0];
  N13[label="+"];
  N14[label=Log];
  N15[label=Query];
  N16[label=6.0];
  N17[label="+"];
  N18[label=Log];
  N19[label=Query];
  N20[label=7.0];
  N21[label="+"];
  N22[label=Log];
  N23[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N11 -> N17;
  N11 -> N21;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
}"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_pow(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                pow_1(),
                pow_2(),
                pow_3(),
                pow_4(),
                pow_5(),
                pow_6(),
                pow_7(),
                pow_8(),
                pow_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=Query];
  N02[label=4.0];
  N03[label=Query];
  N04[label=27.0];
  N05[label=Query];
  N06[label=256.0];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=5.0];
  N12[label="**"];
  N13[label=Query];
  N14[label=6.0];
  N15[label=ToPosReal];
  N16[label="**"];
  N17[label=Query];
  N18[label=7.0];
  N19[label="**"];
  N20[label=Query];
  N21[label=64.0];
  N22[label=Query];
  N23[label=9.0];
  N24[label="**"];
  N25[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N12;
  N10 -> N15;
  N10 -> N24;
  N11 -> N12;
  N12 -> N13;
  N14 -> N16;
  N15 -> N16;
  N15 -> N19;
  N16 -> N17;
  N18 -> N19;
  N19 -> N20;
  N21 -> N22;
  N23 -> N24;
  N24 -> N25;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_neg(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                neg_1(),
                neg_2(),
                neg_3(),
                neg_4(),
                neg_5(),
                neg_6(),
                neg_7(),
                neg_8(),
                neg_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=-1.0];
  N01[label=Query];
  N02[label=-2.0];
  N03[label=Query];
  N04[label=-3.0];
  N05[label=Query];
  N06[label=-4.0];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=5.0];
  N13[label="+"];
  N14[label="-"];
  N15[label=Query];
  N16[label=6.0];
  N17[label="+"];
  N18[label="-"];
  N19[label=Query];
  N20[label=7.0];
  N21[label="+"];
  N22[label="-"];
  N23[label=Query];
  N24[label=-8.0];
  N25[label=Query];
  N26[label=9.0];
  N27[label="+"];
  N28[label="-"];
  N29[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N11 -> N17;
  N11 -> N21;
  N11 -> N27;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N24 -> N25;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_add(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                add_1(),
                add_2(),
                add_3(),
                add_4(),
                add_5(),
                add_6(),
                add_7(),
                add_8(),
                add_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Query];
  N02[label=4.0];
  N03[label=Query];
  N04[label=6.0];
  N05[label=Query];
  N06[label=8.0];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=5.0];
  N13[label="+"];
  N14[label=Query];
  N15[label=6.0];
  N16[label="+"];
  N17[label=Query];
  N18[label=7.0];
  N19[label="+"];
  N20[label=Query];
  N21[label=16.0];
  N22[label=Query];
  N23[label=9.0];
  N24[label="+"];
  N25[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N11 -> N16;
  N11 -> N19;
  N11 -> N24;
  N12 -> N13;
  N13 -> N14;
  N15 -> N16;
  N16 -> N17;
  N18 -> N19;
  N19 -> N20;
  N21 -> N22;
  N23 -> N24;
  N24 -> N25;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_and(self) -> None:
        self.maxDiff = None

        # & operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            and_1(),
            and_2(),
            and_3(),
            and_4(),
            and_5(),
            and_6(),
            and_7(),
            and_8(),
            and_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a & operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a & operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a & operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a & operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_div(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                div_1(),
                div_2(),
                div_3(),
                div_4(),
                div_5(),
                div_6(),
                div_7(),
                div_8(),
                div_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=Query];
  N02[label=2.0];
  N03[label=Query];
  N04[label=3.0];
  N05[label=Query];
  N06[label=4.0];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=0.5];
  N12[label="*"];
  N13[label=Query];
  N14[label=0.25];
  N15[label="*"];
  N16[label=Query];
  N17[label=0.125];
  N18[label="*"];
  N19[label=Query];
  N20[label=8.0];
  N21[label=Query];
  N22[label=0.0625];
  N23[label="*"];
  N24[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N12;
  N10 -> N15;
  N10 -> N18;
  N10 -> N23;
  N11 -> N12;
  N12 -> N13;
  N14 -> N15;
  N15 -> N16;
  N17 -> N18;
  N18 -> N19;
  N20 -> N21;
  N22 -> N23;
  N23 -> N24;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_floordiv(self) -> None:

        self.skipTest(
            "Disabling floordiv tests; produces a deprecation warning in torch."
        )

        self.maxDiff = None

        # "floordiv" operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            floordiv_1(),
            floordiv_2(),
            floordiv_3(),
            floordiv_4(),
            floordiv_5(),
            floordiv_6(),
            floordiv_7(),
            floordiv_8(),
            floordiv_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a // operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a // operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a // operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a // operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_inv(self) -> None:
        self.maxDiff = None

        # ~ operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            inv_1(),
            inv_2(),
            inv_3(),
            inv_4(),
            inv_5(),
            inv_6(),
            inv_7(),
            inv_8(),
            inv_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        # TODO: This is weirdly capitalized and ungrammatical. Fix the error message.
        expected = """
The model uses a Invert operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a Invert operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a Invert operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a Invert operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_lshift(self) -> None:
        self.maxDiff = None

        # << operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            lshift_1(),
            lshift_2(),
            lshift_3(),
            lshift_4(),
            lshift_5(),
            lshift_6(),
            lshift_7(),
            lshift_8(),
            lshift_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a << operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a << operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a << operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a << operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_mod(self) -> None:
        self.maxDiff = None

        # % operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            mod_1(),
            mod_2(),
            mod_3(),
            mod_4(),
            mod_5(),
            mod_6(),
            mod_7(),
            mod_8(),
            mod_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a % operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a % operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a % operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a % operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_mul(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                mul_1(),
                mul_2(),
                mul_3(),
                mul_4(),
                mul_5(),
                mul_6(),
                mul_7(),
                mul_8(),
                mul_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=Query];
  N02[label=4.0];
  N03[label=Query];
  N04[label=9.0];
  N05[label=Query];
  N06[label=16.0];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=5.0];
  N13[label="*"];
  N14[label=Query];
  N15[label=6.0];
  N16[label="*"];
  N17[label=Query];
  N18[label=7.0];
  N19[label="*"];
  N20[label=Query];
  N21[label=64.0];
  N22[label=Query];
  N23[label=9.0];
  N24[label="*"];
  N25[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N11 -> N16;
  N11 -> N19;
  N11 -> N24;
  N12 -> N13;
  N13 -> N14;
  N15 -> N16;
  N16 -> N17;
  N18 -> N19;
  N19 -> N20;
  N21 -> N22;
  N23 -> N24;
  N24 -> N25;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_or(self) -> None:
        self.maxDiff = None

        # & operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            or_1(),
            or_2(),
            or_3(),
            or_4(),
            or_5(),
            or_6(),
            or_7(),
            or_8(),
            or_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a | operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a | operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a | operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a | operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_rshift(self) -> None:
        self.maxDiff = None

        # >> operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            rshift_1(),
            rshift_2(),
            rshift_3(),
            rshift_4(),
            rshift_5(),
            rshift_6(),
            rshift_7(),
            rshift_8(),
            rshift_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a >> operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a >> operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a >> operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a >> operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_pos(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                pos_1(),
                pos_2(),
                pos_5(),
                pos_8(),
                pos_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=Query];
  N02[label=2.0];
  N03[label=Query];
  N04[label=2.0];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=ToPosReal];
  N08[label=5.0];
  N09[label="+"];
  N10[label=Query];
  N11[label=8.0];
  N12[label=Query];
  N13[label=9.0];
  N14[label="+"];
  N15[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N07 -> N09;
  N07 -> N14;
  N08 -> N09;
  N09 -> N10;
  N11 -> N12;
  N13 -> N14;
  N14 -> N15;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_sub(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                sub_1(),
                sub_2(),
                sub_3(),
                sub_4(),
                sub_5(),
                sub_6(),
                sub_7(),
                sub_8(),
                sub_9(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=Query];
  N02[label=3.0];
  N03[label=Query];
  N04[label=4.0];
  N05[label=Query];
  N06[label=2.0];
  N07[label=Beta];
  N08[label=Sample];
  N09[label=ToReal];
  N10[label=-5.0];
  N11[label="+"];
  N12[label=Query];
  N13[label=-6.0];
  N14[label="+"];
  N15[label=Query];
  N16[label=-7.0];
  N17[label="+"];
  N18[label=Query];
  N19[label=8.0];
  N20[label=Query];
  N21[label=-9.0];
  N22[label="+"];
  N23[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N11;
  N09 -> N14;
  N09 -> N17;
  N09 -> N22;
  N10 -> N11;
  N11 -> N12;
  N13 -> N14;
  N14 -> N15;
  N16 -> N17;
  N17 -> N18;
  N19 -> N20;
  N21 -> N22;
  N22 -> N23;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_xor(self) -> None:
        self.maxDiff = None

        # ^ operators are not yet properly supported by the compiler/BMG;
        # update this test when we get them working.

        queries = [
            xor_1(),
            xor_2(),
            xor_3(),
            xor_4(),
            xor_5(),
            xor_6(),
            xor_7(),
            xor_8(),
            xor_9(),
        ]
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer(queries, {}, 1)
        expected = """
The model uses a ^ operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a ^ operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a ^ operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
The model uses a ^ operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_exp(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot(
            [
                exp_1(),
                exp_2(),
                exp_3(),
                exp_4(),
                exp_5(),
                exp_6(),
                exp_7(),
            ],
            {},
        )
        expected = """
digraph "graph" {
  N00[label=2.7182817459106445];
  N01[label=Query];
  N02[label=7.389056205749512];
  N03[label=Query];
  N04[label=20.08553695678711];
  N05[label=Query];
  N06[label="[54.598148345947266,54.598148345947266]"];
  N07[label=Query];
  N08[label=2.0];
  N09[label=Beta];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=5.0];
  N13[label="+"];
  N14[label=Exp];
  N15[label=Query];
  N16[label=6.0];
  N17[label="+"];
  N18[label=Exp];
  N19[label=Query];
  N20[label=7.0];
  N21[label="+"];
  N22[label=Exp];
  N23[label=Query];
  N00 -> N01;
  N02 -> N03;
  N04 -> N05;
  N06 -> N07;
  N08 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N13;
  N11 -> N17;
  N11 -> N21;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_expm1(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([expm1_prob()], {})
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=ToPosReal];
  N4[label=ExpM1];
  N5[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([expm1_real()], {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=ExpM1];
  N5[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([expm1_negreal()], {})
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N3[label="-"];
  N4[label=ExpM1];
  N5[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_arithmetic_logistic(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([logistic_prob()], {})
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=ToReal];
  N4[label=Logistic];
  N5[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([logistic_real()], {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Logistic];
  N5[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = BMGInference().to_dot([logistic_negreal()], {})
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N3[label="-"];
  N4[label=ToReal];
  N5[label=Logistic];
  N6[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_misc_arithmetic(self) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot([stochastic_arithmetic()], {})
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.6000000238418579];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=-0.010050326585769653];
  N07[label=-4.605170249938965];
  N08[label=0.0];
  N09[label=if];
  N10[label=if];
  N11[label="+"];
  N12[label=Exp];
  N13[label=complement];
  N14[label=Bernoulli];
  N15[label=Sample];
  N16[label=Query];
  N00 -> N01;
  N01 -> N02;
  N02 -> N09;
  N03 -> N04;
  N04 -> N05;
  N05 -> N10;
  N06 -> N11;
  N07 -> N09;
  N07 -> N10;
  N08 -> N09;
  N08 -> N10;
  N09 -> N11;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_bmg_neg_of_neg(self) -> None:
        # This test shows that we treat torch.neg the same as the unary negation
        # operator when generating a graph.
        #
        # TODO: This test also shows that we do NOT optimize away negative-of-negative
        # which we certainly could. Once we implement that optimization, come back
        # and fix up this test accordingly.

        self.maxDiff = None
        observed = BMGInference().to_dot([neg_of_neg()], {})
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="-"];
  N5[label="-"];
  N6[label=Normal];
  N7[label=Sample];
  N8[label=Query];
  N0 -> N2;
  N1 -> N2;
  N1 -> N6;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_bmg_subtractions(self) -> None:
        # TODO: Notice in this code generation we end up with
        # the path:
        #
        # Beta -> Sample -> ToPosReal -> Negate -> ToReal -> MultiAdd
        #
        # We could optimize this path to
        #
        # Beta -> Sample -> ToReal -> Negate -> MultiAdd

        self.maxDiff = None
        observed = BMGInference().to_dot([subtractions()], {})
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=2.0];
  N05[label=Beta];
  N06[label=Sample];
  N07[label=HalfCauchy];
  N08[label=Sample];
  N09[label=ToPosReal];
  N10[label="-"];
  N11[label=ToReal];
  N12[label=ToReal];
  N13[label="-"];
  N14[label=ToReal];
  N15[label="+"];
  N16[label="-"];
  N17[label="+"];
  N18[label=Query];
  N00 -> N02;
  N01 -> N02;
  N01 -> N07;
  N02 -> N03;
  N03 -> N17;
  N04 -> N05;
  N04 -> N05;
  N05 -> N06;
  N06 -> N09;
  N06 -> N12;
  N07 -> N08;
  N08 -> N13;
  N09 -> N10;
  N10 -> N11;
  N11 -> N17;
  N12 -> N15;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_unsupported_arithmetic(self) -> None:
        self.maxDiff = None

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_bitxor()], {}, 1)
        expected = """
The model uses a ^ operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
        """
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())

    def test_unsupported_operands(self) -> None:
        self.maxDiff = None
        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_add()], {}, 1)
        # TODO: This error message is terrible; fix it.
        expected = """
The model uses a foo operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
        """
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())

    def test_tensor_mutations_augmented_assignment(self) -> None:
        self.maxDiff = None

        # See notes in mutating_assignments() for details
        observed = BMGInference().to_dot([mutating_assignments()], {})
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=ToPosReal];
  N4[label=3.0];
  N5[label="*"];
  N6[label=7.0];
  N7[label="+"];
  N8[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N7;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
