# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Update this comment

"""A builder for the BeanMachine Graph language

The Beanstalk compiler has, at a high level, five phases.

* First, it transforms a Python model into a semantically
  equivalent program "single assignment" (SA) form that uses
  only a small subset of Python features.

* Second, it transforms that program into a "lifted" form.
  Portions of the program which do not involve samples are
  executed normally, but any computation that involves a
  stochastic node in any way is instead turned into a graph node.

  Jargon note:

  Every graph of a model will have some nodes that represent
  random samples and some which do not. For instance,
  we might have a simple coin flip model with three
  nodes: a sample, a distribution, and a constant probability:

  def flip():
    return Bernoulli(0.5)

  sample --> Bernoulli --> 0.5

  We'll refer to the nodes which somehow involve a sample,
  either directly or indirectly, as "stochastic" nodes.

* Third, we actually execute the lifted program and
  accumulate the graph.

* Fourth, the accumulated graph tracks the type information
  of the original Python program. We mutate the accumulated
  graph into a form where it obeys the rules of the BMG
  type system.

* Fifth, we either create actual BMG nodes in memory via
  native code interop, or we emit a program in Python or
  C++ which does so.

This module implements the graph builder that is called during
execution of the lifted program; it implements phases
three, four and five.
"""

import inspect
import operator
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.profiler as prof
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import BMGNode
from beanmachine.ppl.compiler.special_function_caller import (
    SpecialFunctionCaller,
    canonicalize_function,
)
from beanmachine.ppl.compiler.support import (
    ComputeSupport,
    TooBig,
    Infinite,
    Unknown,
    _limit as max_possibilities,
)
from beanmachine.ppl.inference.utils import _verify_queries_and_observations
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils.memoize import MemoizationKey


def _has_ordinary_value(x: Any) -> bool:
    return not isinstance(x, bn.BMGNode) or isinstance(x, bn.ConstantNode)


def _get_ordinary_value(x: Any) -> Any:
    return x.value if isinstance(x, bn.ConstantNode) else x


builtin_function_or_method = type(abs)


def _is_random_variable_call(f) -> bool:
    return hasattr(f, "is_random_variable") and f.is_random_variable


def _is_functional_call(f) -> bool:
    return hasattr(f, "is_functional") and f.is_functional


def _has_source_code(function: Callable) -> bool:
    try:
        inspect.getsource(function)
    except Exception:
        return False
    return True


class BMGRuntime:

    _bmg: BMGraphBuilder

    # As we construct the graph we may encounter "random variable" values; these
    # refer to a function that we need to transform into the "lifted" form. This
    # map tracks those so that we do not repeat work. However, RVIDs contain a
    # tuple of arguments which might contain tensors, and tensors are hashed by
    # reference, not by value. We therefore construct a map of RVID-equivalents
    # which is hashable by the values of the arguments.

    rv_map: Dict[MemoizationKey, BMGNode]
    lifted_map: Dict[Callable, Callable]

    # The graph we accumulate must be acyclic. We assume that an RVID-returning
    # function is pure, so if at any time such a function calls itself, either
    # it is impure or it is in an infinite recursion; either way, we will not
    # be able to construct a correct graph. When we are calling the lifted
    # form of a functional or random_variable method we track the RVID that
    # was used to trigger the call; if we ever encounter a call with the same
    # RVID while the lifted execution is "in flight", we throw an exception
    # and stop accumulating the graph.

    in_flight: Set[MemoizationKey]

    # We also need to keep track of which query node is associated
    # with each RVID queried by the user. Query nodes are deduplicated
    # so it is possible that two different RVIDs are associated with
    # the same query node.

    _rv_to_query: Dict[RVIdentifier, bn.Query]

    _pd: Optional[prof.ProfilerData]

    _special_function_caller: SpecialFunctionCaller

    def __init__(self) -> None:
        self._bmg = BMGraphBuilder()
        self._pd = None
        self.rv_map = {}
        self.lifted_map = {}
        self.in_flight = set()
        self._rv_to_query = {}
        self._special_function_caller = SpecialFunctionCaller(self._bmg)

    def _begin(self, s: str) -> None:
        pd = self._pd
        if pd is not None:
            pd.begin(s)

    def _finish(self, s: str) -> None:
        pd = self._pd
        if pd is not None:
            pd.finish(s)

    #
    # Operators
    #

    def _possibly_stochastic_op(
        self, normal_op: Callable, stochastic_op: Callable, values: List[Any]
    ) -> Any:
        # We have a bunch of values that are being passed to a function.
        # If all the values are ordinary (non-stochastic) then just call
        # the normal function that takes ordinary values. Otherwise,
        # convert all the non-nodes to nodes and call the node constructor.

        # TODO: This logic is duplicated with SpecialFunctionCaller; move
        # the operator handling into there as well.

        if all(_has_ordinary_value(v) for v in values):
            return normal_op(*(_get_ordinary_value(v) for v in values))
        return stochastic_op(
            *(
                v if isinstance(v, bn.BMGNode) else self._bmg.add_constant(v)
                for v in values
            )
        )

    def handle_greater_than(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.gt, self._bmg.add_greater_than, [input, other]
        )

    def handle_greater_than_equal(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.ge, self._bmg.add_greater_than_equal, [input, other]
        )

    def handle_less_than(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.lt, self._bmg.add_less_than, [input, other]
        )

    def handle_less_than_equal(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.le, self._bmg.add_less_than_equal, [input, other]
        )

    def handle_equal(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.eq, self._bmg.add_equal, [input, other]
        )

    def handle_not_equal(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.ne, self._bmg.add_not_equal, [input, other]
        )

    def handle_is(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.is_, self._bmg.add_is, [input, other]
        )

    def handle_is_not(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.is_not, self._bmg.add_is_not, [input, other]
        )

    def handle_in(self, input: Any, other: Any) -> Any:
        # Note that we cannot use operator.contains here because
        # "x in y" is the same as "contains(y, x)".
        return self._possibly_stochastic_op(
            lambda x, y: x in y, self._bmg.add_in, [input, other]
        )

    def handle_not_in(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            lambda x, y: x not in y, self._bmg.add_not_in, [input, other]
        )

    def handle_multiplication(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.mul, self._bmg.add_multiplication, [input, other]
        )

    def handle_matrix_multiplication(self, input: Any, mat2: Any) -> Any:
        # TODO: We need to make a distinction between torch.mm and
        # torch.matmul because they have different broadcasting behaviors.
        return self._possibly_stochastic_op(
            operator.matmul, self._bmg.add_matrix_multiplication, [input, mat2]
        )

    def handle_addition(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.add, self._bmg.add_addition, [input, other]
        )

    def handle_bitand(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.and_, self._bmg.add_bitand, [input, other]
        )

    def handle_bitor(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.or_, self._bmg.add_bitor, [input, other]
        )

    def handle_bitxor(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.xor, self._bmg.add_bitxor, [input, other]
        )

    def handle_subtraction(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.sub, self._bmg.add_subtraction, [input, other]
        )

    def handle_division(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.truediv, self._bmg.add_division, [input, other]
        )

    def handle_floordiv(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.floordiv, self._bmg.add_floordiv, [input, other]
        )

    def handle_lshift(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.lshift, self._bmg.add_lshift, [input, other]
        )

    def handle_mod(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.mod, self._bmg.add_mod, [input, other]
        )

    def handle_power(self, input: Any, exponent: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.pow, self._bmg.add_power, [input, exponent]
        )

    def handle_rshift(self, input: Any, other: Any) -> Any:
        return self._possibly_stochastic_op(
            operator.rshift, self._bmg.add_rshift, [input, other]
        )

    def _is_stochastic_tuple(self, t: Any):
        # A stochastic tuple is any tuple where any element is either a graph node
        # or a stochastic tuple.
        if not isinstance(t, tuple):
            return False
        for item in t:
            if isinstance(item, BMGNode):
                return True
            if self._is_stochastic_tuple(item):
                return True
        return False

    def _handle_tuple_index(self, left: Any, right: Tuple[Any]) -> Any:
        # We either have a tensor on the left and a stochastic tuple on the
        # right, or a graph node on the left and a tuple, stochastic or not,
        # on the right.  Either way, we decompose it into multiple index
        # operations.  The rules we're using are:
        #
        # * Indexing with an empty tuple is an identity
        # * Indexing with a single-element tuple just uses the element (see below!)
        # * If the tuple has multiple elements, break it up into a head element and
        #   a tail tuple.  Index with the head, and then index that with the tail.
        #
        # TODO: Unfortunately, the second rule does not match the actual behavior of
        # pytorch.  Suppose we have:
        #
        # t = tensor([[10, 20], [30, 40]])
        #
        # What is t[(1, 1)] ?
        #
        # By our proposed transformation this becomes t[1][(1,)] by the third rule, and then
        # t[1][1] by the second rule. This is correct, so what's the problem?  The problem is,
        # what is t[((1, 1),)]?
        #
        # By our second rule, t[((1, 1),)] becomes t[(1, 1)]; now we are in the
        # same case as before and end up with tensor(40). But that's not what torch
        # produces if you run this code! It produces tensor([[30, 40], [30, 40]]).
        #
        # We will come back to this point later and consider how to better represent
        # this kind of indexing operation in the graph; for now we'll just implement
        # the simplified approximation:

        # some_tensor[()] is an identity.
        if len(right) == 0:
            assert isinstance(left, BMGNode)
            return left

        # some_tensor[(x,)] is the same as some_tensor[x]
        if len(right) == 1:
            return self.handle_index(left, right[0])

        # some_tensor[(head, ...tail...)] is the same as some_tensor[head][...tail...]
        h = self.handle_index(left, right[0])
        return self.handle_index(h, right[1:])

    def handle_index(self, left: Any, right: Any) -> Any:
        if isinstance(left, BMGNode) and isinstance(right, tuple):
            return self._handle_tuple_index(left, right)
        if isinstance(left, torch.Tensor) and self._is_stochastic_tuple(right):
            return self._handle_tuple_index(left, right)
        # TODO: What if we have a non-tensor indexed with a stochastic value?
        # A list, for example?

        return self._possibly_stochastic_op(
            lambda x, y: x[y], self._bmg.add_index, [left, right]
        )

    def handle_slice(self, left: Any, lower: Any, upper: Any, step: Any) -> Any:
        if (
            isinstance(left, BMGNode)
            or isinstance(lower, BMGNode)
            or isinstance(upper, BMGNode)
            or isinstance(step, BMGNode)
        ):
            raise ValueError("Stochastic slices are not yet implemented.")
        return left[lower:upper:step]

    def handle_invert(self, input: Any) -> Any:
        return self._possibly_stochastic_op(operator.inv, self._bmg.add_invert, [input])

    def handle_negate(self, input: Any) -> Any:
        return self._possibly_stochastic_op(operator.neg, self._bmg.add_negate, [input])

    def handle_uadd(self, input: Any) -> Any:
        # Unary plus on a graph node is an identity.
        return self._possibly_stochastic_op(operator.pos, lambda x: x, [input])

    def handle_not(self, input: Any) -> Any:
        return self._possibly_stochastic_op(operator.not_, self._bmg.add_not, [input])

    #
    # Augmented assignment operators
    #

    def _handle_augmented_assignment(
        self,
        left: Any,
        right: Any,
        attr: str,  # "__iadd__", for example
        native: Callable,  # operator.iadd, for example
        handler: Callable,  # self.handle_addition, for example
    ) -> Any:
        # Handling augmented assignments (+=, -=, *=, and so on) has a lot of cases;
        # to cut down on code duplication we call this higher-level method. Throughout
        # the comments below we assume that we're handling a +=; the logic is the same
        # for all the operators.

        # TODO: We have a problem that we need to resolve regarding compilation of models
        # which have mutations of aliased tensors. Compare the action of these two similar:
        # models in the original Bean Machine implementation:
        #
        # @functional def foo():
        #   x = flip() # 0 or 1
        #   y = x      # y is an alias for x
        #   y += 1     # y is mutated in place and continues to alias x
        #   return x   # returns 1 or 2
        #
        # vs
        #
        # @functional def foo():
        #   x = flip() # 0 or 1
        #   y = x      # y is an alias for x
        #   y = y + 1  # y no longer aliases x; y is 1 or 2
        #   return x   # returns 0 or 1
        #
        # Suppose we are asked to compile the first model; how should we execute
        # the rewritten form of it so as to accumulate the correct graph? Unlike
        # tensors, graph nodes are not mutable!
        #
        # Here's what we're going to do for now:
        #
        # If neither operand is a graph node then do exactly what the model would
        # normally do:
        #
        if not isinstance(left, BMGNode) and not isinstance(right, BMGNode):
            return native(left, right)

        # At least one operand is a graph node. If we have tensor += graph_node
        # or graph_node += anything then optimistically assume that there
        # is NOT any alias of the mutated left side, and treat the += as though
        # it is a normal addition.
        #
        # TODO: Should we produce some sort of warning here telling the user that
        # the compiled model semantics might be different than the original model?
        # Or is that too noisy? There are going to be a lot of models with += where
        # one of the operands is an ordinary tensor and one is a graph node, but which
        # do not have any aliasing problem.

        if isinstance(left, torch.Tensor) or isinstance(left, BMGNode):
            return handler(left, right)

        # If we've made it here then we have x += graph_node, where x is not a
        # tensor. There are two possibilities: either x is some type which implements
        # mutating in-place +=, or it is not.  If it is, then just call the mutator
        # and hope for the best.
        #
        # TODO: This scenario is another opportunity for a warning or error, since
        # the model is probably not one that can be compiled if it is depending on
        # in-place mutation of an object which has a stochastic quantity added to it.

        assert isinstance(right, BMGNode)
        if hasattr(left, attr):
            # It is possible that the operator exists but either returns
            # NotImplemented or raises NotImplementedError. In either case,
            # assume that we can fall back to non-mutating addition.
            try:
                result = native(left, right)
                if result is not NotImplemented:
                    return result
            except NotImplementedError:
                pass

        # We have x += graph_node, and x is not mutating in place; just
        return handler(left, right)

    def handle_iadd(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__iadd__", operator.iadd, self.handle_addition
        )

    def handle_isub(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__isub__", operator.isub, self.handle_subtraction
        )

    def handle_imul(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__imul__", operator.imul, self.handle_multiplication
        )

    def handle_idiv(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__idiv__", operator.itruediv, self.handle_division
        )

    def handle_ifloordiv(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__ifloordiv__", operator.ifloordiv, self.handle_floordiv
        )

    def handle_imod(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__imod__", operator.imod, self.handle_mod
        )

    def handle_ipow(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__ipow__", operator.ipow, self.handle_power
        )

    def handle_imatmul(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left,
            right,
            "__imatmul__",
            operator.imatmul,
            self.handle_matrix_multiplication,
        )

    def handle_ilshift(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__ilshift__", operator.ilshift, self.handle_lshift
        )

    def handle_irshift(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__irshift__", operator.irshift, self.handle_rshift
        )

    def handle_iand(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__iand__", operator.iand, self.handle_bitand
        )

    def handle_ixor(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__ixor__", operator.ixor, self.handle_bitxor
        )

    def handle_ior(self, left: Any, right: Any) -> Any:
        return self._handle_augmented_assignment(
            left, right, "__ior__", operator.ior, self.handle_bitor
        )

    #
    # Control flow
    #

    def handle_for(self, iter: Any) -> None:
        if isinstance(iter, BMGNode):
            # TODO: Better error
            raise ValueError("Stochastic control flows are not yet implemented.")

    def handle_if(self, test: Any) -> None:
        if isinstance(test, BMGNode):
            # TODO: Better error
            raise ValueError("Stochastic control flows are not yet implemented.")

    #
    # Function calls
    #

    def _handle_random_variable_call_checked(
        self, function: Any, arguments: List[Any], cs: ComputeSupport
    ) -> BMGNode:
        assert isinstance(arguments, list)

        # Identify the index of the leftmost graph node argument:

        index = next(
            (i for i, arg in enumerate(arguments) if isinstance(arg, BMGNode)), -1
        )
        if index == -1:
            # There were no graph node arguments. Just make an ordinary
            # function call
            rv = function(*arguments)
            assert isinstance(rv, RVIdentifier)
            return self._rv_to_node(rv)

        # We have an RV call where one or more arguments are graph nodes;
        # each graph node has finite support and the estimate of the number
        # of combinations we have to try is small.

        # Replace the given argument with all possible values and recurse.
        #
        # TODO: Note that we only memoize calls to RVs when the arguments
        # contain no graph nodes. Is this acceptable? We could save some
        # work if we also memoized calls of the form "rv1(rv2())". Right now
        # we would recompute the support of rv2() on the second such call,
        # and only get the savings of skipping the method calls on each
        # individual call.  Do some performance testing.

        replaced_arg = arguments[index]
        switch_inputs = [replaced_arg]

        for new_arg in cs[replaced_arg]:
            key = self._bmg.add_constant(new_arg)
            new_arguments = list(arguments)
            new_arguments[index] = new_arg
            value = self._handle_random_variable_call_checked(
                function, new_arguments, cs
            )
            switch_inputs.append(key)
            switch_inputs.append(value)
        return self._bmg.add_switch(*switch_inputs)

    def _handle_random_variable_call(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> BMGNode:

        if len(kwargs) != 0:
            # TODO: Better error
            raise ValueError(
                "Random variable function calls must not have named arguments."
            )

        cs = ComputeSupport()

        # If we have one or more graph nodes as arguments to an RV function call
        # then we need to try every possible value for those arguments. We require
        # that there be a finite number of possibilities, and that the total number
        # of branches generated for this call is small. Check that *once* before
        # recursively processing the call one argument at a time.

        # First let's see if any are not yet implemented.
        for arg in arguments:
            if isinstance(arg, BMGNode) and cs[arg] is Unknown:
                # TODO: Better exception
                raise ValueError(
                    f"Stochastic control flow not implemented for {str(arg)}."
                )

        # Are any infinite?
        for arg in arguments:
            if isinstance(arg, BMGNode) and cs[arg] is Infinite:
                # TODO: Better exception
                raise ValueError("Stochastic control flow must have finite support.")

        # Are any finite but too large?
        for arg in arguments:
            if isinstance(arg, BMGNode) and cs[arg] is TooBig:
                # TODO: Better exception
                raise ValueError("Stochastic control flow is too complex.")

        # Every argument has known, finite, small support. How many combinations are there?
        # TODO: Note that this can be a considerable overestimate. For example, if we
        # have outer(inner(), inner(), inner()) and the support of inner has 100 elements,
        # then there are 100 possible code paths to trace through outer, but we assume there
        # are 1000000. Is there anything we can do about that?

        # TODO: Make max_possibilities a global tweakable setting of the accumulator.
        possibilities = 1
        for arg in arguments:
            if isinstance(arg, BMGNode):
                possibilities *= len(cs[arg])
                if possibilities > max_possibilities:
                    # TODO: Better exception
                    raise ValueError("Stochastic control flow is too complex.")

        return self._handle_random_variable_call_checked(function, arguments, cs)

    def _handle_functional_call(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> BMGNode:

        if len(kwargs) != 0:
            # TODO: Better error
            raise ValueError("Functional calls must not have named arguments.")

        # We have a call to a functional function. There are two
        # cases. Either we have only ordinary values for arguments, or
        # we have one or more graph nodes.  *Do we need to handle these
        # two cases differently?*
        #
        # If the arguments are just plain arguments then we can call the
        # function normally, obtain an RVID back, and then use our usual
        # mechanism for turning an RVID into a graph node.
        #
        # What if the arguments are graph nodes? We can just do the same!
        # The callee will immediately return an RVID capturing the values
        # of the graph nodes. We then check to see if this exact call
        # has happened already; if it has, then we use the cached graph
        # node from our RVID->node cache. If it has not, we call the lifted
        # version of the method with the graph node arguments taken from
        # the RVID, and add the resulting graph node to the cache.
        #
        # Since this is a functional, not a random_variable, there is no
        # stochastic control flow to handle; we just pass the graph nodes in
        # as values and let the lifted method handle them.
        #
        # We lose nothing by doing this and we gain memoization that allows
        # us to skip doing the call if we have done it before. That's a win.

        rv = function(*arguments)
        assert isinstance(rv, RVIdentifier)
        return self._rv_to_node(rv)

    def _handle_ordinary_call(
        self, function: Callable, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        if not isinstance(function, Callable):
            raise TypeError(
                f"_handle_ordinary_call requires Callable but got {type(function)}"
            )
        # We have an ordinary function call to a function that is not on
        # our list of special functions, and is not a functional, and
        # is not a random variable.  We still need to lift the function
        # even if its arguments are not graph nodes though! It might do
        # arithmetic on a random variable even though it is not a functional.
        # For example, we might have something like:
        #
        # @random_variable
        # def norm1():
        #   return Normal(0, 1)
        #
        # # not a functional
        # def add_one():        # We call this function with no arguments
        #   return norm1() + 1
        #
        # @random_variable
        # def norm2():
        #   return Normal(add_one(), 1)
        #
        # Ideally we would like add_one to be marked as functional, but
        # given that it is not, we need to detect the call to add_one()
        # as returning a graph node that represents the sum of a sample
        # and a constant.

        # It is not already compiled; if we have source code, compile it.
        #
        # NOTE: Suppose we have a call to a function which is nested inside
        # a function that has already been compiled. Illustrative example:
        #
        # @rv def norm():
        #   def my_sum(x, y):
        #     return x + y
        #   return Normal(my_sum(mean(), offset()), 1.0)
        #
        # When we compile norm() we will *also* compile my_sum. When we then
        # call my_sum, we do *not* want to compile it *again*.  It is already
        # in the form "bmg.add_addition(x, y)" and so on; we do not want to
        # compile that program.
        #
        # Fortunately we do not need to even check because the generated code
        # has no source code! The inspect module does not believe that the code
        # generated from the AST has any source code, so _has_source_code returns
        # false, and we call the compiled function exactly as we should.

        if _has_source_code(function):
            return self._function_to_bmg_function(function)(*arguments, **kwargs)
        # It is not compiled and we have no source code to compile.
        # Just call it and hope for the best.
        # TODO: Do we need to consider the scenario where we do not have
        # source code, we call a function, and it somehow returns an RVID?
        # We *could* convert that to a graph node.
        return function(*arguments, **kwargs)

    def handle_function(
        self,
        function: Any,
        arguments: List[Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:

        if kwargs is None:
            kwargs = {}

        # Some functions we already have special-purpose handlers for,
        # like calls to math.exp or tensor.log.

        if self._special_function_caller.is_special_function(
            function, arguments, kwargs
        ):
            return self._special_function_caller.do_special_call_maybe_stochastic(
                function, arguments, kwargs
            )

        f, args = canonicalize_function(function, arguments)

        if _is_random_variable_call(f):
            return self._handle_random_variable_call(f, args, kwargs)

        if _is_functional_call(f):
            return self._handle_functional_call(f, args, kwargs)

        return self._handle_ordinary_call(f, args, kwargs)

    def _function_to_bmg_function(self, function: Callable) -> Callable:
        from beanmachine.ppl.compiler.bm_to_bmg import _bm_function_to_bmg_function

        # This method presupposes that the function is in its "unbound" form.
        assert not isinstance(function, MethodType)
        if function not in self.lifted_map:
            self.lifted_map[function] = _bm_function_to_bmg_function(function, self)
        return self.lifted_map[function]

    def _rv_to_node(self, rv: RVIdentifier) -> BMGNode:
        key = MemoizationKey(rv.wrapper, rv.arguments)
        if key not in self.rv_map:
            if key in self.in_flight:
                # TODO: Better error message
                raise RecursionError()
            self.in_flight.add(key)
            try:
                # Under what circumstances does a random variable NOT have source code?
                # When it is nested inside another rv that has already been compiled!
                # See the note in _handle_ordinary_call for details.
                if _has_source_code(rv.function):
                    rewritten_function = self._function_to_bmg_function(rv.function)
                else:
                    rewritten_function = rv.function

                # Here we deal with an issue caused by how Python produces the source
                # code of a function.
                #
                # We started with a function that produced a random variable when
                # called, and then we made a transformation based on the *source code*
                # of that original function. The *source code* of that original function
                # might OR might not have been decorated with a random_variable or
                # functional decorator.  For example, if we have:
                #
                # @random_variable
                # def foo():
                #   return Normal(0., 1.)
                #
                # and we have a query on foo() then that is the exact code that
                # we rewrite, and therefore the rewritten function that comes back
                # is *also* run through the random_variable decorator. But if instead
                # we have
                #
                # def foo():
                #   return Normal(0., 1.)
                #
                # bar = random_variable(foo)
                #
                # and a query on bar(), then when we ask Python for the source code of
                # bar, it hands us back the *undecorated* source code for foo, and
                # therefore the rewriter produces an undecorated rewritten function.
                #
                # How can we tell which situation we're in?  Well, if we're in the first
                # situation then when we call the rewritten function, we'll get back a
                # RVID, and if we're in the second situation, we will not.

                value = rewritten_function(*rv.arguments)
                if isinstance(value, RVIdentifier):
                    # We have a rewritten function with a decorator already applied.
                    # Therefore the rewritten form of the *undecorated* function is
                    # stored in the rv.  Call *that* function with the given arguments.
                    value = value.function(*rv.arguments)

                # We now have the value returned by the undecorated random variable
                # regardless of whether the source code was decorated or not.

                # If we are calling a random_variable then we must have gotten
                # back a distribution. This is the first time we have called this
                # rv with these arguments -- because we had a cache miss -- and
                # therefore we should generate a new sample node.  If by contrast
                # we are calling a functional then we check below that we got
                # back either a graph node or a tensor that we can make into a constant.
                if rv.is_random_variable:
                    value = self._handle_sample(value)
            finally:
                self.in_flight.remove(key)
            if isinstance(value, torch.Tensor):
                value = self._bmg.add_constant_tensor(value)
            if not isinstance(value, BMGNode):
                # TODO: Improve error message
                raise TypeError("A functional must return a tensor.")
            self.rv_map[key] = value
            return value
        return self.rv_map[key]

    def _handle_sample(self, operand: Any) -> bn.SampleNode:  # noqa
        """As we execute the lifted program, this method is called every
        time a model function decorated with @bm.random_variable returns; we verify that the
        returned value is a distribution that we know how to accumulate into the
        graph, and add a sample node to the graph."""

        if isinstance(operand, bn.DistributionNode):
            return self._bmg.add_sample(operand)

        if not isinstance(operand, torch.distributions.Distribution):
            # TODO: Better error
            raise TypeError("A random_variable is required to return a distribution.")

        d = self._special_function_caller.distribution_to_node(operand)
        return self._bmg.add_sample(d)

    def handle_dot_get(self, operand: Any, name: str) -> Any:
        # If we have x = foo.bar, foo must not be a sample; we have no way of
        # representing the "get the value of an attribute" operation in BMG.
        # However, suppose foo is a distribution of tensors; we do wish to support
        # operations such as:
        # x = foo.exp
        # y = x()
        # and have y be a graph that applies an EXP node to the SAMPLE node for foo.
        # This will require some cooperation between handling dots and handling
        # functions.

        if isinstance(operand, BMGNode):
            # If we're invoking a function on a graph node during execution of
            # the lifted program, that graph node is almost certainly a tensor
            # in the original program; assume that it is, and see if this is
            # a function on a tensor that we know how to accumulate into the graph.
            return self._special_function_caller.bind_tensor_instance_function(
                operand, name
            )

        return getattr(operand, name)

    def handle_dot_set(self, operand: Any, name: str, value: Any) -> None:
        # If we have foo.bar = x, foo must not be a sample; we have no way of
        # representing the "set the value of an attribute" operation in BMG.
        if isinstance(operand, BMGNode):
            raise ValueError(
                f"Setting the value of attribute {name} is not "
                + "supported in Bean Machine Graph."
            )
        setattr(operand, name, value)

    def handle_subscript_assign(
        self, target: Any, index: Any, stop: Any, step: Any, value: Any
    ) -> None:
        # If we have "target[index:stop:step] = value" (any of index, stop or step
        # can be missing or None) then:
        # * Target must not be a graph node; there are no mutable graph nodes.
        # * Index, stop and step must not be a graph node; we do not have the ability
        #   to compile stochastic mutations of other tensors.
        # * If target is a tensor then value must not be a graph node. We cannot
        #   mutate an existing tensor with a stochastic value.

        if isinstance(target, BMGNode):
            # TODO: Better error
            raise ValueError(
                "Mutating a stochastic value is not supported in Bean Machine Graph."
            )
        if isinstance(index, BMGNode):
            # TODO: Better error
            raise ValueError(
                "Mutating a collection or tensor with a stochastic index is not "
                + "supported in Bean Machine Graph."
            )
        if isinstance(stop, BMGNode):
            # TODO: Better error
            raise ValueError(
                "Mutating a collection or tensor with a stochastic upper index is not "
                + "supported in Bean Machine Graph."
            )
        if isinstance(step, BMGNode):
            # TODO: Better error
            raise ValueError(
                "Mutating a collection or tensor with a stochastic step is not "
                + "supported in Bean Machine Graph."
            )
        if isinstance(value, BMGNode) and isinstance(target, torch.Tensor):
            raise ValueError(
                "Mutating a tensor with a stochastic value is not "
                + "supported in Bean Machine Graph."
            )
        target[index] = value

    def accumulate_graph(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Any],
    ) -> BMGraphBuilder:
        _verify_queries_and_observations(queries, observations, True)
        self._bmg._pd = self._pd
        self._begin(prof.accumulate)
        for rv, val in observations.items():
            node = self._rv_to_node(rv)
            assert isinstance(node, bn.SampleNode)
            self._bmg.add_observation(node, val)
        for qrv in queries:
            node = self._rv_to_node(qrv)
            q = self._bmg.add_query(node)
            self._rv_to_query[qrv] = q
        self._finish(prof.accumulate)
        return self._bmg
