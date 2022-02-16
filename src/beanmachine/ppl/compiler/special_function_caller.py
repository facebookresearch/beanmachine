# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import math
import operator
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import beanmachine.ppl.compiler.bmg_nodes as bn
import torch
import torch.distributions as dist
from beanmachine.ppl.compiler.beanstalk_common import allowed_functions
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import BMGNode
from beanmachine.ppl.compiler.hint import log1mexp, math_log1mexp


def _ordinary_arg_or_const(arg: Any) -> bool:
    return isinstance(arg, bn.ConstantNode) or not isinstance(arg, BMGNode)


def only_ordinary_arguments(args, kwargs) -> bool:
    if any(isinstance(arg, BMGNode) for arg in args):
        return False
    if any(isinstance(arg, BMGNode) for arg in kwargs.values()):
        return False
    return True


def _only_ordinary_arguments_or_constants(
    args: List[Any], kwargs: Dict[str, Any]
) -> bool:
    return all(_ordinary_arg_or_const(arg) for arg in args) and all(
        _ordinary_arg_or_const(arg) for arg in kwargs.values()
    )


def _get_ordinary_value(x: Any) -> Any:
    return x.value if isinstance(x, bn.ConstantNode) else x


def _is_standard_normal(x: Any) -> bool:
    return isinstance(x, dist.Normal) and x.mean == 0.0 and x.stddev == 1.0


def _is_phi_bound(f: Any, arguments: List[Any], kwargs: Dict[str, Any]) -> bool:
    # Is this Normal(0.0, 1.0).cdf(x) ?
    # TODO: Support kwargs
    return (
        isinstance(f, MethodType)
        and f.__func__ is dist.Normal.cdf
        and len(arguments) == 1
        and _is_standard_normal(f.__self__)
    )


def _is_phi_unbound(f: Any, arguments: List[Any], kwargs: Dict[str, Any]) -> bool:
    # Is this Normal.cdf(Normal(0.0, 1.0), x)?
    # TODO: Support kwargs
    return (
        f is dist.Normal.cdf
        and len(arguments) == 2
        and _is_standard_normal(arguments[0])
    )


def _is_phi(f: Any, arguments: List[Any], kwargs: Dict[str, Any]) -> bool:
    return _is_phi_unbound(f, arguments, kwargs) or _is_phi_bound(f, arguments, kwargs)


def _flatten_all_lists(xs):
    """Takes a list-of-lists, with arbitrary nesting level;
    returns an iteration of all elements."""
    if isinstance(xs, list):
        for x in xs:
            yield from _flatten_all_lists(x)
    else:
        yield xs


def _list_to_zeros(xs):
    """Takes a list-of-lists, with arbitrary nesting level;
    returns a list-of-lists of the same shape but with every non-list
    element replaced with zero."""
    if isinstance(xs, list):
        return [_list_to_zeros(x) for x in xs]
    return 0


def _hashable(x: Any) -> bool:
    # Oddly enough, Python does not allow you to test for set inclusion
    # if the object is not hashable. Since it is impossible for an unhashable
    # object to be in a set, Python could simply say no when asked if a set
    # contains any unhashable object. It does not, so we are forced to do so.

    # All hashable objects have a callable __hash__ attribute.
    if not hasattr(x, "__hash__"):
        return False
    if not isinstance(x.__hash__, Callable):
        return False

    # It is possible that callable __hash__ exists but throws, which makes it
    # unhashable. Eliminate that possibility as well.
    try:
        hash(x)
    except Exception:
        return False

    return True


_empty_args = []
_empty_kwargs = {}

# Oddly enough there does not appear to be an easy way to obtain the type
# of builtin methods.
_builtin_function_or_method = type(abs)


def _is_tensor_unbound_instance_method(f: Callable) -> bool:
    # This identifies if a function object is a method *descriptor*
    # such as torch.Tensor.add; that is, the method before it is bound
    # to a particular self. This function does NOT identify if a function
    # is a bound instance method, such as torch.tensor(1.0).add.  See below.
    if not inspect.ismethoddescriptor(f):
        return False
    objc = getattr(f, "__objclass__", None)
    return objc is torch.Tensor or objc in torch.Tensor.__bases__


def _is_tensor_bound_instance_method(f: Callable) -> bool:
    # This identifies if a function object is an instance method of
    # a tensor already bound to a particular self.  All such functions
    # in torch are marked as builtin.
    return isinstance(f, _builtin_function_or_method) and isinstance(
        getattr(f, "__self__", None), torch.Tensor
    )


def _get_unbound_tensor_method(f: Callable) -> Callable:
    # Given a bound-to-self tensor instance method, obtain its corresponding
    # unbound descriptor. In normal Python, the protocol is that the bound
    # method has attribute __func__ pointing back to the descriptor but
    # torch does not follow this protocol. Rather, we'll look it up by name.
    assert _is_tensor_bound_instance_method(f)
    unbound = getattr(torch.Tensor, f.__name__, None)
    assert _is_tensor_unbound_instance_method(unbound)
    return unbound


def canonicalize_function(
    function: Any, arguments: List[Any]
) -> Tuple[Callable, List[Any]]:
    # In Python a function that is a member of a class can be in either a "bound"
    # or "unbound" form. Suppose c is of type C and we are calling foo with argument
    # x. We could have:
    #
    # bound:   c.foo(x)
    # unbound: C.foo(c, x)
    #
    # The bound version calls the unbound version. How? In the bound case the fetch
    # of c.foo returns a method object with attribute __self__ set to c and attribute
    # __func__ set to C.foo.  The call on the method object then invokes
    # __func__(__self__, x).
    #
    # Unfortunately, calls to torch tensor methods do not follow this convention;
    # instead of returning a method object with __func__ and __self__, it returns
    # a builtin method object with __self__ but no __func__, so we call special helpers
    # for those.
    #
    # It is useful when analyzing calls to have them in a consistent form. This function
    # turns bound function calls into the equivalent unbound function call.

    if isinstance(function, MethodType):
        f = function.__func__
        args = [function.__self__] + arguments
        assert isinstance(f, Callable)
    elif _is_tensor_bound_instance_method(function):
        f = _get_unbound_tensor_method(function)
        args = [function.__self__] + arguments
    elif isinstance(function, Callable):
        f = function
        args = arguments
    else:
        raise ValueError(f"Function {function} is not supported by Bean Machine Graph.")
    assert isinstance(f, Callable), (
        "_canonicalize_function should return callable " + f"but got {type(f)} {str(f)}"
    )
    return (f, args)


# This helper class is to solve a problem in the simulated
# execution of the model during graph accumulation. Consider
# a model fragment like:
#
#   n = normal()
#   y = n.exp()
#
# During graph construction, n will be a SampleNode whose
# operand is a NormalNode, but SampleNode does not have a
# method "exp".
#
# The lifted program we execute will be something like:
#
#   n = bmg.handle_function(normal, [])
#   func = bmg.handle_dot(n, "exp")
#   y = bmg.handle_function(func, [])
#
# The "func" that is returned is one of these KnownFunction
# objects, which captures the notion "I am an invocation
# of known function Tensor.exp on a receiver that is a BMG
# node".  We then turn that into a exp node in handle_function.


class KnownFunction:
    receiver: BMGNode
    function: Callable

    def __init__(self, receiver: BMGNode, function: Callable) -> None:
        if not isinstance(receiver, BMGNode):
            raise TypeError(
                f"KnownFunction receiver must be BMGNode but is {type(receiver)}"
            )
        if not isinstance(function, Callable):
            raise TypeError(
                f"KnownFunction function must be Callable but is {type(function)}"
            )

        self.receiver = receiver
        self.function = function


class SpecialFunctionCaller:
    # As we execute the lifted program, we accumulate graph nodes in the
    # graph builder,and the program passes around graph nodes instead of
    # regular values. What happens when a graph node is passed to a
    # function, or used as the receiver of a function? That function will be
    # expecting a regular value as its argument or receiver.
    #
    # Certain function calls are special because they call graph nodes to
    # be created; we have a dictionary here that maps Python function objects
    # to the graph builder method that knows how to create the appropriate
    # node type.
    #
    # There are also some functions which we know can be passed a graph node
    # and will treat it correctly even though it is a graph node and not
    # a value. For example, the function which constructs a dictionary
    # or the function which constructs a list. When we encounter one of
    # these functions in the lifted program, we do not create a graph node
    # or call a special helper function; we simply allow it to be called normally.

    _bmg: BMGraphBuilder
    _function_map: Dict[Callable, Callable]
    _special_tensor_instance_function_names: Set[str]

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self._bmg = bmg
        self._function_map = {
            #
            # Built-in functions
            #
            float: self._builtin_float,
            #
            # Math functions
            #
            math.exp: self._math_exp,
            math.log: self._math_log,
            #
            # Hints
            #
            log1mexp: self._hint_log1mexp,
            math_log1mexp: self._hint_log1mexp,
            #
            # Operators as functions
            #
            operator.add: self._operator_add,
            operator.and_: self._operator_and,
            # TODO: operator.contains
            # TODO: operator.eq
            operator.floordiv: self._operator_floordiv,
            # TODO: operator.ge
            # TODO: operator.gt
            operator.inv: self._operator_inv,
            # TODO: operator.is_
            # TODO: operator.is_not
            # TODO: operator.le
            operator.lshift: self._operator_lshift,
            # TODO: operator.lt
            operator.matmul: self._operator_matmul,
            operator.mod: self._operator_mod,
            operator.mul: self._operator_mul,
            # TODO: operator.ne
            operator.neg: self._operator_neg,
            operator.not_: self._operator_not,
            operator.or_: self._operator_or,
            operator.pos: self._operator_pos,
            operator.pow: self._operator_pow,
            operator.rshift: self._operator_rshift,
            operator.sub: self._operator_sub,
            operator.truediv: self._operator_truediv,
            operator.xor: self._operator_xor,
            #
            #
            # Torch distributions
            #
            # (Remember to add a case to distribution_to_node.)
            #
            dist.Bernoulli: self._dist_bernoulli,
            dist.Beta: self._dist_beta,
            dist.Binomial: self._dist_binomial,
            dist.Categorical: self._dist_categorical,
            # TODO: Cauchy
            dist.Chi2: self._dist_chi2,
            # TODO: ContinuousBernoulli
            dist.Dirichlet: self._dist_dirichlet,
            # TODO: Exponential
            # TODO: FisherSnedecor
            dist.Gamma: self._dist_gamma,
            # TODO: Geometric
            # TODO: Gumbel
            dist.HalfCauchy: self._dist_halfcauchy,
            dist.HalfNormal: self._dist_halfnormal,
            # TODO: Independent
            # TODO: Kumaraswamy
            # TODO: LKJCholesky
            # TODO: Laplace
            # TODO: LogNormal
            # TODO: LowRankMultivariateNormal
            # TODO: MixtureSameFamily
            # TODO: Multinomial
            # TODO: MultivariateNormal
            # TODO: NegativeBinomial
            dist.Normal: self._dist_normal,
            # TODO: OneHotCategorical
            # TODO: Pareto
            # TODO: Poisson
            dist.Poisson: self._dist_poisson,
            # TODO: RelaxedBernoulli
            # TODO: LogitRelaxedBernoulli
            # TODO: RelaxedOneHotCategorical
            dist.StudentT: self._dist_studentt,
            # TODO: TransformedDistribution
            dist.Uniform: self._dist_uniform,
            # TODO: VonMises
            # TODO: Weibull
            #
            # Torch functions
            #
            torch.Tensor.add: self._torch_add,
            torch.add: self._torch_add,
            torch.Tensor.bitwise_and: self._torch_bitwise_and,  # pyre-ignore
            torch.bitwise_and: self._torch_bitwise_and,
            torch.Tensor.bitwise_not: self._torch_bitwise_not,  # pyre-ignore
            torch.bitwise_not: self._torch_bitwise_not,
            torch.Tensor.bitwise_or: self._torch_bitwise_or,  # pyre-ignore
            torch.bitwise_or: self._torch_bitwise_or,
            torch.Tensor.bitwise_xor: self._torch_bitwise_xor,  # pyre-ignore
            torch.bitwise_xor: self._torch_bitwise_xor,
            torch.Tensor.bitwise_left_shift: self._torch_bitwise_left_shift,  # pyre-ignore
            torch.bitwise_left_shift: self._torch_bitwise_left_shift,
            torch.Tensor.bitwise_right_shift: self._torch_bitwise_right_shift,  # pyre-ignore
            torch.bitwise_right_shift: self._torch_bitwise_right_shift,
            torch.Tensor.div: self._torch_div,
            torch.div: self._torch_div,
            torch.Tensor.divide: self._torch_div,  # pyre-ignore
            torch.divide: self._torch_div,
            # TODO: eq
            # TODO: equal
            torch.Tensor.exp: self._torch_exp,  # pyre-ignore
            torch.exp: self._torch_exp,
            # TODO: exp2
            # TODO: special.exp2
            torch.Tensor.expm1: self._torch_expm1,  # pyre-ignore
            torch.expm1: self._torch_expm1,
            torch.special.expm1: self._torch_expm1,
            torch.Tensor.float: self._torch_float,
            # TODO: float_power
            torch.Tensor.floor_divide: self._torch_floor_divide,  # pyre-ignore
            torch.floor_divide: self._torch_floor_divide,
            torch.Tensor.fmod: self._torch_fmod,  # pyre-ignore
            torch.fmod: self._torch_fmod,
            # TODO: ge
            # TODO: greater_equal
            # TODO: gt
            # TODO: greater
            torch.Tensor.int: self._torch_int,
            torch.Tensor.item: self._torch_item,
            # TODO: le
            # TODO: less_equal
            torch.Tensor.log: self._torch_log,
            torch.log: self._torch_log,
            # TODO: log10
            # TODO: log2
            # TODO: logical_and
            torch.Tensor.logical_not: self._torch_logical_not,  # pyre-ignore
            torch.logical_not: self._torch_logical_not,
            # TODO: logical_or
            # TODO: logical_xor
            torch.Tensor.logsumexp: self._torch_logsumexp,
            torch.logsumexp: self._torch_logsumexp,
            # TODO: lt
            # TODO: less
            torch.Tensor.matmul: self._torch_matmul,
            torch.matmul: self._torch_matmul,
            torch.Tensor.mm: self._torch_mm,  # pyre-ignore
            torch.mm: self._torch_mm,
            torch.Tensor.mul: self._torch_mul,
            torch.mul: self._torch_mul,
            torch.Tensor.multiply: self._torch_mul,  # pyre-ignore
            torch.multiply: self._torch_mul,
            # TODO: ne
            # TODO: not_equal
            torch.Tensor.neg: self._torch_neg,
            torch.neg: self._torch_neg,
            torch.Tensor.negative: self._torch_neg,  # pyre-ignore
            torch.negative: self._torch_neg,
            torch.Tensor.pow: self._torch_pow,
            torch.pow: self._torch_pow,
            torch.Tensor.remainder: self._torch_fmod,  # pyre-ignore
            torch.remainder: self._torch_fmod,
            torch.sigmoid: self._torch_sigmoid,
            torch.Tensor.sigmoid: self._torch_sigmoid,
            torch.special.expit: self._torch_sigmoid,
            # TODO: sqrt
            torch.Tensor.sub: self._torch_sub,
            torch.sub: self._torch_sub,
            torch.Tensor.subtract: self._torch_sub,  # pyre-ignore
            torch.subtract: self._torch_sub,
            torch.Tensor.true_divide: self._torch_div,  # pyre-ignore
            torch.true_divide: self._torch_div,
        }
        self._special_tensor_instance_function_names = {
            f.__name__
            for f in self._function_map
            if _is_tensor_unbound_instance_method(f)
        }

    def _is_special_tensor_bound_instance_method_name(self, name: str) -> bool:
        return name in self._special_tensor_instance_function_names

    def bind_tensor_instance_function(
        self, receiver: BMGNode, name: str
    ) -> KnownFunction:
        return KnownFunction(receiver, getattr(torch.Tensor, name))

    def is_special_tensor_bound_instance_method(self, f: Callable) -> bool:
        return self._is_special_tensor_bound_instance_method_name(
            f.__name__
        ) and _is_tensor_bound_instance_method(f)

    def get_special_tensor_unbound_instance_method(self, f: Callable) -> Callable:
        assert self.is_special_tensor_bound_instance_method(f)
        return _get_unbound_tensor_method(f)

    def _make_constant(self, arg: Any) -> BMGNode:
        return arg if isinstance(arg, BMGNode) else self._bmg.add_constant(arg)

    def is_special_function(
        self,
        func: Callable,
        args: List[Any] = _empty_args,
        kwargs: Dict[str, Any] = _empty_kwargs,
    ) -> bool:
        if isinstance(func, KnownFunction):
            return True
        if _is_tensor_unbound_instance_method(func):
            return True
        if _is_tensor_bound_instance_method(func):
            return True
        if _is_phi(func, args, kwargs):
            return True
        if func is torch.tensor and len(args) == 1:
            return True
        if not _hashable(func):
            return False
        if func in allowed_functions:
            return True
        if func in self._function_map:
            return True

        return False

    def _canonicalize_function(
        self, func: Callable, args: List[Any]
    ) -> Tuple[Callable, List[Any]]:
        if isinstance(func, KnownFunction):
            args = [func.receiver] + args
            func = func.function
        else:
            func, args = canonicalize_function(func, args)
        return func, args

    def do_special_call_maybe_stochastic(
        self,
        func: Any,
        args: List[Any],
        kwargs: Dict[str, Any] = _empty_kwargs,
    ) -> Any:
        # If we possibly can, just call the original function with ordinary arguments.
        # Otherwise, convert everything to a graph node and call our helper which
        # does node construction.

        assert self.is_special_function(func, args, kwargs)
        func, args = self._canonicalize_function(func, args)
        if func is torch.tensor:
            return self._tensor_constructor(*args, **kwargs)
        if (
            _only_ordinary_arguments_or_constants(args, kwargs)
            or func in allowed_functions
        ):
            new_args = (_get_ordinary_value(arg) for arg in args)
            new_kwargs = {key: _get_ordinary_value(arg) for key, arg in kwargs.items()}
            return func(*new_args, **new_kwargs)
        return self.do_special_call_always_stochastic(func, args, kwargs)

    def do_special_call_always_stochastic(
        self,
        func: Callable,
        args: List[Any],
        kwargs: Dict[str, Any] = _empty_kwargs,
    ) -> BMGNode:
        # Never call the original function with ordinary arguments. Convert everything
        # to a graph node and call our helper which does node construction.
        assert self.is_special_function(func, args, kwargs)
        # We should never call do_special_call_always_stochastic on (1) a tensor
        # constructor, or (2) a function known to be allowed to take any values.
        assert func not in allowed_functions
        assert func is not torch.tensor
        func, args = self._canonicalize_function(func, args)

        if _is_phi_unbound(func, args, kwargs):
            args = args[1:]
            node_constructor = self._phi
        elif _hashable(func) and func in self._function_map:
            node_constructor = self._function_map[func]
        else:
            # We are trying to do an always-stochastic call on a function that
            # we do not yet know how to handle.
            raise ValueError(f"Function {func} is not supported by Bean Machine Graph.")
        new_args = (self._make_constant(arg) for arg in args)
        new_kwargs = {key: self._make_constant(arg) for key, arg in kwargs.items()}
        return node_constructor(*new_args, **new_kwargs)

    #
    # Builtins; these must have the same signature as their corresponding
    # builtin functions.
    #

    def _builtin_float(self, input: BMGNode) -> BMGNode:
        # TODO: Do we want to do this at all? Why should float(t) insert a
        # TO_REAL node into the graph? We can simply insert TO_REAL where required
        # by the BMG type system.
        return self._bmg.add_to_real(input)

    #
    # Math functions
    #
    def _math_exp(self, input: BMGNode) -> BMGNode:
        # TODO: Right signature?
        return self._bmg.add_exp(input)

    def _math_log(self, input: BMGNode) -> BMGNode:
        return self._bmg.add_log(input)

    #
    # Hints
    # TODO: Eliminate this hack. Write a problem fixer which detects these
    # patterns and rewrites them into the more efficient operator.
    #

    def _hint_log1mexp(self, x: BMGNode) -> BMGNode:
        return self._bmg.add_log1mexp(x)

    #
    # Distributions; these must have the same signature as the corresponding
    # constructor.
    #
    def distribution_to_node(  # noqa
        self, distribution: dist.Distribution
    ) -> bn.DistributionNode:
        t = type(distribution)

        if not self.is_special_function(t):
            # TODO: Better error
            raise TypeError(
                f"Distribution '{t.__name__}' is not supported by Bean Machine Graph."
            )

        if isinstance(distribution, dist.Bernoulli):
            args = [distribution.probs]
        elif isinstance(distribution, dist.Beta):
            args = [distribution.concentration1, distribution.concentration0]
        elif isinstance(distribution, dist.Binomial):
            args = [distribution.total_count, distribution.probs]
        elif isinstance(distribution, dist.Categorical):
            args = [distribution.probs]
        elif isinstance(distribution, dist.Chi2):
            args = [distribution.df]
        elif isinstance(distribution, dist.Dirichlet):
            args = [distribution.concentration]
        elif isinstance(distribution, dist.Gamma):
            args = [distribution.concentration, distribution.rate]
        elif isinstance(distribution, dist.HalfCauchy):
            args = [distribution.scale]
        elif isinstance(distribution, dist.HalfNormal):
            args = [distribution.scale]
        elif isinstance(distribution, dist.Normal):
            args = [distribution.mean, distribution.stddev]
        elif isinstance(distribution, dist.Poisson):
            args = [distribution.rate]
        elif isinstance(distribution, dist.StudentT):
            args = [distribution.df, distribution.loc, distribution.scale]
        elif isinstance(distribution, dist.Uniform):
            args = [distribution.low, distribution.high]
        else:
            raise AssertionError(
                "Internal compiler error: missing case when converting distribution to node."
            )

        d = self.do_special_call_always_stochastic(t, args, {})
        assert isinstance(d, bn.DistributionNode)
        return d

    def _dist_bernoulli(
        self,
        probs: Optional[BMGNode] = None,
        logits: Optional[BMGNode] = None,
        validate_args: Any = None,
    ) -> BMGNode:
        if (probs is None and logits is None) or (
            probs is not None and logits is not None
        ):
            raise ValueError("Bernoulli requires exactly one of probs or logits")
        if logits is not None:
            return self._bmg.add_bernoulli_logit(logits)
        return self._bmg.add_bernoulli(probs)

    def _dist_beta(
        self,
        concentration1: BMGNode,
        concentration0: BMGNode,
        validate_args: Any = None,
    ) -> BMGNode:
        return self._bmg.add_beta(concentration1, concentration0)

    def _dist_binomial(
        self,
        total_count: Optional[BMGNode] = None,
        probs: Optional[BMGNode] = None,
        logits: Optional[BMGNode] = None,
        validate_args: Any = None,
    ) -> BMGNode:
        if (probs is None and logits is None) or (
            probs is not None and logits is not None
        ):
            raise ValueError("Binomial requires exactly one of probs or logits")

        # TODO: Create a test case for Binomial(probs=0.5) where total_count
        # is omitted.
        if total_count is None:
            total_count = self._make_constant(1)

        if logits is not None:
            return self._bmg.add_binomial_logit(total_count, logits)
        return self._bmg.add_binomial(total_count, probs)

    def _dist_categorical(
        self,
        probs: Optional[BMGNode] = None,
        logits: Optional[BMGNode] = None,
        validate_args: Any = None,
    ) -> BMGNode:
        if (probs is None and logits is None) or (
            probs is not None and logits is not None
        ):
            raise ValueError("Categorical requires exactly one of probs or logits")
        if logits is not None:
            return self._bmg.add_categorical_logit(logits)
        return self._bmg.add_categorical(probs)

    def _dist_chi2(self, df: BMGNode, validate_args: Any = None) -> BMGNode:
        return self._bmg.add_chi2(df)

    def _dist_dirichlet(self, concentration: BMGNode, validate_args=None) -> BMGNode:
        return self._bmg.add_dirichlet(concentration)

    def _dist_gamma(
        self, concentration: BMGNode, rate: BMGNode, validate_args=None
    ) -> BMGNode:
        return self._bmg.add_gamma(concentration, rate)

    def _dist_halfcauchy(self, scale: BMGNode, validate_args=None) -> BMGNode:
        return self._bmg.add_halfcauchy(scale)

    def _dist_halfnormal(self, scale: Any, validate_args=None) -> BMGNode:
        return self._bmg.add_halfnormal(scale)

    def _dist_normal(self, loc: BMGNode, scale: BMGNode, validate_args=None) -> BMGNode:
        return self._bmg.add_normal(loc, scale)

    def _dist_poisson(self, rate: BMGNode) -> BMGNode:
        return self._bmg.add_poisson(rate)

    def _dist_studentt(
        self,
        df: BMGNode,
        loc: Optional[BMGNode] = None,
        scale: Optional[BMGNode] = None,
        validate_args=None,
    ) -> BMGNode:
        if loc is None:
            loc = self._make_constant(0)
        if scale is None:
            scale = self._make_constant(1)
        return self._bmg.add_studentt(df, loc, scale)

    def _dist_uniform(self, low: BMGNode, high: BMGNode, validate_args=None) -> BMGNode:
        return self._bmg.add_uniform(low, high)

    #
    # Tensor constructor
    #

    def _tensor_constructor(self, data: Any) -> Any:

        # The tensor constructor is a bit tricky because it takes a single
        # argument that is either a value or a list of values.  We need:
        # (1) a flattened list of all the arguments, and
        # (2) the size of the original tensor.

        flattened_args = list(_flatten_all_lists(data))
        if not any(isinstance(arg, BMGNode) for arg in flattened_args):
            # None of the arguments are graph nodes. We can just
            # construct the tensor normally.
            return torch.tensor(data)
        # At least one of the arguments is a graph node.
        #
        # If we're constructing a singleton tensor and the single value
        # is a graph node, we can just keep it as that graph node.
        if len(flattened_args) == 1:
            return flattened_args[0]

        # We have two or more arguments and at least one is a graph node.
        # Convert them all to graph nodes.
        for index, arg in enumerate(flattened_args):
            if not isinstance(arg, BMGNode):
                flattened_args[index] = self._bmg.add_constant(arg)

        # What shape is this tensor? Rather than duplicating the logic in the
        # tensor class, let's just construct the same shape made of entirely
        # zeros and then ask what shape it is.
        size = torch.tensor(_list_to_zeros(data)).size()
        return self._bmg.add_tensor(size, *flattened_args)

    #
    # Tensor functions; these must have the same signature as the
    # corresponding torch function.
    #
    # TODO: We do not support mutation of stochastic tensors; we should produce an
    # error if there are any "out" values.

    def _phi(self, value: BMGNode) -> BMGNode:
        return self._bmg.add_phi(value)

    def _torch_add(
        self,
        input: BMGNode,
        other: BMGNode,
        alpha: Optional[BMGNode] = None,
        out: Any = None,
    ) -> BMGNode:
        # TODO: tensor add has the semantics input + alpha * other; if alpha is present
        # then we need to generate a multiply and an addition.
        return self._bmg.add_addition(input, other)

    def _torch_bitwise_and(
        self, input: BMGNode, other: BMGNode, out: Any = None
    ) -> BMGNode:
        return self._bmg.add_bitand(input, other)

    def _torch_bitwise_left_shift(
        self, input: BMGNode, other: BMGNode, out: Any = None
    ) -> BMGNode:
        # TODO: In torch, a << b is not bitwise at all. Rather it is simply an
        # an alias for a * (2 ** b). Make a rewriter that turns shifts into
        # this operation.
        return self._bmg.add_lshift(input, other)

    def _torch_bitwise_not(self, input: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_invert(input)

    def _torch_bitwise_or(
        self, input: BMGNode, other: BMGNode, out: Any = None
    ) -> BMGNode:
        return self._bmg.add_bitor(input, other)

    def _torch_bitwise_right_shift(
        self, input: BMGNode, other: BMGNode, out: Any = None
    ) -> BMGNode:
        # TODO: In torch, a >> b is not bitwise at all. Rather it is simply an
        # an alias for a * (2 ** -b). Make a rewriter that turns shifts into
        # this operation.
        return self._bmg.add_rshift(input, other)

    def _torch_bitwise_xor(
        self, input: BMGNode, other: BMGNode, out: Any = None
    ) -> BMGNode:
        return self._bmg.add_bitxor(input, other)

    def _torch_div(
        self,
        input: BMGNode,
        other: BMGNode,
        rounding_mode: Optional[BMGNode] = None,
        out: Any = None,
    ) -> BMGNode:
        # TODO: Should we give an error if there is a rounding mode?
        return self._bmg.add_division(input, other)

    def _torch_exp(self, input: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_exp(input)

    def _torch_expm1(self, input: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_expm1(input)

    def _torch_float(
        self, input: BMGNode, memory_format: Optional[BMGNode] = None
    ) -> BMGNode:
        # TODO: Do we want to do this at all? Why should t.float() insert a
        # TO_REAL node into the graph? We can simply insert TO_REAL where required
        # by the BMG type system.
        # TODO: If we do keep this, what should we do with memory_format?
        return self._bmg.add_to_real(input)

    def _torch_floor_divide(
        self,
        input: BMGNode,
        other: BMGNode,
        out: Any = None,
    ) -> BMGNode:
        return self._bmg.add_floordiv(input, other)

    def _torch_fmod(
        self,
        input: BMGNode,
        other: BMGNode,
        out: Any = None,
    ) -> BMGNode:
        return self._bmg.add_mod(input, other)

    def _torch_int(
        self, input: BMGNode, memory_format: Optional[BMGNode] = None
    ) -> BMGNode:
        # TODO: What should we do with memory_format?
        return self._bmg.add_to_int(input)

    def _torch_item(self, input: BMGNode) -> Any:
        return self._bmg.add_item(input)

    def _torch_log(self, input: BMGNode, out: Any = None) -> Any:
        return self._bmg.add_log(input)

    def _torch_logical_not(self, input: BMGNode, out: Any = None) -> Any:
        return self._bmg.add_not(input)

    def _torch_logsumexp(
        self,
        input: BMGNode,
        dim: BMGNode,
        keepdim: Optional[BMGNode] = None,
        out: Any = None,
    ) -> Any:
        if keepdim is None:
            keepdim = self._make_constant(False)
        return self._bmg.add_logsumexp_torch(input, dim, keepdim)

    def _torch_matmul(self, input: BMGNode, other: BMGNode, out: Any = None) -> BMGNode:
        # TODO: mm and matmul have different behavior; we probably need to make
        # a distinction here.
        return self._bmg.add_matrix_multiplication(input, other)

    def _torch_mm(self, input: BMGNode, mat2: BMGNode, out: Any = None) -> BMGNode:
        # TODO: mm and matmul have different behavior; we probably need to make
        # a distinction here.
        return self._bmg.add_matrix_multiplication(input, mat2)

    def _torch_mul(self, input: BMGNode, other: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_multiplication(input, other)

    def _torch_neg(self, input: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_negate(input)

    def _torch_pow(self, input: BMGNode, exponent: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_power(input, exponent)

    def _torch_sigmoid(self, input: BMGNode, out: Any = None) -> BMGNode:
        return self._bmg.add_logistic(input)

    def _torch_sub(
        self,
        input: BMGNode,
        other: BMGNode,
        alpha: Optional[BMGNode] = None,
        out: Any = None,
    ) -> BMGNode:
        # TODO: tensor sub has the semantics input - alpha * other; if alpha is present
        # then we need to generate a multiply and an subtraction
        return self._bmg.add_subtraction(input, other)

    #
    # Operators as functions
    #

    def _operator_add(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_addition(a, b)

    def _operator_and(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_bitand(a, b)

    def _operator_floordiv(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_floordiv(a, b)

    def _operator_inv(self, obj: BMGNode) -> BMGNode:
        return self._bmg.add_invert(obj)

    def _operator_lshift(self, a: BMGNode, b: BMGNode) -> BMGNode:
        # TODO: In torch, a << b is not bitwise at all. Rather it is simply an
        # an alias for a * (2 ** b). Make a rewriter that turns shifts into
        # this operation.
        return self._bmg.add_lshift(a, b)

    def _operator_matmul(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_matrix_multiplication(a, b)

    def _operator_mod(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_mod(a, b)

    def _operator_mul(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_multiplication(a, b)

    def _operator_neg(self, obj: BMGNode) -> BMGNode:
        return self._bmg.add_negate(obj)

    def _operator_not(self, obj: BMGNode) -> BMGNode:
        return self._bmg.add_not(obj)

    def _operator_or(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_bitor(a, b)

    def _operator_pos(self, obj: BMGNode) -> BMGNode:
        # unary + is an identity on graph nodes
        return obj

    def _operator_pow(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_power(a, b)

    def _operator_rshift(self, a: BMGNode, b: BMGNode) -> BMGNode:
        # TODO: In torch, a >> b is not bitwise at all. Rather it is simply an
        # an alias for a * (2 ** -b). Make a rewriter that turns shifts into
        # this operation.
        return self._bmg.add_rshift(a, b)

    def _operator_sub(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_subtraction(a, b)

    def _operator_truediv(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_division(a, b)

    def _operator_xor(self, a: BMGNode, b: BMGNode) -> BMGNode:
        return self._bmg.add_bitxor(a, b)
