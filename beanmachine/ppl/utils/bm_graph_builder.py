# Copyright (c) Facebook, Inc. and its affiliates.
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

import torch  # isort:skip  torch has to be imported before graph
import math
import sys
from typing import Any, Callable, Dict, List

# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this mModuleNotFoundError
# pyre-ignore-all-errors
from beanmachine.graph import Graph
from beanmachine.ppl.compiler.bmg_nodes import (
    AdditionNode,
    BernoulliNode,
    BetaNode,
    BinomialNode,
    BMGNode,
    BooleanNode,
    CategoricalNode,
    ConstantNode,
    DirichletNode,
    DistributionNode,
    DivisionNode,
    ExpNode,
    HalfCauchyNode,
    IfThenElseNode,
    IndexNode,
    LogNode,
    MapNode,
    MatrixMultiplicationNode,
    MultiplicationNode,
    NaturalNode,
    NegateNode,
    NormalNode,
    NotNode,
    Observation,
    OperatorNode,
    PositiveRealNode,
    PowerNode,
    ProbabilityNode,
    Query,
    RealNode,
    SampleNode,
    StudentTNode,
    TensorNode,
    ToRealNode,
    ToTensorNode,
    UniformNode,
)
from beanmachine.ppl.compiler.bmg_types import Natural, PositiveReal, Probability
from beanmachine.ppl.utils.dotbuilder import DotBuilder
from beanmachine.ppl.utils.memoize import memoize
from torch import Tensor
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    HalfCauchy,
    Normal,
    StudentT,
    Uniform,
)


builtin_function_or_method = type(abs)

#####
##### The following classes define the various graph node types.
#####

# TODO: This section is over two thousand lines of code, none of which
# makes use of the actual graph builder; the node types could be
# moved into a module of their own.


# If we encounter a function call on a stochastic node during
# graph accumulation we will attempt to build a node in the graph
# for that invocation. These are the instance methods on tensors
# that we recognize and can build a graph node for.  See
# KnownFunction below for more details.

known_tensor_instance_functions = [
    "add",
    "div",
    "exp",
    "float",
    "log",
    "logical_not",
    "mm",
    "mul",
    "neg",
    "pow",
]


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
    function: Any

    def __init__(self, receiver: BMGNode, function: Any) -> None:
        self.receiver = receiver
        self.function = function


#####
##### That's it for the graph nodes.
#####


def is_from_lifted_module(f) -> bool:
    return (
        hasattr(f, "__module__")
        and f.__module__ in sys.modules
        and hasattr(sys.modules[f.__module__], "_lifted_to_bmg")
    )


def is_ordinary_call(f, args, kwargs) -> bool:
    if is_from_lifted_module(f):
        return True
    if any(isinstance(arg, BMGNode) for arg in args):
        return False
    if any(isinstance(arg, BMGNode) for arg in kwargs.values()):
        return False
    return True


class BMGraphBuilder:

    # Note that Python 3.7 guarantees that dictionaries maintain insertion order.
    nodes: Dict[BMGNode, int]

    function_map: Dict[Callable, Callable]

    def __init__(self):
        self.nodes = {}
        self.function_map = {
            # Math functions
            math.exp: self.handle_exp,
            math.log: self.handle_log,
            # Tensor instance functions
            torch.Tensor.add: self.handle_addition,
            torch.Tensor.div: self.handle_division,
            torch.Tensor.exp: self.handle_exp,
            torch.Tensor.float: self.handle_to_real,
            torch.Tensor.logical_not: self.handle_not,
            torch.Tensor.log: self.handle_log,
            torch.Tensor.mm: self.handle_matrix_multiplication,
            torch.Tensor.mul: self.handle_multiplication,
            torch.Tensor.neg: self.handle_negate,
            torch.Tensor.pow: self.handle_power,
            # Tensor static functions
            torch.add: self.handle_addition,
            torch.div: self.handle_division,
            torch.exp: self.handle_exp,
            # Note that torch.float is not a function.
            torch.log: self.handle_log,
            torch.logical_not: self.handle_not,
            torch.mm: self.handle_matrix_multiplication,
            torch.mul: self.handle_multiplication,
            torch.neg: self.handle_negate,
            torch.pow: self.handle_power,
            # Distribution constructors
            Bernoulli: self.handle_bernoulli,
            Beta: self.handle_beta,
            Categorical: self.handle_categorical,
            Dirichlet: self.handle_dirichlet,
            HalfCauchy: self.handle_halfcauchy,
            Normal: self.handle_normal,
            StudentT: self.handle_studentt,
            Uniform: self.handle_uniform,
        }

    def add_node(self, node: BMGNode) -> None:
        if node not in self.nodes:
            for child in node.children:
                self.add_node(child)
            self.nodes[node] = len(self.nodes)

    def add_constant(self, value: Any) -> ConstantNode:
        if isinstance(value, bool):
            return self.add_boolean(value)
        if isinstance(value, int):
            return self.add_real(value)
        if isinstance(value, float):
            return self.add_real(value)
        if isinstance(value, Tensor):
            return self.add_tensor(value)
        raise TypeError("value must be a bool, real or tensor")

    @memoize
    def add_real(self, value: float) -> RealNode:
        node = RealNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_probability(self, value: float) -> ProbabilityNode:
        node = ProbabilityNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_pos_real(self, value: float) -> PositiveRealNode:
        node = PositiveRealNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_natural(self, value: int) -> NaturalNode:
        node = NaturalNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_boolean(self, value: bool) -> BooleanNode:
        node = BooleanNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_tensor(self, value: Tensor) -> TensorNode:
        node = TensorNode(value)
        self.add_node(node)
        return node

    @memoize
    def add_bernoulli(
        self, probability: BMGNode, is_logits: bool = False
    ) -> BernoulliNode:
        node = BernoulliNode(probability, is_logits)
        self.add_node(node)
        return node

    def handle_bernoulli(self, probs: Any = None, logits: Any = None) -> BernoulliNode:
        if (probs is None and logits is None) or (
            probs is not None and logits is not None
        ):
            raise ValueError("handle_bernoulli requires exactly one of probs or logits")
        probability = logits if probs is None else probs
        if not isinstance(probability, BMGNode):
            probability = self.add_constant(probability)
        return self.add_bernoulli(probability, logits is not None)

    @memoize
    def add_binomial(
        self, count: BMGNode, probability: BMGNode, is_logits: bool = False
    ) -> BinomialNode:
        node = BinomialNode(count, probability, is_logits)
        self.add_node(node)
        return node

    def handle_binomial(
        self, total_count: Any, probs: Any = None, logits: Any = None
    ) -> BinomialNode:
        if (probs is None and logits is None) or (
            probs is not None and logits is not None
        ):
            raise ValueError("handle_binomial requires exactly one of probs or logits")
        probability = logits if probs is None else probs
        if not isinstance(total_count, BMGNode):
            total_count = self.add_constant(total_count)
        if not isinstance(probability, BMGNode):
            probability = self.add_constant(probability)
        return self.add_binomial(total_count, probability, logits is not None)

    @memoize
    def add_categorical(
        self, probability: BMGNode, is_logits: bool = False
    ) -> CategoricalNode:
        node = CategoricalNode(probability, is_logits)
        self.add_node(node)
        return node

    def handle_categorical(
        self, probs: Any = None, logits: Any = None
    ) -> CategoricalNode:
        if (probs is None and logits is None) or (
            probs is not None and logits is not None
        ):
            raise ValueError(
                "handle_categorical requires exactly one of probs or logits"
            )
        probability = logits if probs is None else probs
        if not isinstance(probability, BMGNode):
            probability = self.add_constant(probability)
        return self.add_categorical(probability, logits is not None)

    @memoize
    def add_halfcauchy(self, scale: BMGNode) -> HalfCauchyNode:
        node = HalfCauchyNode(scale)
        self.add_node(node)
        return node

    def handle_halfcauchy(self, scale: Any, validate_args=None) -> HalfCauchyNode:
        if not isinstance(scale, BMGNode):
            scale = self.add_constant(scale)
        return self.add_halfcauchy(scale)

    @memoize
    def add_normal(self, mu: BMGNode, sigma: BMGNode) -> NormalNode:
        node = NormalNode(mu, sigma)
        self.add_node(node)
        return node

    def handle_normal(self, loc: Any, scale: Any, validate_args=None) -> NormalNode:
        if not isinstance(loc, BMGNode):
            loc = self.add_constant(loc)
        if not isinstance(scale, BMGNode):
            scale = self.add_constant(scale)
        return self.add_normal(loc, scale)

    @memoize
    def add_dirichlet(self, concentration: BMGNode) -> DirichletNode:
        node = DirichletNode(concentration)
        self.add_node(node)
        return node

    def handle_dirichlet(self, concentration: Any, validate_args=None) -> DirichletNode:
        if not isinstance(concentration, BMGNode):
            concentration = self.add_constant(concentration)
        return self.add_dirichlet(concentration)

    @memoize
    def add_studentt(self, df: BMGNode, loc: BMGNode, scale: BMGNode) -> StudentTNode:
        node = StudentTNode(df, loc, scale)
        self.add_node(node)
        return node

    def handle_studentt(
        self, df: Any, loc: Any = 0.0, scale: Any = 1.0, validate_args=None
    ) -> StudentTNode:
        if not isinstance(df, BMGNode):
            df = self.add_constant(df)
        if not isinstance(loc, BMGNode):
            loc = self.add_constant(loc)
        if not isinstance(scale, BMGNode):
            scale = self.add_constant(scale)
        return self.add_studentt(df, loc, scale)

    @memoize
    def add_uniform(self, low: BMGNode, high: BMGNode) -> UniformNode:
        node = UniformNode(low, high)
        self.add_node(node)
        return node

    def handle_uniform(self, low: Any, high: Any, validate_args=None) -> UniformNode:
        if not isinstance(low, BMGNode):
            low = self.add_constant(low)
        if not isinstance(high, BMGNode):
            high = self.add_constant(high)
        return self.add_uniform(low, high)

    @memoize
    def add_beta(self, alpha: BMGNode, beta: BMGNode) -> BetaNode:
        node = BetaNode(alpha, beta)
        self.add_node(node)
        return node

    def handle_beta(
        self, concentration1: Any, concentration0: Any, validate_args=None
    ) -> BetaNode:
        if not isinstance(concentration1, BMGNode):
            concentration1 = self.add_constant(concentration1)
        if not isinstance(concentration0, BMGNode):
            concentration0 = self.add_constant(concentration0)
        return self.add_beta(concentration1, concentration0)

    @memoize
    def add_addition(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value + right.value)
        node = AdditionNode(left, right)
        self.add_node(node)
        return node

    def handle_addition(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input + other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value + other.value
        return self.add_addition(input, other)

    @memoize
    def add_multiplication(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value * right.value)
        node = MultiplicationNode(left, right)
        self.add_node(node)
        return node

    def handle_multiplication(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input * other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value * other.value
        return self.add_multiplication(input, other)

    @memoize
    def add_if_then_else(
        self, condition: BMGNode, consequence: BMGNode, alternative: BMGNode
    ) -> BMGNode:
        if isinstance(condition, BooleanNode):
            return consequence if condition.value else alternative
        node = IfThenElseNode(condition, consequence, alternative)
        self.add_node(node)
        return node

    @memoize
    def add_matrix_multiplication(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(torch.mm(left.value, right.value))
        node = MatrixMultiplicationNode(left, right)
        self.add_node(node)
        return node

    def handle_matrix_multiplication(self, input: Any, mat2: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(mat2, BMGNode)):
            return torch.mm(input, mat2)
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(mat2, BMGNode):
            mat2 = self.add_constant(mat2)
        if isinstance(input, ConstantNode) and isinstance(mat2, ConstantNode):
            return torch.mm(input.value, mat2.value)
        return self.add_matrix_multiplication(input, mat2)

    @memoize
    def add_division(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value / right.value)
        node = DivisionNode(left, right)
        self.add_node(node)
        return node

    # TODO: Do we need to represent both integer and float division?
    def handle_division(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input / other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value / other.value
        return self.add_division(input, other)

    @memoize
    def add_power(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            return self.add_constant(left.value ** right.value)
        node = PowerNode(left, right)
        self.add_node(node)
        return node

    def handle_power(self, input: Any, exponent: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(exponent, BMGNode)):
            return input ** exponent
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(exponent, BMGNode):
            exponent = self.add_constant(exponent)
        if isinstance(input, ConstantNode) and isinstance(exponent, ConstantNode):
            return input.value ** exponent.value
        return self.add_power(input, exponent)

    @memoize
    def add_index(self, left: MapNode, right: BMGNode) -> BMGNode:
        # TODO: Is there a better way to say "if list length is 1" that bails out
        # TODO: if it is greater than 1?
        if len(list(right.support())) == 1:
            return left[right]
        node = IndexNode(left, right)
        self.add_node(node)
        return node

    def handle_index(self, left: Any, right: Any) -> Any:
        if isinstance(left, BMGNode) and not isinstance(left, MapNode):
            # TODO: Improve this error message
            raise ValueError("handle_index requires a collection")
        if isinstance(right, ConstantNode):
            result = left[right.value]
            return result.value if isinstance(result, ConstantNode) else result
        if not isinstance(right, BMGNode):
            result = left[right]
            return result.value if isinstance(result, ConstantNode) else result
        m = self.collection_to_map(left)
        return self.add_index(m, right)

    @memoize
    def add_negate(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(-operand.value)
        node = NegateNode(operand)
        self.add_node(node)
        return node

    def handle_negate(self, input: Any) -> Any:
        if not isinstance(input, BMGNode):
            return -input
        if isinstance(input, ConstantNode):
            return -input.value
        return self.add_negate(input)

    # TODO: What should the result of NOT on a tensor be?
    # TODO: Should it be legal at all in the graph?
    # TODO: In Python, (not tensor(x)) is equal to (not x).
    # TODO: It is NOT equal to (tensor(not x)), which is what
    # TODO: you might expect.
    @memoize
    def add_not(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(not operand.value)
        node = NotNode(operand)
        self.add_node(node)
        return node

    def handle_not(self, input: Any) -> Any:
        if not isinstance(input, BMGNode):
            return not input
        if isinstance(input, ConstantNode):
            return not input.value
        return self.add_not(input)

    @memoize
    def add_to_real(self, operand: BMGNode) -> ToRealNode:
        if isinstance(operand, RealNode):
            return operand
        if isinstance(operand, ConstantNode):
            return self.add_real(float(operand.value))
        node = ToRealNode(operand)
        self.add_node(node)
        return node

    def handle_to_real(self, operand: Any) -> Any:
        if not isinstance(operand, BMGNode):
            return float(operand)
        if isinstance(operand, ConstantNode):
            return float(operand.value)
        return self.add_to_real(operand)

    @memoize
    def add_to_tensor(self, operand: BMGNode) -> ToTensorNode:
        node = ToTensorNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_exp(self, operand: BMGNode) -> ExpNode:
        if isinstance(operand, TensorNode):
            return self.add_constant(torch.exp(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math.exp(operand.value))
        node = ExpNode(operand)
        self.add_node(node)
        return node

    def handle_exp(self, input: Any) -> Any:
        if isinstance(input, Tensor):
            return torch.exp(input)
        if isinstance(input, TensorNode):
            return torch.exp(input.value)
        if not isinstance(input, BMGNode):
            return math.exp(input)
        if isinstance(input, ConstantNode):
            return math.exp(input.value)
        return self.add_exp(input)

    @memoize
    def add_log(self, operand: BMGNode) -> ExpNode:
        if isinstance(operand, TensorNode):
            return self.add_constant(torch.log(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math.log(operand.value))
        node = LogNode(operand)
        self.add_node(node)
        return node

    def handle_log(self, input: Any) -> Any:
        if isinstance(input, Tensor):
            return torch.log(input)
        if isinstance(input, TensorNode):
            return torch.log(input.value)
        if not isinstance(input, BMGNode):
            return math.log(input)
        if isinstance(input, ConstantNode):
            return math.log(input.value)
        return self.add_log(input)

    def _canonicalize_function(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ):
        if kwargs is None:
            kwargs = {}
        if isinstance(function, KnownFunction):
            f = function.function
            args = [function.receiver] + arguments
        elif (
            isinstance(function, builtin_function_or_method)
            and isinstance(function.__self__, Tensor)
            and function.__name__ in known_tensor_instance_functions
        ):
            f = getattr(Tensor, function.__name__)
            args = [function.__self__] + arguments
        elif isinstance(function, Callable):
            f = function
            args = arguments
        else:
            raise ValueError(
                f"Function {function} is not supported by Bean Machine Graph."
            )
        return (f, args, kwargs)

    def handle_function(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any] = None
    ) -> Any:
        f, args, kwargs = self._canonicalize_function(function, arguments, kwargs)
        if is_ordinary_call(f, args, kwargs):
            return f(*args, **kwargs)
        # TODO: Do a sanity check that the arguments match and give
        # TODO: a good error if they do not. Alternatively, catch
        # TODO: the exception if the call fails and replace it with
        # TODO: a more informative error.
        if f in self.function_map:
            return self.function_map[f](*args, **kwargs)
        raise ValueError(f"Function {f} is not supported by Bean Machine Graph.")

    @memoize
    def add_map(self, *elements) -> MapNode:
        # TODO: Verify that the list is well-formed.
        node = MapNode(elements)
        self.add_node(node)
        return node

    def collection_to_map(self, collection) -> MapNode:
        if isinstance(collection, MapNode):
            return collection
        if isinstance(collection, list):
            copy = []
            for i in range(0, len(collection)):
                copy.append(self.add_constant(i))
                item = collection[i]
                node = item if isinstance(item, BMGNode) else self.add_constant(item)
                copy.append(node)
            return self.add_map(*copy)
        # TODO: Dictionaries? Tuples?
        raise ValueError("collection_to_map requires a list")

    # Do NOT memoize add_sample; each sample node must be unique
    def add_sample(self, operand: DistributionNode) -> SampleNode:
        node = SampleNode(operand)
        self.add_node(node)
        return node

    def handle_sample(self, operand: Any) -> SampleNode:
        if isinstance(operand, DistributionNode):
            return self.add_sample(operand)
        if isinstance(operand, Bernoulli):
            b = self.handle_bernoulli(operand.probs)
            return self.add_sample(b)
        if isinstance(operand, Binomial):
            b = self.handle_binomial(operand.total_count, operand.probs)
            return self.add_sample(b)
        if isinstance(operand, Categorical):
            b = self.handle_categorical(operand.probs)
            return self.add_sample(b)
        if isinstance(operand, Dirichlet):
            b = self.handle_dirichlet(operand.concentration)
            return self.add_sample(b)
        if isinstance(operand, HalfCauchy):
            b = self.handle_halfcauchy(operand.scale)
            return self.add_sample(b)
        if isinstance(operand, Normal):
            b = self.handle_normal(operand.mean, operand.stddev)
            return self.add_sample(b)
        if isinstance(operand, StudentT):
            b = self.handle_studentt(operand.df, operand.loc, operand.scale)
            return self.add_sample(b)
        if isinstance(operand, Uniform):
            b = self.handle_uniform(operand.low, operand.high)
            return self.add_sample(b)
        if isinstance(operand, Beta):
            b = self.handle_beta(operand.concentration1, operand.concentration0)
            return self.add_sample(b)
        raise ValueError(
            f"Operand {str(operand)} is not a valid target for a sample operation."
        )

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

        # TODO: There are a great many more pure instance functions on tensors;
        # TODO: which do we wish to support?

        if isinstance(operand, BMGNode):
            if operand.node_type == Tensor:
                if name in known_tensor_instance_functions:
                    return KnownFunction(operand, getattr(Tensor, name))
            raise ValueError(
                f"Fetching the value of attribute {name} is not "
                + "supported in Bean Machine Graph."
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

    def add_observation(self, observed: SampleNode, value: Any) -> Observation:
        node = Observation(observed, value)
        self.add_node(node)
        return node

    def handle_query(self, value: Any) -> Any:
        # When we have an @query function, we need to put a node
        # in the graph indicating that the value returned is of
        # interest to the user, and the inference engine should
        # accumulate information about it.  Under what circumstances
        # might that be useful? If the query function returns an
        # ordinary value, there is nothing we can infer from it.
        # But if it returns an operation in a probabilistic workflow,
        # that node in the graph should be marked as being of interest.
        #
        # Adding a query to a node is idempotent; querying twice is the
        # same as querying once, so the add_query method is memoized.
        #
        # Note that this function does not return the added query node.
        # The query node is just a marker, not a value; we just keep
        # using the value as it was before. We insert a node into the
        # graph if necessary and then hand back the argument.

        if isinstance(value, OperatorNode):
            self.add_query(value)
        return value

    @memoize
    def add_query(self, operator: OperatorNode) -> Query:
        node = Query(operator)
        self.add_node(node)
        return node

    def to_dot(self) -> str:
        db = DotBuilder()
        max_length = len(str(len(self.nodes) - 1))

        def to_id(index):
            return "N" + str(index).zfill(max_length)

        for node, index in self.nodes.items():
            n = to_id(index)
            db.with_node(n, node.label)
            for (child, label) in zip(node.children, node.edges):
                db.with_edge(n, to_id(self.nodes[child]), label)
        return str(db)

    def to_bmg(self) -> Graph:
        g = Graph()
        d: Dict[BMGNode, int] = {}
        self._fix_types()
        for node in self._traverse_from_roots():
            d[node] = node._add_to_graph(g, d)
        return g

    def _resort_nodes(self) -> Dict[BMGNode, int]:
        # Renumber the nodes so that the ids are in numerical order.
        sorted_nodes = {}
        for index, node in enumerate(self._traverse_from_roots()):
            sorted_nodes[node] = index
        return sorted_nodes

    def to_python(self) -> str:
        self._fix_types()
        header = """from beanmachine import graph
from torch import tensor
g = graph.Graph()
"""
        sorted_nodes = self._resort_nodes()
        return header + "\n".join(n._to_python(sorted_nodes) for n in sorted_nodes)

    def to_cpp(self) -> str:
        self._fix_types()
        sorted_nodes = self._resort_nodes()
        return "graph::Graph g;\n" + "\n".join(
            n._to_cpp(sorted_nodes) for n in sorted_nodes
        )

    def _traverse_from_roots(self) -> List[BMGNode]:
        result = []
        seen = set()

        def is_root(n: BMGNode) -> bool:
            return (
                isinstance(n, SampleNode)
                or isinstance(n, Observation)
                or isinstance(n, Query)
            )

        def key(n: BMGNode) -> int:
            return self.nodes[n]

        def visit(n: BMGNode) -> None:
            if n in seen:
                return
            for c in n.children:
                visit(c)
            seen.add(n)
            result.append(n)

        roots = sorted((n for n in self.nodes if is_root(n)), key=key, reverse=False)
        for r in roots:
            visit(r)
        return result

    # TODO: All the type fixing code which follows uses exceptions as the
    # TODO: error reporting mechanism. This is fine for now but eventually
    # TODO: we'll need to design a proper user-centric error reporting
    # TODO: mechanism.

    def _fix_types(self) -> None:
        # So far these rewrites can add nodes but none of them need further
        # rewriting. If this changes, we might need to iterate until
        # we reach a fixpoint.
        for node in self._traverse_from_roots():
            self._fix_type(node)

    def _fix_type(self, node: BMGNode) -> None:
        # A sample's type depends entirely on its distribution's
        # node_type, so just fix the distribution.
        if isinstance(node, SampleNode):
            self._fix_type(node.operand)
        if isinstance(node, BernoulliNode):
            self._fix_bernoulli(node)
        elif isinstance(node, BetaNode):
            self._fix_beta(node)
        if isinstance(node, BinomialNode):
            self._fix_binomial(node)
        elif isinstance(node, HalfCauchyNode):
            self._fix_half_cauchy(node)
        elif isinstance(node, NormalNode):
            self._fix_normal(node)
        elif isinstance(node, StudentTNode):
            self._fix_studentt(node)

    # Ensures that Bernoulli nodes take the appropriate
    # input type; once they do, they are updated to automatically
    # mark samples as being bools.
    def _fix_bernoulli(self, node: BernoulliNode) -> None:
        if node.types_fixed:
            return
        if node.is_logits:
            prob = self._ensure_real(node.probability)
        else:
            prob = self._ensure_probability(node.probability)
        node.probability = prob
        node.types_fixed = True

    # Ensures that binomial nodes take the appropriate
    # input types; once they do, they are updated to automatically
    # mark samples as being naturals.
    def _fix_binomial(self, node: BinomialNode) -> None:
        if node.types_fixed:
            return
        if node.is_logits:
            raise ValueError("Logits binomial is not yet implemented.")
            # prob = self._ensure_real(node.probability)
        else:
            prob = self._ensure_probability(node.probability)
        node.count = self._ensure_natural(node.count)
        node.probability = prob
        node.types_fixed = True

    def _fix_beta(self, node: BetaNode) -> None:
        if node.types_fixed:
            return
        node.alpha = self._ensure_pos_real(node.alpha)
        node.beta = self._ensure_pos_real(node.beta)
        node.types_fixed = True

    def _fix_half_cauchy(self, node: HalfCauchyNode) -> None:
        if node.types_fixed:
            return
        node.scale = self._ensure_pos_real(node.scale)
        node.types_fixed = True

    def _fix_normal(self, node: NormalNode) -> None:
        if node.types_fixed:
            return
        node.mu = self._ensure_real(node.mu)
        node.sigma = self._ensure_pos_real(node.sigma)
        node.types_fixed = True

    def _fix_studentt(self, node: StudentTNode) -> None:
        if node.types_fixed:
            return
        node.df = self._ensure_pos_real(node.df)
        node.loc = self._ensure_real(node.loc)
        node.scale = self._ensure_pos_real(node.scale)
        node.types_fixed = True

    def _ensure_probability(self, node: BMGNode) -> BMGNode:
        # The fixed type might already be correct.
        self._fix_type(node)
        if node.node_type == Probability:
            return node
        if isinstance(node, ConstantNode):
            return self._constant_to_probability(node)
        raise ValueError("Conversion to probability node not yet implemented.")

    def _constant_to_probability(self, node: ConstantNode) -> ProbabilityNode:
        if isinstance(node, TensorNode):
            if node.value.shape.numel() != 1:
                raise ValueError(
                    "To use a tensor as a probability it must "
                    + "have exactly one element."
                )
        v = float(node.value)
        if v < 0.0 or v > 1.0:
            raise ValueError("A probability must be between 0.0 and 1.0.")
        return self.add_probability(v)

    def _ensure_real(self, node: BMGNode) -> BMGNode:
        # The fixed type might already be correct.
        self._fix_type(node)
        if node.node_type == float:
            return node
        if isinstance(node, ConstantNode):
            return self._constant_to_real(node)
        if node.node_type == bool:
            return self._bool_to_real(node)
        raise ValueError("Conversion to real node not yet implemented.")

    def _bool_to_real(self, node: BMGNode) -> BMGNode:
        # TODO: If there is a query node pointing to the original node
        # then it needs to be retargetted to the new one.
        one = self.add_real(1.0)
        zero = self.add_real(0.0)
        return self.add_if_then_else(node, one, zero)

    def _constant_to_real(self, node: ConstantNode) -> RealNode:
        if isinstance(node, TensorNode):
            if node.value.shape.numel() != 1:
                raise ValueError(
                    "To use a tensor as a real number it must "
                    + "have exactly one element."
                )
        v = float(node.value)
        return self.add_real(v)

    def _ensure_pos_real(self, node: BMGNode) -> BMGNode:
        # The fixed type might already be correct.
        self._fix_type(node)
        if node.node_type == PositiveReal:
            return node
        if isinstance(node, ConstantNode):
            return self._constant_to_pos_real(node)
        raise ValueError("Conversion to positive real node not yet implemented.")

    def _constant_to_pos_real(self, node: ConstantNode) -> PositiveRealNode:
        if isinstance(node, TensorNode):
            if node.value.shape.numel() != 1:
                raise ValueError(
                    "To use a tensor as a positive real number it must "
                    + "have exactly one element."
                )
        v = float(node.value)
        if v < 0.0:
            raise ValueError("A positive real must be greater than 0.0.")
        return self.add_pos_real(v)

    def _ensure_natural(self, node: BMGNode) -> BMGNode:
        # The fixed type might already be correct.
        self._fix_type(node)
        if node.node_type == Natural:
            return node
        if isinstance(node, ConstantNode):
            return self._constant_to_natural(node)
        raise ValueError("Conversion to natural node not yet implemented.")

    def _constant_to_natural(self, node: ConstantNode) -> NaturalNode:
        # The fixed type might already be correct.
        self._fix_type(node)
        if isinstance(node, TensorNode):
            if node.value.shape.numel() != 1:
                raise ValueError(
                    "To use a tensor as a natural number it must "
                    + "have exactly one element."
                )
        v = float(node.value)
        if v < 0.0:
            raise ValueError("A natural must be positive.")
        if not v.is_integer():
            raise ValueError("A natural must be an integer.")
        return self.add_natural(int(v))
