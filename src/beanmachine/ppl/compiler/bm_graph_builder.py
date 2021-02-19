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
import inspect
import math
import sys
from types import MethodType
from typing import Any, Callable, Dict, List

# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this mModuleNotFoundError
# pyre-ignore-all-errors
# TODO: It is somewhat confusing to import a type named "Graph" here;
# Consider renaming it to NativeGraph or some other more descriptive
# name that tells the reader that this is a really-truly BMG graph
# that we are constructing in memory.
from beanmachine.graph import Graph, InferenceType
from beanmachine.ppl.compiler.bmg_nodes import (
    AdditionNode,
    BernoulliNode,
    BetaNode,
    BinomialNode,
    BMGNode,
    BooleanNode,
    CategoricalNode,
    Chi2Node,
    ComplementNode,
    ConstantNode,
    ConstantTensorNode,
    DirichletNode,
    DistributionNode,
    DivisionNode,
    EqualNode,
    ExpM1Node,
    ExpNode,
    FlatNode,
    GammaNode,
    GreaterThanEqualNode,
    GreaterThanNode,
    HalfCauchyNode,
    IfThenElseNode,
    IndexNode,
    LessThanEqualNode,
    LessThanNode,
    LogisticNode,
    LogNode,
    LogSumExpNode,
    MapNode,
    MatrixMultiplicationNode,
    MultiplicationNode,
    NaturalNode,
    NegateNode,
    NegativeRealNode,
    NormalNode,
    NotEqualNode,
    NotNode,
    Observation,
    OperatorNode,
    PhiNode,
    PositiveRealNode,
    PowerNode,
    ProbabilityNode,
    Query,
    RealNode,
    SampleNode,
    StudentTNode,
    TensorNode,
    ToPositiveRealNode,
    ToProbabilityNode,
    ToRealNode,
    UniformNode,
    positive_infinity,
)
from beanmachine.ppl.compiler.bmg_types import (
    BMGLatticeType,
    Boolean,
    Natural,
    NegativeReal,
    One,
    PositiveReal,
    Probability,
    Real,
    Zero,
)
from beanmachine.ppl.inference.monte_carlo_samples import MonteCarloSamples
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from beanmachine.ppl.utils.beanstalk_common import allowed_functions
from beanmachine.ppl.utils.dotbuilder import DotBuilder
from beanmachine.ppl.utils.memoize import MemoizationKey, memoize
from torch import Tensor, tensor
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Chi2,
    Dirichlet,
    Gamma,
    HalfCauchy,
    Normal,
    StudentT,
    Uniform,
)


def _flatten_all_lists(lst):
    """Takes a list-of-lists, with arbitrary nesting level;
    returns an iteration of all elements."""
    for item in lst:
        if isinstance(item, list):
            yield from _flatten_all_lists(item)
        else:
            yield item


def _list_to_zeros(lst):
    """Takes a list-of-lists, with arbitrary nesting level;
    returns a list-of-lists of the same shape but with every non-list
    element replaced with zero."""
    return [_list_to_zeros(item) if isinstance(item, list) else 0 for item in lst]


builtin_function_or_method = type(abs)

# If we encounter a function call on a stochastic node during
# graph accumulation we will attempt to build a node in the graph
# for that invocation. These are the instance methods on tensors
# that we recognize and can build a graph node for.  See
# KnownFunction below for more details.

known_tensor_instance_functions = [
    "add",
    "div",
    "exp",
    "expm1",
    "float",
    "log",
    "logical_not",
    "logsumexp",
    "mm",
    "mul",
    "neg",
    "pow",
    "sigmoid",
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


def is_from_lifted_module(f) -> bool:
    return (
        hasattr(f, "__module__")
        and f.__module__ in sys.modules
        and hasattr(sys.modules[f.__module__], "_lifted_to_bmg")
    )


def only_ordinary_arguments(args, kwargs) -> bool:
    if any(isinstance(arg, BMGNode) for arg in args):
        return False
    if any(isinstance(arg, BMGNode) for arg in kwargs.values()):
        return False
    return True


def _is_random_variable_call(f) -> bool:
    return hasattr(f, "is_random_variable") and f.is_random_variable


def _is_functional_call(f) -> bool:
    return hasattr(f, "is_functional") and f.is_functional


def _is_phi(f: Any, arguments: List[Any], kwargs: Dict[str, Any]) -> bool:
    # We need to know if this call is Normal.cdf(Normal(0.0, 1.0), x).
    # (Note that we have already rewritten Normal(0.0, 1.0).cdf(x) into
    # this form.)
    # TODO: Support kwargs
    if f is not Normal.cdf or len(arguments) < 2:
        return False
    s = arguments[0]
    return isinstance(s, Normal) and s.mean == 0.0 and s.stddev == 1.0


def _has_source_code(function: Callable) -> bool:
    try:
        inspect.getsource(function)
    except Exception:
        return False
    return True


standard_normal = Normal(0.0, 1.0)


def phi(x: Any) -> Any:
    return standard_normal.cdf(x)


class BMGraphBuilder:
    """A graph builder which accumulates stochastic graph nodes as a model executes,
    and then transforms that into a valid Bean Machine Graph."""

    # ####
    # #### State and initialization
    # ####

    # We keep a list of all the nodes in the graph and associate a unique
    # integer with each.

    # TODO: The original idea was to use these integers when generating code
    # that constructs the graph, or DOT files that display the graph.
    # However, the integer generated is ordered according to when the node
    # was created, which is not necessarily the order in which we would
    # like to enumerate them for code generation purposes.
    #
    # We have therefore changed the code generation process to do a deterministic
    # topological sort of the nodes, and then number them in topological sort
    # order when emitting code; that way the code is generated so that each node
    # is numbered in the order it appears in the code. This is more pleasant
    # to read and understand, but the real benefit is that it makes the test
    # cases more stable and easier to verify.
    #
    # We can replace this dictionary with an unordered set; consider doing so.

    nodes: Dict[BMGNode, int]

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

    function_map: Dict[Callable, Callable]

    # As we construct the graph we may encounter "random variable" values; these
    # refer to a function that we need to transform into the "lifted" form. This
    # map tracks those so that we do not repeat work. However, RVIDs contain a
    # tuple of arguments which might contain tensors, and tensors are hashed by
    # reference, not by value. We therefore construct a map of RVID-equivalents
    # which is hashable by the values of the arguments.

    rv_map: Dict[MemoizationKey, BMGNode]
    lifted_map: Dict[Callable, Callable]

    # We also need to keep track of which query nodes are associated
    # with which RVIDs:

    query_rv_map: Dict[Query, RVIdentifier]

    def __init__(self) -> None:
        self.rv_map = {}
        self.lifted_map = {}
        self.nodes = {}
        self.query_rv_map = {}
        self.function_map = {
            # Math functions
            math.exp: self.handle_exp,
            math.log: self.handle_log,
            # Tensor instance functions
            torch.Tensor.add: self.handle_addition,
            torch.Tensor.div: self.handle_division,
            torch.Tensor.exp: self.handle_exp,
            torch.Tensor.expm1: self.handle_expm1,
            torch.Tensor.float: self.handle_to_real,
            torch.Tensor.logical_not: self.handle_not,
            torch.Tensor.log: self.handle_log,
            torch.Tensor.logsumexp: self.handle_logsumexp,
            torch.Tensor.mm: self.handle_matrix_multiplication,
            torch.Tensor.mul: self.handle_multiplication,
            torch.Tensor.neg: self.handle_negate,
            torch.Tensor.pow: self.handle_power,
            torch.Tensor.sigmoid: self.handle_logistic,
            # Tensor static functions
            torch.add: self.handle_addition,
            torch.div: self.handle_division,
            torch.exp: self.handle_exp,
            torch.expm1: self.handle_expm1,
            # Note that torch.float is not a function.
            torch.log: self.handle_log,
            torch.logsumexp: self.handle_logsumexp,
            torch.logical_not: self.handle_not,
            torch.mm: self.handle_matrix_multiplication,
            torch.mul: self.handle_multiplication,
            torch.neg: self.handle_negate,
            torch.pow: self.handle_power,
            torch.sigmoid: self.handle_logistic,
            # Distribution constructors
            Bernoulli: self.handle_bernoulli,
            Beta: self.handle_beta,
            Binomial: self.handle_binomial,
            Categorical: self.handle_categorical,
            Dirichlet: self.handle_dirichlet,
            Chi2: self.handle_chi2,
            Gamma: self.handle_gamma,
            HalfCauchy: self.handle_halfcauchy,
            Normal: self.handle_normal,
            StudentT: self.handle_studentt,
            Uniform: self.handle_uniform,
        }

    # ####
    # #### Node creation and accumulation
    # ####

    # This code is called while the lifted program executes.
    #
    # The "add" methods unconditionally create a new graph node
    # and add to the builder *if it does not already exist*.
    # By memoizing almost all the "add" methods we ensure that
    # the graph is deduplicated automatically.
    #
    # The "handle" methods, in contrast, conditionally create new
    # graph nodes only when required because an operation on a
    # stochastic value must be accumulated into the graph.
    #
    # For example, if we have a call to handle_addition where all
    # operands are ordinary floats (or constant graph nodes)
    # then there is no need to add a new node to the graph. But
    # if we have an addition of 1.0 to a stochastic node -- perhaps
    # a sample node, or perhaps some other graph node that eventually
    # involves a sample node -- then we need to construct a new
    # addition node, which is then returned and becomes the value
    # manipulated by the executing lifted program.
    #
    # TODO: The code in the "handle" methods which folds operations
    # on constant nodes and regular values is a holdover from an
    # earlier prototyping stage in which all values were lifted to
    # graph nodes. These scenarios should now be impossible, and we
    # should take a work item to remove this now-unnecessary code.

    def add_node(self, node: BMGNode) -> None:
        """This adds a node we've recently created to the node set;
        it maintains the invariant that all the input nodes are also added."""
        if node not in self.nodes:
            for i in node.inputs:
                self.add_node(i)
            self.nodes[node] = len(self.nodes)

    # ####
    # #### Graph accumulation for constant values
    # ####

    # This code handles creating nodes for ordinary values such as
    # floating point values and tensors created during the execution
    # of the lifted program. We only create graph nodes for an ordinary
    # value when that value is somehow involved in a stochastic
    # operation.

    # During the execution of the lifted program we should only be
    # creating nodes for real, Boolean and tensor values; during the
    # phase where we ensure that the BMG type system constraints are met
    # we construct probability, postive real, and natural number nodes.

    def add_constant(self, value: Any) -> ConstantNode:
        """This takes any constant value of a supported type,
        creates a constant graph node for it, and adds it to the builder"""
        if isinstance(value, bool):
            return self.add_boolean(value)
        if isinstance(value, int):
            return self.add_real(value)
        if isinstance(value, float):
            return self.add_real(value)
        if isinstance(value, Tensor):
            return self.add_constant_tensor(value)
        raise TypeError("value must be a bool, real or tensor")

    def add_constant_of_type(
        self, value: Any, node_type: BMGLatticeType
    ) -> ConstantNode:
        """This takes any constant value of a supported type and creates a
        constant graph node of the stated type for it, and adds it to the builder"""
        if node_type == Boolean:
            return self.add_boolean(bool(value))
        if node_type == Probability:
            return self.add_probability(float(value))
        if node_type == Natural:
            return self.add_natural(int(value))
        if node_type == PositiveReal:
            return self.add_pos_real(float(value))
        if node_type == NegativeReal:
            return self.add_neg_real(float(value))
        if node_type == Real:
            return self.add_real(float(value))
        if node_type == Tensor:
            if isinstance(value, Tensor):
                return self.add_constant_tensor(value)
            return self.add_constant_tensor(tensor(value))
        raise TypeError("node type must be a valid BMG type")

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
    def add_neg_real(self, value: float) -> NegativeRealNode:
        node = NegativeRealNode(value)
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
    def add_constant_tensor(self, value: Tensor) -> ConstantTensorNode:
        node = ConstantTensorNode(value)
        self.add_node(node)
        return node

    # ####
    # #### Graph accumulation for distributions
    # ####

    # TODO: This code is mostly but not entirely in alpha order
    # by distribution type; we might reorganize it to make it
    # slightly easier to follow.

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

    # TODO: Add a note here describing why it is important that the function
    # signatures of the handler methods for distributions match those of
    # the torch distribution constructors.

    # TODO: Verify that they do match.

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
    def add_chi2(self, df: BMGNode) -> Chi2Node:
        node = Chi2Node(df)
        self.add_node(node)
        return node

    def handle_chi2(self, df: Any, validate_args=None) -> Chi2Node:
        if not isinstance(df, BMGNode):
            df = self.add_constant(df)
        return self.add_chi2(df)

    @memoize
    def add_gamma(self, concentration: BMGNode, rate: BMGNode) -> GammaNode:
        node = GammaNode(concentration, rate)
        self.add_node(node)
        return node

    def handle_gamma(
        self, concentration: Any, rate: Any, validate_args=None
    ) -> GammaNode:
        if not isinstance(concentration, BMGNode):
            concentration = self.add_constant(concentration)
        if not isinstance(rate, BMGNode):
            rate = self.add_constant(rate)
        return self.add_gamma(concentration, rate)

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
    def add_flat(self) -> FlatNode:
        node = FlatNode()
        self.add_node(node)
        return node

    # ####
    # #### Graph accumulation for operators
    # ####

    # The handler methods here are both invoked directly, when, say
    # there was an explicit addition in the original model, and
    # indirectly as the result of processing a function call such
    # as tensor.add.

    # TODO: This code is not very well organized; consider sorting it
    # into alpha order by operation.

    @memoize
    def add_greater_than(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value > right.value)

        node = GreaterThanNode(left, right)
        self.add_node(node)
        return node

    def handle_greater_than(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input > other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value > other.value
        return self.add_greater_than(input, other)

    @memoize
    def add_greater_than_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value >= right.value)

        node = GreaterThanEqualNode(left, right)
        self.add_node(node)
        return node

    def handle_greater_than_equal(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input >= other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value >= other.value
        return self.add_greater_than_equal(input, other)

    @memoize
    def add_less_than(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value < right.value)

        node = LessThanNode(left, right)
        self.add_node(node)
        return node

    def handle_less_than(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input < other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value < other.value
        return self.add_less_than(input, other)

    @memoize
    def add_less_than_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value <= right.value)

        node = LessThanEqualNode(left, right)
        self.add_node(node)
        return node

    def handle_less_than_equal(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input <= other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value <= other.value
        return self.add_less_than_equal(input, other)

    @memoize
    def add_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value == right.value)

        node = EqualNode(left, right)
        self.add_node(node)
        return node

    def handle_equal(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input == other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value == other.value
        return self.add_equal(input, other)

    @memoize
    def add_not_equal(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value != right.value)

        node = NotEqualNode(left, right)
        self.add_node(node)
        return node

    def handle_not_equal(self, input: Any, other: Any) -> Any:
        if (not isinstance(input, BMGNode)) and (not isinstance(other, BMGNode)):
            return input != other
        if not isinstance(input, BMGNode):
            input = self.add_constant(input)
        if not isinstance(other, BMGNode):
            other = self.add_constant(other)
        if isinstance(input, ConstantNode) and isinstance(other, ConstantNode):
            return input.value != other.value
        return self.add_not_equal(input, other)

    @memoize
    def add_addition(self, left: BMGNode, right: BMGNode) -> BMGNode:
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value + right.value)
            if left.inf_type == Zero:
                return right
        if isinstance(right, ConstantNode):
            if right.inf_type == Zero:
                return left

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
        if isinstance(left, ConstantNode):
            if isinstance(right, ConstantNode):
                return self.add_constant(left.value * right.value)
            t = left.inf_type
            if t == One:
                return right
            if t == Zero:
                return left
        if isinstance(right, ConstantNode):
            t = right.inf_type
            if t == One:
                return left
            if t == Zero:
                return right
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

    # TODO: Indexes and maps are slightly tricky; add some comments here
    # explaining what's going on.

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

    @memoize
    def add_complement(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(1 - operand.value)
        node = ComplementNode(operand)
        self.add_node(node)
        return node

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

    # TODO: Add some comments here explaining under what circumstances
    # we accumulate these conversion nodes.

    @memoize
    def add_to_real(self, operand: BMGNode) -> BMGNode:
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
    def add_to_positive_real(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, PositiveRealNode):
            return operand
        if isinstance(operand, ConstantNode):
            return self.add_positive_real(float(operand.value))
        node = ToPositiveRealNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_to_probability(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ProbabilityNode):
            return operand
        if isinstance(operand, ConstantNode):
            return self.add_probability(float(operand.value))
        node = ToProbabilityNode(operand)
        self.add_node(node)
        return node

    @memoize
    def add_exp(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantTensorNode):
            return self.add_constant(torch.exp(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math.exp(operand.value))
        node = ExpNode(operand)
        self.add_node(node)
        return node

    def handle_exp(self, input: Any) -> Any:
        if isinstance(input, Tensor):
            return torch.exp(input)
        if isinstance(input, ConstantTensorNode):
            return torch.exp(input.value)
        if not isinstance(input, BMGNode):
            return math.exp(input)
        if isinstance(input, ConstantNode):
            return math.exp(input.value)
        return self.add_exp(input)

    @memoize
    def add_expm1(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantTensorNode):
            return self.add_constant(torch.expm1(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(torch.expm1(torch.tensor(operand.value)))
        node = ExpM1Node(operand)
        self.add_node(node)
        return node

    def handle_expm1(self, input: Any) -> Any:
        if isinstance(input, Tensor):
            return torch.expm1(input)
        if isinstance(input, ConstantTensorNode):
            return torch.expm1(input.value)
        if not isinstance(input, BMGNode):
            return torch.expm1(input)
        if isinstance(input, ConstantNode):
            return torch.expm1(input.value)
        return self.add_expm1(input)

    @memoize
    def add_logistic(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantTensorNode):
            return self.add_constant(torch.sigmoid(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(torch.sigmoid(torch.tensor(operand.value)))
        node = LogisticNode(operand)
        self.add_node(node)
        return node

    def handle_logistic(self, input: Any) -> Any:
        if isinstance(input, Tensor):
            return torch.sigmoid(input)
        if isinstance(input, ConstantTensorNode):
            return torch.sigmoid(input.value)
        if not isinstance(input, BMGNode):
            return torch.sigmoid(input)
        if isinstance(input, ConstantNode):
            return torch.sigmoid(input.value)
        return self.add_logistic(input)

    @memoize
    def add_phi(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantNode):
            return self.add_constant(phi(operand.value))
        node = PhiNode(operand)
        self.add_node(node)
        return node

    def handle_phi(self, input: Any) -> Any:
        if not isinstance(input, BMGNode):
            return phi(input)
        if isinstance(input, ConstantNode):
            return phi(input.value)
        return self.add_phi(input)

    @memoize
    def add_log(self, operand: BMGNode) -> BMGNode:
        if isinstance(operand, ConstantTensorNode):
            return self.add_constant(torch.log(operand.value))
        if isinstance(operand, ConstantNode):
            return self.add_constant(math.log(operand.value))
        node = LogNode(operand)
        self.add_node(node)
        return node

    def handle_log(self, input: Any) -> Any:
        if isinstance(input, Tensor):
            return torch.log(input)
        if isinstance(input, ConstantTensorNode):
            return torch.log(input.value)
        if not isinstance(input, BMGNode):
            return math.log(input)
        if isinstance(input, ConstantNode):
            return math.log(input.value)
        return self.add_log(input)

    @memoize
    def add_tensor(self, size: torch.Size, *data: BMGNode) -> TensorNode:
        node = TensorNode(list(data), size)
        self.add_node(node)
        return node

    @memoize
    def add_logsumexp(self, *inputs: BMGNode) -> TensorNode:
        node = LogSumExpNode(list(inputs))
        self.add_node(node)
        return node

    def handle_logsumexp(self, input: Any, dim: Any, keepdim: Any = False) -> Any:
        # TODO: Handle the situation where dim or keepdim are graph nodes.
        # Produce an error.
        if not isinstance(input, BMGNode):
            return torch.logsumexp(input=input, dim=dim, keepdim=keepdim)
        if isinstance(input, ConstantTensorNode):
            return torch.logsumexp(input=input.value, dim=dim, keepdim=keepdim)
        if isinstance(input, ConstantNode):
            return torch.logsumexp(input=tensor(input.value), dim=dim, keepdim=keepdim)
        # TODO: Handle the situation where the dim is not 1 or the shape is not
        # one-dimensional; produce an error.
        return self.add_logsumexp(*[input])

    def _canonicalize_function(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ):
        """The purpose of this helper method is to uniformly handle all
        possibility for function calls. That is, the receiver can be stochastic,
        normal, or missing, and the arguments can be stochastic or normal values,
        and can be positional or named. We take in an object representing the function
        (possibly a KnownFunction helper that is bound to a receiver), and the arguments.
        We produce a function object that requires no receiver, and an argument list
        that has the receiver, if any, as its first member."""
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
        elif isinstance(function, MethodType):
            # In Python, if we are calling a method of a class with a "self"
            # parameter then the callable we get is already partially evaluated.
            # That is, if we have
            #
            # class C:
            #   def m(self, x):...
            # c = C();
            # cm = c.m
            # cm(1)
            #
            # Then cm is a brand-new object with attributes __self__ and __func__,
            # and calling cm(1) actually calls cm.__func__(cm.__self__, 1)
            #
            # We simulate that here.
            f = function.__func__
            args = [function.__self__] + arguments

        elif isinstance(function, Callable):
            f = function
            args = arguments
        else:
            raise ValueError(
                f"Function {function} is not supported by Bean Machine Graph."
            )
        return (f, args, kwargs)

    def _handle_random_variable_call_checked(
        self, function: Any, arguments: List[Any]
    ) -> BMGNode:
        assert isinstance(arguments, list)

        # We have a call to a random variable function. There are two
        # cases. Either we have only ordinary values for arguments, or
        # we have one or more graph nodes.
        if only_ordinary_arguments(arguments, {}):
            rv = function(*arguments)
            assert isinstance(rv, RVIdentifier)
            return self._rv_to_node(rv)

        # TODO: We have a finite stochastic control flow; we need to make many
        # TODO: calls to the random variable function and record the
        # TODO: results for each.  Right now this is handled by the
        # TODO: generated @probabilistic decorator, but that mechanism
        # TODO: should be removed and instead inlined in this function.
        # TODO: Just throw and we'll fix it later.

        raise ValueError("Jitted stochastic control flows are not yet implemented")

    def _handle_random_variable_call(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> BMGNode:

        if len(kwargs) != 0:
            # TODO: Better error
            raise ValueError(
                "Random variable function calls must not have named arguments."
            )

        # If we have one or more graph nodes as arguments to an RV function call
        # then we need to try every possible value for those arguments. We require
        # that there be a finite number of possibilities, and that the total number
        # of branches generated for this call is small. Check that *once* before
        # recursively processing the call one argument at a time.

        # TODO: Make this a global tweakable setting of the accumulator.
        max_possibilities = 1000
        possibilities = 1
        for arg in arguments:
            if isinstance(arg, BMGNode):
                possibilities *= arg.support_size()
                if possibilities == positive_infinity:
                    # TODO: Better exception
                    raise ValueError(
                        "Stochastic control flow must have finite support."
                    )
                if possibilities > max_possibilities:
                    # TODO: Better exception
                    raise ValueError("Stochastic control flow is too complex.")

        return self._handle_random_variable_call_checked(function, arguments)

    def _handle_functional_call(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> BMGNode:

        if len(kwargs) != 0:
            # TODO: Better error
            raise ValueError("Functional calls must not have named arguments.")

        # TODO: What happens if we have a @functional that does not return
        # TODO: a graph node? A functional that returns a constant is
        # TODO: weird, but it should be legal.  Figure out what the
        # TODO: right thing to do is for this scenario.

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
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
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
        if _has_source_code(function):
            return self._function_to_bmg_function(function)(*arguments, **kwargs)
        # It is not compiled and we have no source code to compile.
        # Just call it and hope for the best.
        # TODO: Do we need to consider the scenario where we do not have
        # source code, we call a function, and it somehow returns an RVID?
        # We *could* convert that to a graph node.
        return function(*arguments, **kwargs)

    def _handle_tensor_constructor(
        self, arguments: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(arguments, list)
        # TODO: Handle kwargs
        flattened_args = list(_flatten_all_lists(arguments))
        if not any(isinstance(arg, BMGNode) for arg in flattened_args):
            # None of the arguments are graph nodes. We can just
            # construct the tensor normally.
            return tensor(*arguments, **kwargs)
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
                flattened_args[index] = self.add_constant(arg)

        # What shape is this tensor? Rather than duplicating the logic in the
        # tensor class, let's just construct the same shape made of entirely
        # zeros and then ask what shape it is.
        size = tensor(_list_to_zeros(arguments)).size()

        return self.add_tensor(size, *flattened_args)

    def handle_function(
        self, function: Any, arguments: List[Any], kwargs: Dict[str, Any] = None
    ) -> Any:
        f, args, kwargs = self._canonicalize_function(function, arguments, kwargs)

        if _is_phi(f, args, kwargs):
            return self.handle_phi(*(args[1:]), **kwargs)

        if is_from_lifted_module(f):
            # It's already compiled; just call it.
            return function(*arguments, **kwargs)

        if _is_random_variable_call(f):
            return self._handle_random_variable_call(f, args, kwargs)

        if _is_functional_call(f):
            return self._handle_functional_call(f, args, kwargs)

        # If we get here, we have a function call from a module that
        # is not already compiled, and it is not a random variable
        # or functional.

        # Some functions are perfectly safe for a graph node.
        # We do not need to compile them.

        if f in allowed_functions:
            return f(*args, **kwargs)

        # We have special processing if we're trying to create a tensor;
        # if any element of the new tensor is a graph node then we'll
        # need to create a TensorNode.

        if f is tensor:
            return self._handle_tensor_constructor(args, kwargs)

        # Some functions we already have special-purpose handlers for,
        # like calls to math.exp or tensor.log. If we know there are no
        # graph nodes in the arguments we can just call the function directly
        # and get the values. If there are graph nodes in the arguments then
        # we can call our special handlers.

        # TODO: Do a sanity check that the arguments match and give
        # TODO: a good error if they do not. Alternatively, catch
        # TODO: the exception if the call fails and replace it with
        # TODO: a more informative error.
        if f in self.function_map:
            if only_ordinary_arguments(args, kwargs):
                return f(*args, **kwargs)
            return self.function_map[f](*args, **kwargs)

        return self._handle_ordinary_call(f, args, kwargs)

    def _function_to_bmg_function(self, function: Callable) -> Callable:
        from beanmachine.ppl.compiler.bm_to_bmg import _bm_function_to_bmg_function

        # TODO: What happens if the function is a member of a class?
        # TODO: Do we recompile it for different instances of the
        # TODO: receiver? Do we recompile it on every call? Check this.
        if function not in self.lifted_map:
            self.lifted_map[function] = _bm_function_to_bmg_function(function, self)
        return self.lifted_map[function]

    def _rv_to_node(self, rv: RVIdentifier) -> BMGNode:
        key = MemoizationKey(rv.wrapper, rv.arguments)
        if key not in self.rv_map:
            value = self._function_to_bmg_function(rv.function)(*rv.arguments)
            if isinstance(value, Tensor):
                value = self.add_constant_tensor(value)
            if not isinstance(value, BMGNode):
                raise TypeError("A functional must return a tensor.")
            self.rv_map[key] = value
            return value
        return self.rv_map[key]

    @memoize
    def add_map(self, *elements: BMGNode) -> MapNode:
        # TODO: Verify that the list is well-formed.
        node = MapNode(list(elements))
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

    def handle_sample(self, operand: Any) -> SampleNode:  # noqa
        """As we execute the lifted program, this method is called every
        time a model function decorated with @bm.random_variable returns; we verify that the
        returned value is a distribution that we know how to accumulate into the
        graph, and add a sample node to the graph."""
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
        if isinstance(operand, Chi2):
            b = self.handle_chi2(operand.df)
            return self.add_sample(b)
        if isinstance(operand, Gamma):
            b = self.handle_gamma(operand.concentration, operand.rate)
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
        # TODO: Get this into alpha order
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
            # If we're invoking a function on a graph node during execution of
            # the lifted program, that graph node is almost certainly a tensor
            # in the original program; assume that it is, and see if this is
            # a function on a tensor that we know how to accumulate into the graph.
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

    # TODO: Should this be idempotent?
    # TODO: Should it be an error to add two unequal observations to one node?
    def add_observation(self, observed: SampleNode, value: Any) -> Observation:
        node = Observation(observed, value)
        self.add_node(node)
        return node

    @memoize
    def add_query(self, operator: BMGNode) -> Query:
        # TODO: BMG requires that the target of a query be classified
        # as an operator and that queries be unique; that is, every node
        # is queried *exactly* zero or one times. Rather than making
        # those restrictions here, instead detect bad queries in the
        # problem fixing phase and report accordingly.
        node = Query(operator)
        self.add_node(node)
        return node

    # ####
    # #### Output
    # ####

    # The graph builder can either construct the BMG nodes in memory
    # directly, or return a Python or C++ program which does so.
    #
    # In addition, for debugging purposes we support dumping the
    # graph as a DOT graph description.
    #
    # Note that there is a slight difference here; the DOT dump
    # can give the entire graph that we have accumulated, including
    # any orphaned nodes, or just the graph as it would be generated
    # in BMG, depending on the value of the after_transform flag.

    # The other output formats do a topological
    # sort of the graph nodes and only output the nodes which are
    # inputs into samples, queries, and observations.

    def to_dot(
        self,
        graph_types: bool = False,
        inf_types: bool = False,
        edge_requirements: bool = False,
        point_at_input: bool = False,
        after_transform: bool = False,
        label_edges: bool = True,
    ) -> str:
        """This dumps the entire accumulated graph state, including
        orphans, as a DOT graph description; nodes are enumerated in the order
        they were created."""
        db = DotBuilder()

        if after_transform:
            from beanmachine.ppl.compiler.fix_problems import fix_problems

            fix_problems(self).raise_errors()
            nodes = self._resort_nodes()
        else:
            nodes = self.nodes

        max_length = len(str(len(nodes) - 1))

        def to_id(index) -> str:
            return "N" + str(index).zfill(max_length)

        for node, index in nodes.items():
            n = to_id(index)
            node_label = node.label
            if graph_types:
                node_label += ":" + node.graph_type.short_name
            if inf_types:
                node_label += ">=" + node.inf_type.short_name
            db.with_node(n, node_label)
            for (i, edge_name, req) in zip(node.inputs, node.edges, node.requirements):
                if label_edges:
                    edge_label = edge_name
                    if edge_requirements:
                        edge_label += ":" + req.short_name
                elif edge_requirements:
                    edge_label = req.short_name
                else:
                    edge_label = ""

                # Bayesian networks are typically drawn with the arrows
                # in the direction of data flow, not in the direction
                # of dependency.
                start_node = to_id(nodes[i]) if point_at_input else n
                end_node = n if point_at_input else to_id(nodes[i])
                db.with_edge(start_node, end_node, edge_label)
        return str(db)

    def to_bmg(self) -> Graph:
        """This transforms the accumulated graph into a BMG type system compliant
        graph and then creates the graph nodes in memory."""
        from beanmachine.ppl.compiler.fix_problems import fix_problems

        fix_problems(self).raise_errors()
        g = Graph()
        d: Dict[BMGNode, int] = {}
        for node in self._traverse_from_roots():
            d[node] = node._add_to_graph(g, d)
        return g

    def _resort_nodes(self) -> Dict[BMGNode, int]:
        """This renumbers the nodes so that the ids are in topological order;
        it returns a dictionary mapping nodes to integers."""
        sorted_nodes = {}
        for index, node in enumerate(self._traverse_from_roots()):
            sorted_nodes[node] = index
        return sorted_nodes

    def to_python(self) -> str:
        """This transforms the accumulatled graph into a BMG type system compliant
        graph and then creates a Python program which creates the graph."""
        from beanmachine.ppl.compiler.fix_problems import fix_problems

        fix_problems(self).raise_errors()

        header = """from beanmachine import graph
from torch import tensor
g = graph.Graph()
"""
        sorted_nodes = self._resort_nodes()
        return header + "\n".join(n._to_python(sorted_nodes) for n in sorted_nodes)

    def to_cpp(self) -> str:
        """This transforms the accumulatled graph into a BMG type system compliant
        graph and then creates a C++ program which creates the graph."""

        from beanmachine.ppl.compiler.fix_problems import fix_problems

        fix_problems(self).raise_errors()
        sorted_nodes = self._resort_nodes()
        return "graph::Graph g;\n" + "\n".join(
            n._to_cpp(sorted_nodes) for n in sorted_nodes
        )

    def _traverse_from_roots(self) -> List[BMGNode]:
        """This returns a list of the reachable graph nodes
        in topologically sorted order. The ordering invariants are
        (1) all sample, observation and query nodes are enumerated
        in the order they were added, and
        (2) all inputs are enumerated before their outputs, and
        (3) inputs to the "left" are enumerated before those to
        the "right"."""

        # We require here that the graph is acyclic.

        # TODO: The graph should be acyclic by construction;
        # we detect cycles while executing the lifted model.
        # However, we might want to add a quick cycle checking
        # pass here as a sanity check.

        # TODO: Do we require sample nodes to be roots? Could we
        # get by with just observations and queries?
        def is_root(n: BMGNode) -> bool:
            return (
                isinstance(n, SampleNode)
                or isinstance(n, Observation)
                or isinstance(n, Query)
            )

        def key(n: BMGNode) -> int:
            return self.nodes[n]

        # We cannot use a recursive algorithm because the graph may have
        # paths that are deeper than the recursion limit in Python.
        # Instead we'll use a list as a stack.  But we cannot simply do
        # a normal iterative depth-first or postorder traversal because
        # that violates our stated invariants above: all inputs are always
        # enumerated before the node which inputs them, and nodes to the
        # left are enumerated before nodes to the right.
        #
        # What we do here is a modified depth first traversal which maintains
        # our invariants.

        result = []
        work_stack = sorted(
            (n for n in self.nodes if is_root(n)), key=key, reverse=True
        )
        already_in_result = set()
        inputs_already_pushed = set()

        while len(work_stack) != 0:
            # Peek the top of the stack but do not pop it yet.
            current = work_stack[-1]
            if current in already_in_result:
                # The top of the stack has already been put into the
                # result list. There is nothing more to do with this node,
                # so we can simply pop it away.
                work_stack.pop()
            elif current in inputs_already_pushed:
                # The top of the stack is not on the result list, but we have
                # already pushed all of its inputs onto the stack. Since they
                # are gone from the stack, we must have already put all of them
                # onto the result list, and therefore we are justified in putting
                # this node onto the result list too.
                work_stack.pop()
                result.append(current)
                already_in_result.add(current)
            else:
                # The top of the stack is not on the result list and its inputs
                # have never been put onto the stack. Leave it on the stack so that
                # we come back to it later after all of its inputs have been
                # put on the result list, and put its inputs on the stack.
                #
                # We want to process the left inputs before the right inputs, so
                # reverse them so that the left inputs go on the stack last, and
                # are therefore closer to the top.
                for i in reversed(current.inputs):
                    work_stack.append(i)
                inputs_already_pushed.add(current)

        return result

    def all_observations(self) -> List[Observation]:
        return sorted(
            (n for n in self.nodes if isinstance(n, Observation)),
            key=lambda n: self.nodes[n],
        )

    def accumulate_graph(
        self,
        queries: List[RVIdentifier],
        observations: Dict[RVIdentifier, Any],
    ) -> None:
        # TODO: Add error handling and tests for scenarios where
        # an rv does not return a distribution or a functional does
        # not return an operation.
        for rv, val in observations.items():
            node = self._rv_to_node(rv)
            self.add_observation(node, val)
        for qrv in queries:
            node = self._rv_to_node(qrv)
            q = self.add_query(node)
            self.query_rv_map[q] = qrv

    def infer(
        self,
        num_samples: int,
    ) -> MonteCarloSamples:
        # TODO: Add num_chains to API
        # TODO: Add inference kind to API
        from beanmachine.ppl.compiler.fix_problems import fix_problems

        fix_problems(self).raise_errors()
        g = Graph()

        d: Dict[BMGNode, int] = {}
        q: Dict[int, RVIdentifier] = {}
        for node in self._traverse_from_roots():
            # When we add a query to the graph, we get back the
            # column index in the sample set for that query.
            # Make a note of what RVID that is associated with.
            d[node] = node._add_to_graph(g, d)
            if isinstance(node, Query) and node in self.query_rv_map:
                q[d[node]] = self.query_rv_map[node]

        # Suppose we have two queries and three samples; the shape we get
        # from BMG is:
        #
        # [ [s00, s01], [s10, s11], [s20, s21] ]
        #
        # But what we need is:
        #
        # {
        #   RV0: tensor([[s00, s10, s20]]),
        #   RV1: tensor([[s01, s11, s21]])
        # }
        #
        # Start by taking the transpose.

        samples = g.infer(num_samples, InferenceType.NMC)
        transp = tensor(samples).transpose(0, 1)

        # Now we've got
        #
        # [ [s00, s10, s20], [s01, s11, s21] ]
        #
        # And we can turn this into a dictionary with a comprehension:

        result: Dict[RVIdentifier, Tensor] = {
            q[index]: samples.reshape([1] + list(samples.shape))
            for index, samples in enumerate(transp)
            if index in q
        }
        return MonteCarloSamples(result)

    def infer_deprecated(
        self,
        queries: List[OperatorNode],
        observations: Dict[SampleNode, Any],
        num_samples: int,
    ) -> List[Any]:
        # TODO: Remove this method

        from beanmachine.ppl.compiler.fix_problems import fix_problems

        # TODO: Error checking; verify that all observations are samples
        # TODO: and all queries are operators.

        for node, val in observations.items():
            self.add_observation(node, val)

        for q in queries:
            self.add_query(q)

        fix_problems(self).raise_errors()
        g = Graph()

        d: Dict[BMGNode, int] = {}
        for node in self._traverse_from_roots():
            d[node] = node._add_to_graph(g, d)

        return g.infer(num_samples, InferenceType.NMC)
