# Copyright (c) Facebook, Inc. and its affiliates.
# from beanmachine.graph import Graph
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
import collections.abc
import functools
import itertools
import math
import operator
import sys
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterator, List

# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this mModuleNotFoundError
# pyre-ignore-all-errors
from beanmachine.graph import AtomicType, DistributionType as dt, Graph, OperatorType
from beanmachine.ppl.compiler.bmg_types import Natural, PositiveReal, Probability
from beanmachine.ppl.utils.dotbuilder import DotBuilder
from beanmachine.ppl.utils.memoize import memoize
from torch import Tensor, tensor
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
from torch.distributions.utils import broadcast_all


def prod(x):
    """Compute the product of a sequence of values of arbitrary length"""
    return functools.reduce(operator.mul, x, 1)


builtin_function_or_method = type(abs)

#####
##### The following classes define the various graph node types.
#####

# TODO: This section is over two thousand lines of code, none of which
# makes use of the actual graph builder; the node types could be
# moved into a module of their own.


class BMGNode(ABC):
    """The base class for all graph nodes."""

    # A node has a list of its child nodes,
    # each of which has an associated edge label.
    #
    # TODO: "children" is misleading, as the graph is not a tree
    # and the edges are not logically representing a parent-child
    # relationship. Rather, each "child" would better be thought
    # of as an "input" or a "dependency". A distribution is not
    # the *child* of a sample; rather, the sample depends on the
    # distribution to determine its semantics. Similarly, addends
    # are not children of a sum; they are the inputs to the sum.
    # Consider refactoring "children" throughout to "inputs" or
    # "dependencies".

    children: List["BMGNode"]
    edges: List[str]

    def __init__(self, children: List["BMGNode"]):
        self.children = children

    @property
    @abstractmethod
    def node_type(self) -> Any:
        """The type information associated with this node.
Every node has an associated type; before the type fixing phase
the type is the data type as it would be in the original
Python model. After type fixing it is the type in the BMG type
system."""
        pass

    @property
    @abstractmethod
    def size(self) -> torch.Size:
        """The tensor size associated with this node.
If the node represents a scalar value then produce Size([])."""
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """This gives the label of the node When generating a DOT file
for graph visualization and debugging."""
        pass

    @abstractmethod
    def _add_to_graph(self, g: Graph, d: Dict["BMGNode", int]) -> int:
        """This adds a node to an in-memory BMG instance. Each node
in BMG is associated with an integer handle; this returns
the handle for this node assigned by BMG."""
        pass

    @abstractmethod
    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        """We can emit the graph as a Python program which, when executed,
builds a BMG instance. This method returns a string of Python
code to construct this node. The dictionary associates a unique
integer with each node that can be used to construct an identifier."""
        pass

    @abstractmethod
    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        """We can emit the graph as a C++ program which, when executed,
builds a BMG instance. This method returns a string of C++
code to construct this node. The dictionary associates a unique
integer with each node that can be used to construct an identifier."""
        pass

    @abstractmethod
    def support(self) -> Iterator[Any]:
        """To build the graph of all possible control flows through
the model we need to know for any given node what are
all the possible values it could attain; we require that
the set be finite and will throw an exception if it is not."""
        pass


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


# When constructing the support of various nodes we are often
# having to remove duplicates from a set of possible values.
# Unfortunately, it is not easy to do so with torch tensors.
# This helper class implements a set of tensors.

# TODO: Move this to its own module.


class SetOfTensors(collections.abc.Set):
    """Tensors cannot be put into a normal set because tensors that compare as
     equal do not hash to equal hashes. This is a linear-time set implementation.
     Most of the time the sets will be very small. """

    elements: List[Tensor]

    def __init__(self, iterable):
        self.elements = []
        for value in iterable:
            t = value if isinstance(value, Tensor) else tensor(value)
            if t not in self.elements:
                self.elements.append(t)

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, value):
        return value in self.elements

    def __len__(self):
        return len(self.elements)


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


# TODO: There is a bunch of replicated code in the various
# _value_to_python code below; consider refactoring to match
# the C++ technique here of just having a helper method like
# the one above that deals with displaying constants as Python.


def _value_to_cpp(value: Any) -> str:

    """Generate the program text of an expression
representing a constant value in the C++ output."""

    if isinstance(value, Tensor):
        # TODO: What if the tensor is not made of floats?
        values = ",".join(str(element) for element in value.storage())
        dims = ",".join(str(dim) for dim in value.shape)
        return f"torch::from_blob((float[]){{{values}}}, {{{dims}}})"
    return str(value).lower()


class ConstantNode(BMGNode, metaclass=ABCMeta):
    """This is the base type for all nodes representing constants.
Note that every constant node has an associated type in the
BMG type system; nodes that represent the "real" 1.0,
the "positive real" 1.0, the "probability" 1.0 and the
"natural" 1 are all different nodes and are NOT deduplicated."""

    edges = []
    value: Any

    def __init__(self):
        BMGNode.__init__(self, [])

    @abstractmethod
    def _value_to_python(self) -> str:
        pass

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = _value_to_cpp(self.value)
        return f"uint n{n} = g.add_constant({v});"

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = self._value_to_python()
        return f"n{n} = g.add_constant({v})"

    # The support of a constant is just the value.
    def support(self) -> Iterator[Any]:
        yield self.value


class BooleanNode(ConstantNode):
    """A Boolean constant"""

    value: bool

    def __init__(self, value: bool):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def node_type(self) -> Any:
        return bool

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    @property
    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(bool(self.value))

    def _value_to_python(self) -> str:
        return str(bool(self.value))

    def __bool__(self) -> bool:
        return self.value


class RealNode(ConstantNode):
    """An unrestricted real constant"""

    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def node_type(self) -> Any:
        return float

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    @property
    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(float(self.value))

    def _value_to_python(self) -> str:
        return str(float(self.value))

    def __bool__(self) -> bool:
        return bool(self.value)


# TODO: This code is mis-placed, as it is right in the middle
# of all the constants classes.  Move it to be near IndexNode
# and IfThenElse node, as it is closely related to both.


class MapNode(BMGNode):

    """This class represents a point in a program where there are
multiple control flows based on the value of a stochastic node."""

    # For example, suppose we have this contrived model:
    #
    #   @sample def weird(i):
    #     if i == 0:
    #       return Normal(0.0, 1.0)
    #     return Normal(1.0, 1.0)
    #
    #   @sample def flips():
    #     return Binomial(2, 0.5)
    #
    #   @sample def really_weird():
    #     return Normal(weird(flips()), 1.0)
    #
    # There are three possibilities for weird(flips()) on the last line;
    # what we need to represent in the graph is:
    #
    # * sample once from Normal(0.0, 1.0), call this weird(0)
    # * sample twice from Normal(1.0, 1.0), call these weird(1) and weird(2)
    # * sample once from flips()
    # * choose one of weird(i) based on the sample from flips().
    #
    # We represent this with two nodes: a map and an index:
    #
    # sample --- Normal(_, 1.0)        ---0--- sample --- Normal(0.0, 1.0)
    #                    \           /
    #                   index ---> map----1--- sample --- Normal(1.0, 1.0)
    #                        \       \                   /
    #                         \        ---2--- sample --
    #                          \
    #                            --- sample -- Binomial(2, 0.5)
    #
    # As an implementation detail, we represent the key-value pairs in the
    # map by a convention:
    #
    # * Even numbered children are keys
    # * All keys are constant nodes
    # * Odd numbered children are values associated with the previous
    #   sibling as the key.

    # TODO: We do not yet have this choice node in BMG, and the design
    # is not yet settled.
    #
    # The accumulator creates the map based on the actual values seen
    # as indices in the execution of the model; in the contrived example
    # above the indices are 0, 1, 2 but there is no reason why they could
    # not have been 1, 10, 100 instead; all that matters is that there
    # were three of them and so three code paths were explored.
    #
    # That's why this is implemented as (and named) "map"; it is an arbitrary
    # collection of key-value pairs where the keys are drawn from the support
    # of a distribution.
    #
    # The design decision to be made here is whether we need an arbitrary map
    # node in BMG, as we accumulate here, or if we need the more restricted
    # case of an "array" or "list", where we are mapping 0, 1, 2, ... n-1
    # to n graph nodes, and the indexed sample is drawn from a distribution
    # whose support is exactly 0 to n-1.

    def __init__(self, children: List[BMGNode]):
        # TODO: Check that keys are all constant nodes.
        # TODO: Check that there is one value for each key.
        # TODO: Verify that there is at least one pair.
        BMGNode.__init__(self, children)
        self.edges = [str(x) for x in range(len(children))]

    @property
    def node_type(self) -> Any:
        return Dict

    @property
    def size(self) -> torch.Size:
        return self.children[1].size

    @property
    def label(self) -> str:
        return "map"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: map nodes are not currently part of the graph
        return -1

    def _to_python(self, d: Dict[BMGNode, int]) -> str:
        # TODO: map nodes are not currently part of the graph
        return ""

    def _to_cpp(self, d: Dict[BMGNode, int]) -> str:
        # TODO: map nodes are not currently part of the graph
        return ""

    def support(self) -> Iterator[Any]:
        return []

    # TODO: The original plan was to represent Python values such as
    # lists, tuples and dictionaries as one of these map nodes,
    # and it was convenient during prototyping that a map node
    # have an indexer that behaved like the Python indexer it was
    # emulating. This idea has been abandoned, so this code can
    # be deleted in a cleanup pass.

    def __getitem__(self, key) -> BMGNode:
        if isinstance(key, BMGNode) and not isinstance(key, ConstantNode):
            raise ValueError("BeanMachine map must be indexed with a constant value")
        # Linear search is fine.  We're not going to do this a lot, and the
        # maps will be small.
        k = key.value if isinstance(key, ConstantNode) else key
        for i in range(len(self.children) // 2):
            if self.children[i * 2].value == k:
                return self.children[i * 2 + 1]
        raise ValueError("Key not found in map")


class ProbabilityNode(ConstantNode):
    """A real constant restricted to values from 0.0 to 1.0"""

    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def node_type(self) -> Any:
        return Probability

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    @property
    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant_probability(float(self.value))

    def _value_to_python(self) -> str:
        return str(float(self.value))

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = _value_to_cpp(self.value)
        return f"uint n{n} = g.add_constant_probability({v});"

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = self._value_to_python()
        return f"n{n} = g.add_constant_probability({v})"


class PositiveRealNode(ConstantNode):
    """A real constant restricted to non-negative values"""

    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def node_type(self) -> Any:
        return PositiveReal

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    @property
    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant_pos_real(float(self.value))

    def _value_to_python(self) -> str:
        return str(float(self.value))

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = _value_to_cpp(self.value)
        return f"uint n{n} = g.add_constant_pos_real({v});"

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = self._value_to_python()
        return f"n{n} = g.add_constant_pos_real({v})"


class NaturalNode(ConstantNode):
    """An integer constant restricted to non-negative values"""

    value: int

    def __init__(self, value: int):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def node_type(self) -> Any:
        return Natural

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    @property
    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(self.value)

    def _value_to_python(self) -> str:
        return str(self.value)

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = _value_to_cpp(self.value)
        return f"uint n{n} = g.add_constant({v});"

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = self._value_to_python()
        return f"n{n} = g.add_constant({v})"


class TensorNode(ConstantNode):
    """A tensor constant"""

    value: Tensor

    def __init__(self, value: Tensor):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def _tensor_to_python(t: Tensor) -> str:
        if len(t.shape) == 0:
            return str(t.item())
        return "[" + ",".join(TensorNode._tensor_to_python(c) for c in t) + "]"

    @staticmethod
    def _tensor_to_label(t: Tensor) -> str:
        length = len(t.shape)
        if length == 0 or length == 1:
            return TensorNode._tensor_to_python(t)
        return "[" + ",\\n".join(TensorNode._tensor_to_label(c) for c in t) + "]"

    # TODO: Do tensor types need to describe their contents?
    @property
    def node_type(self) -> Any:
        return Tensor

    @property
    def size(self) -> torch.Size:
        return self.value.size()

    @property
    def label(self) -> str:
        return TensorNode._tensor_to_label(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(self.value)

    def _value_to_python(self) -> str:
        t = TensorNode._tensor_to_python(self.value)
        return f"tensor({t})"

    def __bool__(self) -> bool:
        return bool(self.value)


class DistributionNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all nodes that represent
probability distributions."""

    types_fixed: bool

    def __init__(self, children: List[BMGNode]):
        self.types_fixed = False
        BMGNode.__init__(self, children)

    # Distribution nodes do not have a type themselves.
    # (In BMG they have type "unknown", but "no type" would be
    # a more accurate way to characterize it.)
    # However, we do know the type that will be produced when
    # sampling this distribution, and we need that information to
    # correctly assign a type to SampleNodes.
    #
    # The node_type property of a distribution will return the
    # Python type that was used to construct the distribution in
    # the original model.

    @abstractmethod
    def sample_type(self) -> Any:
        pass


class BernoulliNode(DistributionNode):
    """The Bernoulli distribution is a coin flip; it takes
a probability and each sample is either 0.0 or 1.0.

The probability can be expressed either as a normal
probability between 0.0 and 1.0, or as log-odds, which
is any real number. That is, to represent, say,
13 heads for every 17 tails, the logits would be log(13/17).

If the model gave the probability as a value when executing
the program then torch will automatically translate
logits to normal probabilities. If however the model gives
a stochastic node as the argument and uses logits, then
we generate a different node in BMG."""

    edges = ["probability"]
    is_logits: bool

    def __init__(self, probability: BMGNode, is_logits: bool = False):
        self.is_logits = is_logits
        DistributionNode.__init__(self, [probability])

    @property
    def probability(self) -> BMGNode:
        return self.children[0]

    @probability.setter
    def probability(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def node_type(self) -> Any:
        return Bernoulli

    @property
    def size(self) -> torch.Size:
        if self.types_fixed:
            return torch.Size([])
        return self.probability.size

    def sample_type(self) -> Any:
        if self.types_fixed:
            return bool
        return self.probability.node_type

    @property
    def label(self) -> str:
        return "Bernoulli" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability) + ")"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        dist_type = dt.BERNOULLI_LOGIT if self.is_logits else dt.BERNOULLI
        return g.add_distribution(dist_type, AtomicType.BOOLEAN, [d[self.probability]])

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        logit = "_LOGIT" if self.is_logits else ""
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + f"  graph.DistributionType.BERNOULLI{logit},\n"
            + "  graph.AtomicType.BOOLEAN,\n"
            + f"  [n{d[self.probability]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        logit = "_LOGIT" if self.is_logits else ""
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + f"  graph::DistributionType::BERNOULLI{logit},\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.probability]}}}));"
        )

    def support(self) -> Iterator[Any]:
        if self.types_fixed:
            return [False, True]
        s = self.size
        return (tensor(i).view(s) for i in itertools.product(*([[0.0, 1.0]] * prod(s))))


class BinomialNode(DistributionNode):
    """The Binomial distribution is the extension of the
Bernoulli distribution to multiple flips. The input
is the count of flips and the probability of each
coming up heads; each sample is the number of heads
after "count" flips.

The probability can be expressed either as a normal
probability between 0.0 and 1.0, or as log-odds, which
is any real number. That is, to represent, say,
13 heads for every 17 tails, the logits would be log(13/17).

If the model gave the probability as a value when executing
the program then torch will automatically translate
logits to normal probabilities. If however the model gives
a stochastic node as the argument and uses logits, then
we generate a different node in BMG."""

    # TODO: We do not yet have a BMG node for Binomial
    # with logits. When we do, add support for it as we
    # did with Bernoulli above.

    edges = ["count", "probability"]
    is_logits: bool

    def __init__(self, count: BMGNode, probability: BMGNode, is_logits: bool = False):
        self.is_logits = is_logits
        DistributionNode.__init__(self, [count, probability])

    @property
    def count(self) -> BMGNode:
        return self.children[0]

    @count.setter
    def count(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def probability(self) -> BMGNode:
        return self.children[1]

    @probability.setter
    def probability(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def node_type(self) -> Any:
        return Binomial

    @property
    def size(self) -> torch.Size:
        if self.types_fixed:
            return torch.Size([])
        return broadcast_all(
            torch.zeros(self.count.size), torch.zeros(self.probability.size)
        ).size()

    def sample_type(self) -> Any:
        if self.types_fixed:
            return Natural
        return Tensor

    @property
    def label(self) -> str:
        return "Binomial" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return f"Binomial({self.count}, {self.probability})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: Fix this when we support binomial logits.
        return g.add_distribution(
            dt.BINOMIAL, AtomicType.NATURAL, [d[self.count], d[self.probability]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.BINOMIAL,\n"
            + "  graph.AtomicType.NATURAL,\n"
            + f"  [n{d[self.count]}, n{d[self.probability]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BINOMIAL,\n"
            + "  graph::AtomicType::NATURAL,\n"
            + f"  std::vector<uint>({{n{d[self.count]}, "
            + f"n{d[self.probability]}}}));"
        )

    # TODO: We will need to implement computation of the support
    # of an arbitrary binomial distribution because samples are
    # discrete values between 0 and count, which is typically small.
    # Though implementing support computation if count is non-stochastic
    # is straightforward, we do not yet have the gear to implement
    # this for stochastic counts. Consider this contrived case:
    #
    # @sample def a(): return Binomial(2, 0.5)
    # @sample def b(): return Binomial(a() + 1, 0.4)
    # @sample def c(i): return Normal(0.0, 2.0)
    # @sample def d(): return Normal(c(b()), 3.0)
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
    def support(self) -> Iterator[Any]:
        raise ValueError("Support of binomial is not yet implemented.")


class CategoricalNode(DistributionNode):
    """The categorical distribution is the extension of the
Bernoulli distribution to multiple outcomes; rather
than flipping an unfair coin, this is rolling an unfair
n-sided die.

The input is the probability of each of n possible outcomes,
and each sample is drawn from 0, 1, 2, ... n-1.

The probability can be expressed either as a normal
probability between 0.0 and 1.0, or as log-odds, which
is any real number. That is, to represent, say,
13 heads for every 17 tails, the logits would be log(13/17).

If the model gave the probability as a value when executing
the program then torch will automatically translate
logits to normal probabilities. If however the model gives
a stochastic node as the argument and uses logits, then
we generate a different node in BMG."""

    # Note that a vector of n probabilities that adds to 1.0 is
    # called a simplex.
    #
    # TODO: we may wish to add simplexes to the
    # BMG type system at some point, and also bounded integers.
    #
    # TODO: We do not yet have a BMG node for categorical
    # distributions; when we do, finish this implementation.

    edges = ["probability"]
    is_logits: bool

    def __init__(self, probability: BMGNode, is_logits: bool = False):
        self.is_logits = is_logits
        DistributionNode.__init__(self, [probability])

    @property
    def probability(self) -> BMGNode:
        return self.children[0]

    @probability.setter
    def probability(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def node_type(self) -> Any:
        return Categorical

    @property
    def size(self) -> torch.Size:
        return self.probability.size[0:-1]

    def sample_type(self) -> Any:
        # TODO: When we support bounded integer types
        # TODO: this should indicate that it is a tensor
        # TODO: of bound integers.
        return self.probability.node_type

    @property
    def label(self) -> str:
        return "Categorical" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return "Categorical(" + str(self.probability) + ")"

    # TODO: Delete this cut-n-pasted incorrect code and just
    # raise errors instead to indicate that this feature is not
    # yet implemented

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: Handle case where child is logits
        # TODO: This is incorrect.
        return g.add_distribution(
            dt.BERNOULLI, AtomicType.BOOLEAN, [d[self.probability]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        # TODO: This is incorrect.
        return (
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.probability]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        # TODO: This is incorrect.
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.probability]}}}));"
        )

    def support(self) -> Iterator[Any]:
        s = self.probability.size
        r = list(range(s[-1]))
        sr = s[:-1]
        return (tensor(i).view(sr) for i in itertools.product(*([r] * prod(sr))))


class DirichletNode(DistributionNode):
    """The Dirichlet distribution generates simplexs -- vectors
whose members are probabilities that add to 1.0, and
so it is useful for generating inputs to the categorical
distribution."""

    # TODO: We do not yet have a BMG node for Dirichlet
    # distributions; when we do, finish this implementation.
    edges = ["concentration"]

    def __init__(self, concentration: BMGNode):
        DistributionNode.__init__(self, [concentration])

    @property
    def concentration(self) -> BMGNode:
        return self.children[0]

    @concentration.setter
    def concentration(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def node_type(self) -> Any:
        return Dirichlet

    @property
    def size(self) -> torch.Size:
        return self.concentration.size

    def sample_type(self) -> Any:
        return self.concentration.node_type

    @property
    def label(self) -> str:
        return "Dirichlet"

    def __str__(self) -> str:
        return f"Dirichlet({str(self.concentration)})"

    # TODO: Delete this cut-n-pasted incorrect code and just
    # raise errors instead to indicate that this feature is not
    # yet implemented

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            # TODO: Fix this when we add the node type to BMG
            dt.BERNOULLI,
            AtomicType.BOOLEAN,
            [d[self.concentration]],
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.concentration]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.concentration]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a Dirichlet.
        raise ValueError(f"Dirichlet distribution does not have finite support.")


class HalfCauchyNode(DistributionNode):
    """The Cauchy distribution is a bell curve with zero mean
and a heavier tail than the normal distribution; it is useful
for generating samples that are not as clustered around
the mean as a normal.

The half Cauchy distribution is just the distribution you
get when you take the absolute value of the samples from
a Cauchy distribution. The input is a positive scale factor
and a sample is a positive real number."""

    # TODO: Add support for the Cauchy distribution as well.

    edges = ["scale"]

    def __init__(self, scale: BMGNode):
        DistributionNode.__init__(self, [scale])

    @property
    def scale(self) -> BMGNode:
        return self.children[0]

    @scale.setter
    def scale(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def node_type(self) -> Any:
        return HalfCauchy

    @property
    def size(self) -> torch.Size:
        return self.scale.size

    def sample_type(self) -> Any:
        return self.scale.node_type

    @property
    def label(self) -> str:
        return "HalfCauchy"

    def __str__(self) -> str:
        return f"HalfCauchy({str(self.scale)})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(dt.HALF_CAUCHY, AtomicType.POS_REAL, [d[self.scale]])

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.HALF_CAUCHY,\n"
            + "  graph.AtomicType.POS_REAL,\n"
            + f"  [n{d[self.scale]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::HALF_CAUCHY,\n"
            + "  graph::AtomicType::POS_REAL,\n"
            + f"  std::vector<uint>({{n{d[self.scale]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a half Cauchy.
        raise ValueError(f"HalfCauchy distribution does not have finite support.")


class NormalNode(DistributionNode):

    """The normal (or "Gaussian") distribution is a bell curve with
a given mean and standard deviation."""

    edges = ["mu", "sigma"]

    def __init__(self, mu: BMGNode, sigma: BMGNode):
        DistributionNode.__init__(self, [mu, sigma])

    @property
    def mu(self) -> BMGNode:
        return self.children[0]

    @mu.setter
    def mu(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def sigma(self) -> BMGNode:
        return self.children[1]

    @sigma.setter
    def sigma(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def node_type(self) -> Any:
        return Normal

    @property
    def size(self) -> torch.Size:
        return self.mu.size

    def sample_type(self) -> Any:
        return self.mu.node_type

    @property
    def label(self) -> str:
        return "Normal"

    def __str__(self) -> str:
        return f"Normal({str(self.mu)},{str(self.sigma)})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            dt.NORMAL, AtomicType.REAL, [d[self.mu], d[self.sigma]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.NORMAL,\n"
            + "  graph.AtomicType.REAL,\n"
            + f"  [n{d[self.mu]}, n{d[self.sigma]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::NORMAL,\n"
            + "  graph::AtomicType::REAL,\n"
            + f"  std::vector<uint>({{n{d[self.mu]}, n{d[self.sigma]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a normal.
        raise ValueError(f"Normal distribution does not have finite support.")


class StudentTNode(DistributionNode):
    """The Student T distribution is a bell curve with zero mean
and a heavier tail than the normal distribution. It is
useful in statistical analysis because a common situation
is to have observations of a normal process but to not
know the true mean. Samples from the T distribution can
be used to represent the difference between an observed mean
and the true mean."""

    edges = ["df", "loc", "scale"]

    def __init__(self, df: BMGNode, loc: BMGNode, scale: BMGNode):
        DistributionNode.__init__(self, [df, loc, scale])

    @property
    def df(self) -> BMGNode:
        return self.children[0]

    @df.setter
    def df(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def loc(self) -> BMGNode:
        return self.children[1]

    @loc.setter
    def loc(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def scale(self) -> BMGNode:
        return self.children[2]

    @scale.setter
    def scale(self, p: BMGNode) -> None:
        self.children[2] = p

    @property
    def node_type(self) -> Any:
        return StudentT

    def sample_type(self) -> Any:
        if self.fixed_types:
            return float
        return self.df.node_type

    @property
    def size(self) -> torch.Size:
        return self.df.size

    @property
    def label(self) -> str:
        return "StudentT"

    def __str__(self) -> str:
        return f"StudentT({str(self.df)},{str(self.loc)},{str(self.scale)})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            dt.STUDENT_T, AtomicType.REAL, [d[self.df], d[self.loc], d[self.scale]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.STUDENT_T,\n"
            + "  graph.AtomicType.REAL,\n"
            + f"  [n{d[self.df]}, n{d[self.loc]}, n{d[self.scale]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::STUDENT_T,\n"
            + "  graph::AtomicType::REAL,\n"
            + "  std::vector<uint>("
            + f"{{n{d[self.df]}, n{d[self.loc]}, n{d[self.scale]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a student t.
        raise ValueError(f"StudentT distribution does not have finite support.")


class UniformNode(DistributionNode):

    """The Uniform distribution is a "flat" distribution of values
between 0.0 and 1.0."""

    # TODO: We do not yet have an implementation of the uniform
    # distribution as a BMG node. When we do, implement the
    # feature here.

    edges = ["low", "high"]

    def __init__(self, low: BMGNode, high: BMGNode):
        DistributionNode.__init__(self, [low, high])

    @property
    def low(self) -> BMGNode:
        return self.children[0]

    @low.setter
    def low(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def high(self) -> BMGNode:
        return self.children[1]

    @high.setter
    def high(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def node_type(self) -> Any:
        return Uniform

    def sample_type(self) -> Any:
        return self.low.node_type

    @property
    def size(self) -> torch.Size:
        return self.low.size

    @property
    def label(self) -> str:
        return "Uniform"

    def __str__(self) -> str:
        return f"Uniform({str(self.low)},{str(self.high)})"

    # TODO: Delete this cut-n-pasted incorrect code and just
    # raise errors instead to indicate that this feature is not
    # yet implemented

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            # TODO: Fix this when we add the node type to BMG
            dt.BERNOULLI,
            AtomicType.BOOLEAN,
            [d[self.low]],
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.low]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.low]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a uniform.
        raise ValueError(f"Uniform distribution does not have finite support.")


# TODO: The rest of the distribution nodes are in alphabetical order;
# consider moving this code up higher in the module.


class BetaNode(DistributionNode):
    """The beta distribution samples are values between 0.0 and 1.0, and
so is useful for creating probabilities."""

    edges = ["alpha", "beta"]

    def __init__(self, alpha: BMGNode, beta: BMGNode):
        DistributionNode.__init__(self, [alpha, beta])

    @property
    def alpha(self) -> BMGNode:
        return self.children[0]

    @alpha.setter
    def alpha(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def beta(self) -> BMGNode:
        return self.children[1]

    @beta.setter
    def beta(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def node_type(self) -> Any:
        return Beta

    def sample_type(self) -> Any:
        if self.fixed_types:
            return Probability
        return self.alpha.node_type

    @property
    def size(self) -> torch.Size:
        return self.alpha.size

    @property
    def label(self) -> str:
        return "Beta"

    def __str__(self) -> str:
        return f"Beta({str(self.alpha)},{str(self.beta)})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            dt.BETA, AtomicType.PROBABILITY, [d[self.alpha], d[self.beta]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.BETA,\n"
            + "  graph.AtomicType.PROBABILITY,\n"
            + f"  [n{d[self.alpha]}, n{d[self.beta]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BETA,\n"
            + "  graph::AtomicType::PROBABILITY,\n"
            + f"  std::vector<uint>({{n{d[self.alpha]}, n{d[self.beta]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a beta.
        raise ValueError(f"Beta distribution does not have finite support.")


class OperatorNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all operators.
The children are the operands of each operator."""

    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)


class IfThenElseNode(OperatorNode):
    """This class represents a stochastic choice between two options, where
the condition is a Boolean."""

    # This node will only be generated when tranforming the Python version of
    # the graph into the BMG format; for instance, if we have a multiplication
    # of a Bernoulli sample node by 2.0, in the Python form we'll have a scalar
    # multiplied by a sample of type tensor. In the BMG form the sample will be
    # of type Boolean and we cannot multiply a Boolean by a Real. Instead we'll
    # generate "if_then_else(sample, 0.0, 1.0) * 2.0" which typechecks in the
    # BMG type system.
    #
    # Eventually we will probably use this node to represent Python's
    # "consequence if condition else alternative" syntax, and possibly
    # other conditional stochastic control flows.
    edges = ["condition", "consequence", "alternative"]

    def __init__(self, condition: BMGNode, consequence: BMGNode, alternative: BMGNode):
        OperatorNode.__init__(self, [condition, consequence, alternative])

    @property
    def node_type(self) -> Any:
        return self.consequence.node_type

    @property
    def condition(self) -> BMGNode:
        return self.children[0]

    @condition.setter
    def condition(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def consequence(self) -> BMGNode:
        return self.children[1]

    @consequence.setter
    def consequence(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def alternative(self) -> BMGNode:
        return self.children[2]

    @alternative.setter
    def alternative(self, p: BMGNode) -> None:
        self.children[2] = p

    @property
    def label(self) -> str:
        return "if"

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        i = str(self.condition)
        t = str(self.consequence)
        e = str(self.alternative)
        return f"(if {i} then {t} else {e})"

    def support(self) -> Iterator[Any]:
        raise ValueError("support of IfThenElseNode not yet implemented")

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_operator(
            OperatorType.IF_THEN_ELSE,
            [d[self.condition], d[self.consequence], d[self.alternative]],
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        i = d[self.condition]
        t = d[self.consequence]
        e = d[self.alternative]
        return (
            f"n{n} = g.add_operator(\n"
            + "  graph.OperatorType.IF_THEN_ELSE,\n"
            + f"  [n{i}, n{t}, n{e}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        i = d[self.condition]
        t = d[self.consequence]
        e = d[self.alternative]
        return (
            f"n{n} = g.add_operator(\n"
            + "  graph::OperatorType::IF_THEN_ELSE,\n"
            + f"  std::vector<uint>({{n{i}, n{t}, n{e}}}));"
        )


class BinaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    """This is the base class for all binary operators."""

    edges = ["left", "right"]
    operator_type: OperatorType

    def __init__(self, left: BMGNode, right: BMGNode):
        OperatorNode.__init__(self, [left, right])

    # TODO: We do not correctly compute the type of a node in the
    # graph accumulated from initially executing the Python program,
    # and neither do we yet correctly impose the BMG type system
    # restrictions on binary operators -- namely that both input
    # types must be the same, and the output type is also that type.

    @property
    def node_type(self) -> Any:
        if self.left.node_type == Tensor or self.right.node_type == Tensor:
            return Tensor
        return float

    @property
    def left(self) -> BMGNode:
        return self.children[0]

    @left.setter
    def left(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def right(self) -> BMGNode:
        return self.children[1]

    @right.setter
    def right(self, p: BMGNode) -> None:
        self.children[1] = p

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_operator(self.operator_type, [d[self.left], d[self.right]])

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        ot = self.operator_type
        left = d[self.left]
        right = d[self.right]
        return f"n{n} = g.add_operator(graph.{ot}, [n{left}, n{right}])"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        ot = str(self.operator_type).replace(".", "::")
        left = d[self.left]
        right = d[self.right]
        return (
            f"uint n{d[self]} = g.add_operator(\n"
            + f"  graph::{ot}, std::vector<uint>({{n{left}, n{right}}}));"
        )


class AdditionNode(BinaryOperatorNode):
    """This represents an addition of values."""

    # TODO: We accumulate nodes as strictly binary operations, but BMG supports
    # n-ary addition; we might consider a graph transformation that turns
    # chained additions into a single addition node.
    operator_type = OperatorType.ADD

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) + torch.zeros(self.right.size)).size()

    @property
    def label(self) -> str:
        return "+"

    def __str__(self) -> str:
        return "(" + str(self.left) + "+" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l + r for l in self.left.support() for r in self.right.support()
        )


class MultiplicationNode(BinaryOperatorNode):
    """This represents a multiplication of nodes."""

    # TODO: We accumulate nodes as strictly binary operations, but BMG supports
    # n-ary multiplication; we might consider a graph transformation that turns
    # chained multiplications into a single multiplication node.
    operator_type = OperatorType.MULTIPLY

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "*"

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) * torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "*" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l * r for l in self.left.support() for r in self.right.support()
        )


class MatrixMultiplicationNode(BinaryOperatorNode):

    """"This represents a matrix multiplication."""

    # TODO: We do not yet have an implementation of matrix
    # multiplication as a BMG node. When we do, implement the
    # feature here.
    operator_type = OperatorType.MULTIPLY

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "*"

    @property
    def size(self) -> torch.Size:
        return torch.zeros(self.left.size).mm(torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "*" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            torch.mm(l, r) for l in self.left.support() for r in self.right.support()
        )


class DivisionNode(BinaryOperatorNode):
    """This represents a division."""

    # TODO: There is no division node in BMG, and it is not clear how
    # we will represent it; obviously divisions by constants can be
    # turned into multiplications easily enough, but division by a
    # stochastic node has no representation.
    # We could add a division node, or we could add a power node
    # and generate c/d as Multiply(c, Power(d, -1.0))
    # Either way, when we decide, implement the transformation to
    # BMG nodes accordingly.
    operator_type = OperatorType.MULTIPLY

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "/"

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) / torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "/" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Filter out division by zero?
        return SetOfTensors(
            l / r for l in self.left.support() for r in self.right.support()
        )


class IndexNode(BinaryOperatorNode):
    """This represents a stochastic choice of multiple options; the left
operand must be a map, and the right is the stochastic value used to
choose an element from the map."""

    # See notes on MapNode for an explanation of this code.
    # TODO: Move this closer to the map code.
    operator_type = OperatorType.MULTIPLY  # TODO

    def __init__(self, left: MapNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "index"

    @property
    def size(self) -> torch.Size:
        return self.left.size

    def __str__(self) -> str:
        return str(self.left) + "[" + str(self.right) + "]"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l for r in self.right.support() for l in self.left[r].support()
        )


class PowerNode(BinaryOperatorNode):
    """This represents an x-to-the-y operation."""

    # TODO: We haven't added power to the C++ implementation of BMG yet.
    # TODO: When we do, update this.
    operator_type = OperatorType.MULTIPLY  # TODO

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "**"

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) ** torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "**" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l ** r for l in self.left.support() for r in self.right.support()
        )


class UnaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    """This is the base type of unary operator nodes."""

    edges = ["operand"]
    operator_type: OperatorType

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

    @property
    def node_type(self) -> Any:
        # In BMG, the output type of all unary operators
        # is the same as the input;
        # TODO: Is that the case when the graph represents
        # the Python semantics? Is there work to do here?
        return self.operand.node_type

    @property
    def operand(self) -> BMGNode:
        return self.children[0]

    @operand.setter
    def operand(self, p: BMGNode) -> None:
        self.children[0] = p

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_operator(self.operator_type, [d[self.operand]])

    def _to_python(self, d: Dict[BMGNode, int]) -> str:
        n = d[self]
        o = d[self.operand]
        ot = str(self.operator_type)
        return f"n{n} = g.add_operator(graph.{ot}, [n{o}])"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        o = d[self.operand]
        # Since OperatorType is not actually an enum, there is no
        # name attribute to use.
        ot = str(self.operator_type).replace(".", "::")
        return (
            f"uint n{n} = g.add_operator(\n"
            + f"  graph::{ot}, std::vector<uint>({{n{o}}}));"
        )


class NegateNode(UnaryOperatorNode):

    """This represents a unary minus."""

    # TODO: Add notes about the semantics of the various BMG
    # negation nodes.

    operator_type = OperatorType.NEGATE

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def label(self) -> str:
        return "-"

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "-" + str(self.operand)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(-o for o in self.operand.support())


class NotNode(UnaryOperatorNode):
    """This represents a logical not."""

    # TODO: Add notes about the semantics of the various BMG
    # negation nodes.

    # TODO: We do not support NOT in BMG yet; when we do, update this.
    operator_type = OperatorType.NEGATE  # TODO

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def node_type(self) -> Any:
        return bool

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    @property
    def label(self) -> str:
        return "not"

    def __str__(self) -> str:
        return "not " + str(self.operand)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(not o for o in self.operand.support())


# TODO: Describe the situations in which we generate this.


class ToRealNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_REAL

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def node_type(self) -> Any:
        return float

    @property
    def label(self) -> str:
        return "ToReal"

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "ToReal(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(float(o) for o in self.operand.support())


# TODO: Describe the situations in which we generate this.


class ToTensorNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_TENSOR

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def label(self) -> str:
        return "ToTensor"

    def __str__(self) -> str:
        return "ToTensor(" + str(self.operand) + ")"

    @property
    def size(self) -> torch.Size:
        # TODO: Is this correct?
        return torch.Size([1])

    @property
    def node_type(self) -> Any:
        return Tensor

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.tensor(o) for o in self.operand.support())


class ExpNode(UnaryOperatorNode):
    """This represents an exponentiation operation; it is generated when
a model contains calls to Tensor.exp or math.exp."""

    operator_type = OperatorType.EXP

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def label(self) -> str:
        return "Exp"

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Exp(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Not always a tensor.
        return SetOfTensors(torch.exp(o) for o in self.operand.support())


class LogNode(UnaryOperatorNode):

    """This represents a log operation; it is generated when
a model contains calls to Tensor.log or math.log."""

    # TODO: We do not support LOG in BMG yet; when we do, update this:
    operator_type = OperatorType.EXP  # TODO

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def label(self) -> str:
        return "Log"

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Log(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Not always a tensor.
        return SetOfTensors(torch.log(o) for o in self.operand.support())


class SampleNode(UnaryOperatorNode):
    """This represents a single unique sample from a distribution;
if a graph has two sample nodes both taking input from the same
distribution, each sample is logically distinct. But if a graph
has two nodes that both input from the same sample node, we must
treat those two uses of the sample as though they had identical
values."""

    operator_type = OperatorType.SAMPLE

    def __init__(self, operand: DistributionNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def node_type(self) -> Any:
        return self.operand.sample_type()

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    @property
    def operand(self) -> DistributionNode:
        c = self.children[0]
        assert isinstance(c, DistributionNode)
        return c

    @operand.setter
    def operand(self, p: DistributionNode) -> None:
        self.children[0] = p

    @property
    def label(self) -> str:
        return "Sample"

    def __str__(self) -> str:
        return "Sample(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return self.operand.support()


class Observation(BMGNode):
    """This represents an observed value of a sample. For example
we might have a prior that a mint produces a coin that is
uniformly unfair. We could then observe a flip of the coin
and if heads, that is small but not zero evidence that
the coin is unfair in the heads direction. Given that
observation, our belief in the true unfairness of the coin
should no loger be uniform."""

    # TODO: Here we treat an observation as node which takes input
    # from a sample and has an associated value. This implementation
    # choice differs from BMG, which does not treat observations as
    # nodes in the graph; since an observation is never the input
    # of any other node, this makes sense.  We might consider
    # following this pattern and making the observation not inherit
    # from BMGNode.
    #
    # TODO: **Observations are logically distinct from models.**
    # That is, it is common to have one model and many different
    # sets of observations. (And similarly but less common, we
    # could imagine having one set of observations used by many
    # models.)  Moreover it is not yet clear how exactly
    # observation nodes are to be generated by the compiler;
    # from what model source code, if any, do we generate these
    # nodes?  This code is only used right now for testing purposes
    # and we need to do significant design work to figure out
    # how users wish to generate observations of models that have
    # been compiled into BMG.
    value: Any
    edges = ["operand"]

    def __init__(self, observed: SampleNode, value: Any):
        self.value = value
        BMGNode.__init__(self, [observed])

    @property
    def observed(self) -> SampleNode:
        c = self.children[0]
        assert isinstance(c, SampleNode)
        return c

    @observed.setter
    def operand(self, p: SampleNode) -> None:
        self.children[0] = p

    @property
    def node_type(self) -> Any:
        return type(self.value)

    @property
    def size(self) -> torch.Size:
        if isinstance(self.value, Tensor):
            return self.value.size()
        return torch.Size([])

    @property
    def label(self) -> str:
        return "Observation " + str(self.value)

    def __str__(self) -> str:
        return str(self.observed) + "=" + str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        g.observe(d[self.observed], self.value)
        return -1

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return f"g.observe(n{d[self.observed]}, {self.value})"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        v = _value_to_cpp(self.value)
        return f"g.observe([n{d[self.observed]}], {v});"

    def support(self) -> Iterator[Any]:
        return []


class Query(BMGNode):
    """A query is a marker on a node in the graph that indicates
to the inference engine that the user is interested in
getting a distribution of values of that node. It always
points to an operator node.

We represent queries in models with the @query annotation;
the compiler causes the returned nodes of such models
to have a query node accumulated into the graph builder.
"""

    # TODO: As with observations, properly speaking there is no
    # need to represent a query as a *node*, and BMG does not
    # do so. We might wish to follow this pattern as well.

    edges = ["operator"]

    def __init__(self, operator: OperatorNode):
        BMGNode.__init__(self, [operator])

    @property
    def operator(self) -> OperatorNode:
        c = self.children[0]
        assert isinstance(c, OperatorNode)
        return c

    @operator.setter
    def operator(self, p: OperatorNode) -> None:
        self.children[0] = p

    @property
    def node_type(self) -> Any:
        return self.operator.node_type

    @property
    def size(self) -> torch.Size:
        return self.operator.size

    @property
    def label(self) -> str:
        return "Query"

    def __str__(self) -> str:
        return "Query(" + str(self.operator) + ")"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        g.query(d[self.operator])
        return -1

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return f"g.query(n{d[self.operator]})"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return f"g.query(n{d[self.operator]});"

    def support(self) -> Iterator[Any]:
        return []


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
