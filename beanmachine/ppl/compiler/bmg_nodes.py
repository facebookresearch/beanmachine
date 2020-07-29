# Copyright (c) Facebook, Inc. and its affiliates.

import torch  # isort:skip  torch has to be imported before graph

import collections
import functools
import itertools
import operator
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Iterator, List

from beanmachine.graph import AtomicType, DistributionType as dt, Graph, OperatorType
from beanmachine.ppl.compiler.bmg_types import (
    Malformed,
    Natural,
    PositiveReal,
    Probability,
    Real,
    Requirement,
    supremum,
    type_of_value,
    upper_bound,
)
from beanmachine.ppl.compiler.internal_error import InternalError
from torch import Tensor, tensor
from torch.distributions.utils import broadcast_all


# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this mModuleNotFoundError
# pyre-ignore-all-errors


def prod(x):
    """Compute the product of a sequence of values of arbitrary length"""
    return functools.reduce(operator.mul, x, 1)


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
    def graph_type(self) -> type:
        """The type of the node in the graph type system."""
        pass

    @property
    @abstractmethod
    def inf_type(self) -> type:
        """BMG nodes have type requirements on their inputs; the *infimum type* of
a node is the *smallest* BMG type that a node may be converted to if required by
an input."""
        pass

    @property
    @abstractmethod
    def requirements(self) -> List[Requirement]:
        """BMG nodes have type requirements on their inputs; this property
produces a list of Requirements; a type indicates an exact Requirement;
an UpperBound indicates that the input must be smaller than or equal to
the Requirement."""
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

    # Many of the node types defined in this module have no direct counterparts
    # in BMG; they are generated during execution of the model and accumulated
    # into a graph builder. They must then be transformed into semantically-
    # equivalent supported nodes.

    def _supported_in_bmg(self) -> bool:
        return False

    def _add_to_graph(self, g: Graph, d: Dict["BMGNode", int]) -> int:
        """This adds a node to an in-memory BMG instance. Each node
in BMG is associated with an integer handle; this returns
the handle for this node assigned by BMG."""
        raise InternalError(f"{str(type(self))} is not supported in BMG.")

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        """We can emit the graph as a Python program which, when executed,
builds a BMG instance. This method returns a string of Python
code to construct this node. The dictionary associates a unique
integer with each node that can be used to construct an identifier."""
        raise InternalError(f"{str(type(self))} is not supported in BMG.")

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        """We can emit the graph as a C++ program which, when executed,
builds a BMG instance. This method returns a string of C++
code to construct this node. The dictionary associates a unique
integer with each node that can be used to construct an identifier."""
        raise InternalError(f"{str(type(self))} is not supported in BMG.")

    @abstractmethod
    def support(self) -> Iterator[Any]:
        """To build the graph of all possible control flows through
the model we need to know for any given node what are
all the possible values it could attain; we require that
the set be finite and will throw an exception if it is not."""
        pass


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


# ####
# #### Nodes representing constant values
# ####


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

    @property
    def inf_type(self) -> type:
        # The infimum type of a constant is derived from the value,
        # not from the kind of constant node we have. For instance,
        # a NaturalNode containing zero and a TensorNode containing
        # tensor([[[0.0]]]) both have infimum type "bool" because
        # we can convert zero to False, and bool is the smallest type
        # in the lattice. Remember, the infimum type answers the question
        # "what types can this node be converted to?" and not "what is the
        # type of this node?"
        return type_of_value(self.value)

    @property
    def requirements(self) -> List[Requirement]:
        return []

    @abstractmethod
    def _value_to_python(self) -> str:
        pass

    def _supported_in_bmg(self) -> bool:
        return True

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
    def graph_type(self) -> type:
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


class NaturalNode(ConstantNode):
    """An integer constant restricted to non-negative values"""

    value: int

    def __init__(self, value: int):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def graph_type(self) -> type:
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


class PositiveRealNode(ConstantNode):
    """A real constant restricted to non-negative values"""

    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def graph_type(self) -> type:
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


class ProbabilityNode(ConstantNode):
    """A real constant restricted to values from 0.0 to 1.0"""

    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def graph_type(self) -> type:
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


class RealNode(ConstantNode):
    """An unrestricted real constant"""

    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    @property
    def graph_type(self) -> type:
        return Real

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

    @property
    def graph_type(self) -> type:
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


# ####
# #### Nodes representing distributions
# ####


class DistributionNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all nodes that represent
probability distributions."""

    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)


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
    def graph_type(self) -> type:
        return bool

    @property
    def inf_type(self) -> type:
        return bool

    @property
    def requirements(self) -> List[Requirement]:
        # The input to a Bernoulli must be exactly a real number if "logits",
        # and otherwise must be a Probability.
        return [Real if self.is_logits else Probability]

    @property
    def size(self) -> torch.Size:
        return self.probability.size

    @property
    def label(self) -> str:
        return "Bernoulli" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability) + ")"

    def _supported_in_bmg(self) -> bool:
        return True

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
        s = self.size
        return (tensor(i).view(s) for i in itertools.product(*([[0.0, 1.0]] * prod(s))))


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
    def graph_type(self) -> type:
        return Probability

    @property
    def inf_type(self) -> type:
        return Probability

    @property
    def requirements(self) -> List[Requirement]:
        # Both inputs to a beta must be positive reals
        return [PositiveReal, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.alpha.size

    @property
    def label(self) -> str:
        return "Beta"

    def __str__(self) -> str:
        return f"Beta({str(self.alpha)},{str(self.beta)})"

    def _supported_in_bmg(self) -> bool:
        return True

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
        raise ValueError("Beta distribution does not have finite support.")


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
    def graph_type(self) -> type:
        return Natural

    @property
    def inf_type(self) -> type:
        return Natural

    @property
    def requirements(self) -> List[Requirement]:
        # The left input to a binomial must be a natural; the right
        # input must be a real number if "logits" and a Probability
        # otherwise.
        return [Natural, Real if self.is_logits else Probability]

    @property
    def size(self) -> torch.Size:
        return broadcast_all(
            torch.zeros(self.count.size), torch.zeros(self.probability.size)
        ).size()

    @property
    def label(self) -> str:
        return "Binomial" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return f"Binomial({self.count}, {self.probability})"

    def _supported_in_bmg(self) -> bool:
        return not self.is_logits

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: Fix this when we support binomial logits.
        if self.is_logits:
            raise InternalError("Binomial with logits is not supported in BMG.")
        return g.add_distribution(
            dt.BINOMIAL, AtomicType.NATURAL, [d[self.count], d[self.probability]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        if self.is_logits:
            raise InternalError("Binomial with logits is not supported in BMG.")
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.BINOMIAL,\n"
            + "  graph.AtomicType.NATURAL,\n"
            + f"  [n{d[self.count]}, n{d[self.probability]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        if self.is_logits:
            raise InternalError("Binomial with logits is not supported in BMG.")
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
    def graph_type(self) -> Any:
        return Natural

    @property
    def inf_type(self) -> type:
        return Natural

    @property
    def requirements(self) -> List[Requirement]:
        # The input to a categorical must be a tensor.
        # TODO: We do not yet support categoricals in BMG and when we do,
        # we will likely need to implement a simplex type rather than
        # a tensor. Fix this up when categoricals are implemented.
        return [Tensor]

    @property
    def size(self) -> torch.Size:
        return self.probability.size[0:-1]

    @property
    def label(self) -> str:
        return "Categorical" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return "Categorical(" + str(self.probability) + ")"

    def support(self) -> Iterator[Any]:
        s = self.probability.size
        r = list(range(s[-1]))
        sr = s[:-1]
        return (tensor(i).view(sr) for i in itertools.product(*([r] * prod(sr))))


class Chi2Node(DistributionNode):
    """The chi2 distribution is a distribution of positive
real numbers; it is a special case of the gamma distribution."""

    edges = ["df"]

    def __init__(self, df: BMGNode):
        DistributionNode.__init__(self, [df])

    @property
    def df(self) -> BMGNode:
        return self.children[0]

    @df.setter
    def df(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def graph_type(self) -> type:
        return PositiveReal

    @property
    def inf_type(self) -> type:
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        return [PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.df.size

    @property
    def label(self) -> str:
        return "Chi2"

    def __str__(self) -> str:
        return f"Chi2({str(self.df)})"

    def _supported_in_bmg(self) -> bool:
        # Not supported directly; we replace it with a gamma, which is supported.
        return False

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a Chi2.
        raise ValueError("Chi2 distribution does not have finite support.")


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
    def graph_type(self) -> type:
        return Tensor

    @property
    def inf_type(self) -> type:
        return Tensor

    @property
    def requirements(self) -> List[Requirement]:
        # The input to a Direchlet must be a tensor.
        # TODO: We do not yet support Dirichlet in BMG; when we do
        # verify that this is correct. Also, we may wish at that
        # time to also note that the sample type is a simplex.
        return [Tensor]

    @property
    def size(self) -> torch.Size:
        return self.concentration.size

    @property
    def label(self) -> str:
        return "Dirichlet"

    def __str__(self) -> str:
        return f"Dirichlet({str(self.concentration)})"

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a Dirichlet.
        raise ValueError("Dirichlet distribution does not have finite support.")


class FlatNode(DistributionNode):

    """The Flat distribution the standard uniform distribution from 0.0 to 1.0."""

    edges = []

    def __init__(self):
        DistributionNode.__init__(self, [])

    @property
    def graph_type(self) -> type:
        return Probability

    @property
    def inf_type(self) -> type:
        return Probability

    @property
    def requirements(self) -> List[Requirement]:
        return []

    def _supported_in_bmg(self) -> bool:
        return True

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(dt.FLAT, AtomicType.PROBABILITY, [])

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.FLAT,\n"
            + "  graph.AtomicType.PROBABILITY,\n"
            + "  [])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::FLAT,\n"
            + "  graph::AtomicType::PROBABILITY,\n"
            + "  std::vector<uint>({}));"
        )

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    @property
    def label(self) -> str:
        return "Flat"

    def __str__(self) -> str:
        return "Flat()"

    def support(self) -> Iterator[Any]:
        raise ValueError("Flat distribution does not have finite support.")


class GammaNode(DistributionNode):
    """The gamma distribution is a distribution of positive
real numbers characterized by positive real concentration and rate
parameters."""

    edges = ["concentration", "rate"]

    def __init__(self, concentration: BMGNode, rate: BMGNode):
        DistributionNode.__init__(self, [concentration, rate])

    @property
    def concentration(self) -> BMGNode:
        return self.children[0]

    @concentration.setter
    def concentration(self, p: BMGNode) -> None:
        self.children[0] = p

    @property
    def rate(self) -> BMGNode:
        return self.children[1]

    @rate.setter
    def rate(self, p: BMGNode) -> None:
        self.children[1] = p

    @property
    def graph_type(self) -> type:
        return PositiveReal

    @property
    def inf_type(self) -> type:
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        return [PositiveReal, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.concentration.size

    @property
    def label(self) -> str:
        return "Gamma"

    def __str__(self) -> str:
        return f"Gamma({str(self.concentration)}, {str(self.rate)})"

    def _supported_in_bmg(self) -> bool:
        return True

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            dt.GAMMA, AtomicType.POS_REAL, [d[self.concentration], d[self.rate]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(\n"
            + "  graph.DistributionType.GAMMA,\n"
            + "  graph.AtomicType.POS_REAL,\n"
            + f"  [n{d[self.concentration]}, n{d[self.rate]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::GAMMA,\n"
            + "  graph::AtomicType::POS_REAL,\n"
            + f"  std::vector<uint>({{n{d[self.concentration]}, n{d[self.rate]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a Gamma.
        raise ValueError("Gamma distribution does not have finite support.")


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
    def graph_type(self) -> type:
        return PositiveReal

    @property
    def inf_type(self) -> type:
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        # The input to a HalfCauchy must be a positive real.
        return [PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.scale.size

    @property
    def label(self) -> str:
        return "HalfCauchy"

    def __str__(self) -> str:
        return f"HalfCauchy({str(self.scale)})"

    def _supported_in_bmg(self) -> bool:
        return True

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
        raise ValueError("HalfCauchy distribution does not have finite support.")


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
    def graph_type(self) -> type:
        return Real

    @property
    def inf_type(self) -> type:
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        # The mean of a normal must be a real; the standard deviation
        # must be a positive real.
        return [Real, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.mu.size

    @property
    def label(self) -> str:
        return "Normal"

    def __str__(self) -> str:
        return f"Normal({str(self.mu)},{str(self.sigma)})"

    def _supported_in_bmg(self) -> bool:
        return True

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
        raise ValueError("Normal distribution does not have finite support.")


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
    def graph_type(self) -> type:
        return Real

    @property
    def inf_type(self) -> type:
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        return [PositiveReal, Real, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.df.size

    @property
    def label(self) -> str:
        return "StudentT"

    def __str__(self) -> str:
        return f"StudentT({str(self.df)},{str(self.loc)},{str(self.scale)})"

    def _supported_in_bmg(self) -> bool:
        return True

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
        raise ValueError("StudentT distribution does not have finite support.")


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
    def graph_type(self) -> type:
        return Real

    @property
    def inf_type(self) -> type:
        # TODO: We will probably need to be smarter here
        # once this is implemented in BMG.
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: We do not yet support arbitrary Uniform distributions in
        # BMG; when we do, revisit this code.
        # TODO: If we know that a Uniform is bounded by constants 0.0 and 1.0,
        # we can generate a Flat distribution node for BMG.
        return [Real, Real]

    @property
    def size(self) -> torch.Size:
        return self.low.size

    @property
    def label(self) -> str:
        return "Uniform"

    def __str__(self) -> str:
        return f"Uniform({str(self.low)},{str(self.high)})"

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a uniform.
        raise ValueError("Uniform distribution does not have finite support.")


# ####
# #### Operators
# ####


class OperatorNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all operators.
The children are the operands of each operator."""

    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)


# ####
# #### Ternary operators
# ####


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
    def graph_type(self) -> type:
        if self.consequence.graph_type == self.alternative.graph_type:
            return self.consequence.graph_type
        return Malformed

    @property
    def inf_type(self) -> type:
        return supremum(self.consequence.inf_type, self.alternative.inf_type)

    @property
    def requirements(self) -> List[Requirement]:
        # We require that the consequence and alternative types of the if-then-else
        # be exactly the same. In order to minimize the output type of the node
        # we will take the supremum of the infimums of the input types, and then
        # require that the inputs each be of that type.
        # The condition input must be a bool.
        it = self.inf_type
        return [bool, it, it]

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

    def _supported_in_bmg(self) -> bool:
        return True


# ####
# #### Binary operators
# ####


class BinaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    """This is the base class for all binary operators."""

    edges = ["left", "right"]
    operator_type: OperatorType

    def __init__(self, left: BMGNode, right: BMGNode):
        OperatorNode.__init__(self, [left, right])

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
        if not self._supported_in_bmg():
            raise InternalError(f"{str(type(self))} is not supported in BMG.")
        return g.add_operator(self.operator_type, [d[self.left], d[self.right]])

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        if not self._supported_in_bmg():
            raise InternalError(f"{str(type(self))} is not supported in BMG.")
        n = d[self]
        ot = self.operator_type
        left = d[self.left]
        right = d[self.right]
        return f"n{n} = g.add_operator(graph.{ot}, [n{left}, n{right}])"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        if not self._supported_in_bmg():
            raise InternalError(f"{str(type(self))} is not supported in BMG.")
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
    def inf_type(self) -> type:
        return supremum(self.left.inf_type, self.right.inf_type, PositiveReal)

    @property
    def graph_type(self) -> type:
        if self.left.graph_type == self.right.graph_type:
            return self.left.graph_type
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # We require that the input types of an addition be exactly the same.
        # In order to minimize the output type of the node we will take the
        # supremum of the infimums of the input types, and then require that
        # the inputs each be of that type.
        it = self.inf_type
        return [it, it]

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
            el + ar for el in self.left.support() for ar in self.right.support()
        )

    def _supported_in_bmg(self) -> bool:
        return True


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
            el / ar for el in self.left.support() for ar in self.right.support()
        )

    @property
    def inf_type(self) -> type:
        # TODO: We do not support division in BMG yet; when we do, implement
        # this correctly. Best guess so far: division is defined only on
        # positive reals, reals and tensors.
        return supremum(self.left.inf_type, self.right.inf_type, PositiveReal)

    @property
    def graph_type(self) -> type:
        # TODO: We do not support division in BMG yet; when we do, implement
        # this correctly. Best guess so far: left, right and output types
        # must be the same, and must be PositiveReal, Real or tensor.
        lgt = self.left.graph_type
        if lgt != self.right.graph_type:
            return Malformed
        if lgt != PositiveReal and lgt != Real and lgt != Tensor:
            return Malformed
        return lgt

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: We do not support division in BMG yet; when we do, implement
        # this correctly.
        it = self.inf_type
        return [it, it]


class MapNode(BMGNode):

    """This class represents a point in a program where there are
multiple control flows based on the value of a stochastic node."""

    # For example, suppose we have this contrived model:
    #
    #   @bm.random_variable def weird(i):
    #     if i == 0:
    #       return Normal(0.0, 1.0)
    #     return Normal(1.0, 1.0)
    #
    #   @bm.random_variable def flips():
    #     return Binomial(2, 0.5)
    #
    #   @bm.random_variable def really_weird():
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
    def inf_type(self) -> type:
        # The inf type of a map is the supremum of the types of all
        # its inputs.
        return supremum(
            *[self.children[i * 2 + 1].inf_type for i in range(len(self.children) // 2)]
        )

    @property
    def graph_type(self) -> type:
        first = self.children[0].graph_type
        for i in range(len(self.children) // 2):
            if self.children[i * 2 + 1].graph_type != first:
                return Malformed
        return first

    @property
    def requirements(self) -> List[Requirement]:
        it = self.inf_type
        # TODO: This isn't quite right; when we support this kind of node
        # in BMG, fix this.
        return [upper_bound(Tensor), it] * (len(self.children) // 2)

    @property
    def size(self) -> torch.Size:
        return self.children[1].size

    @property
    def label(self) -> str:
        return "map"

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


class IndexNode(BinaryOperatorNode):
    """This represents a stochastic choice of multiple options; the left
operand must be a map, and the right is the stochastic value used to
choose an element from the map."""

    # See notes on MapNode for an explanation of this code.

    def __init__(self, left: MapNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "index"

    @property
    def inf_type(self) -> type:
        # The inf type of an index is that of its map.
        return self.left.inf_type

    @property
    def graph_type(self) -> type:
        return self.left.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        it = self.inf_type
        # TODO: This isn't quite right; when we support this kind of node
        # in BMG, fix this.
        return [upper_bound(Tensor), it]

    @property
    def size(self) -> torch.Size:
        return self.left.size

    def __str__(self) -> str:
        return str(self.left) + "[" + str(self.right) + "]"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el for ar in self.right.support() for el in self.left[ar].support()
        )


class MatrixMultiplicationNode(BinaryOperatorNode):

    """"This represents a matrix multiplication."""

    # TODO: We do not yet have an implementation of matrix
    # multiplication as a BMG node. When we do, implement the
    # feature here.

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "*"

    @property
    def inf_type(self) -> type:
        # TODO: We do not yet support matrix multiplication in BMG;
        # when we do, revisit this code.
        return supremum(self.left.inf_type, self.right.inf_type)

    @property
    def graph_type(self) -> type:
        if self.left.graph_type == self.right.graph_type:
            return self.left.graph_type
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: We do not yet support matrix multiplication in BMG;
        # when we do, revisit this code.
        it = self.inf_type
        return [it, it]

    @property
    def size(self) -> torch.Size:
        return torch.zeros(self.left.size).mm(torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "*" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            torch.mm(el, ar)
            for el in self.left.support()
            for ar in self.right.support()
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
    def graph_type(self) -> type:
        # A multiplication node must have its left, right and output
        # types all the same, and that type must be Probability or
        # larger. If these conditions are not met then the node is malformed.
        # However, we can convert some multiplications into if-then-else
        # nodes which are well-formed. We will express these facts
        # in the requirements and inf type computation, and the problem fixer
        # will turn malformed multiplications into a correct form.
        lgt = self.left.graph_type
        if lgt != self.right.graph_type:
            return Malformed
        if lgt == bool or lgt == Natural:
            return Malformed
        return lgt

    @property
    def inf_type(self) -> type:
        # As noted above, we can multiply two probabilities, two positive
        # reals, two reals or two tensors and get the same type out. However
        # if we have a model in which a bool or natural is multiplied by a
        # bool or natural, then we can create a legal BMG graph as follows:
        #
        # bool1 * bool2 can become "if bool1 then bool2 else false"
        # bool * nat and nat * bool can become "if bool then nat else 0"
        # nat * nat must convert both nats to positive real.
        #
        # So what then is the inf type? Remember, the inf type is the smallest
        # type that this node can be converted to, so let's say that.

        # If either operand is bool then we can convert to an if-then-else
        # and keep the type the same as the other operand:

        lit = self.left.inf_type
        rit = self.right.inf_type
        if lit == bool:
            return rit
        if rit == bool:
            return lit

        # If neither type is bool then the best we can do is the sup
        # of the left type, the right type, and Probability.
        return supremum(self.left.inf_type, self.right.inf_type, Probability)

    @property
    def requirements(self) -> List[Requirement]:
        # As noted above, we have special handling if an operand to a multiplication
        # is a bool. In those cases, we can simply impose a requirement that can
        # always be met: that the operands be of their inf types.
        lit = self.left.inf_type
        rit = self.right.inf_type
        if lit == bool or rit == bool:
            return [lit, rit]
        # If we're not in one of those special cases then we require that both
        # operands be the inf type, which, recall, is the sup of the left, right
        # and Probability.
        it = self.inf_type
        return [it, it]

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) * torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "*" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el * ar for el in self.left.support() for ar in self.right.support()
        )

    def _supported_in_bmg(self) -> bool:
        return True


class PowerNode(BinaryOperatorNode):
    """This represents an x-to-the-y operation."""

    operator_type = OperatorType.POW

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def label(self) -> str:
        return "**"

    @property
    def inf_type(self) -> type:
        # Given the inf types of the operands, what is the smallest
        # possible type we could make the result?
        #
        # BMG supports a power node that has seven possible combinations of
        # base and exponent type:
        #
        # T ** T   --> T
        # P ** R+  --> P
        # P ** R   --> R+
        # R+ ** R+ --> R+
        # R+ ** R  --> R+
        # R ** R+  --> R
        # R ** R   --> R
        #
        # Note that P ** R is the only case where the type of the result is not
        # equal to the type of the base.
        #
        # The smallest type we can make the return is:
        # * treat the base type as the larger of its inf type and Probability
        # * treat the exp type as the larger of its inf type and Positive Real.
        # * return the best match from the table above.
        #
        # TODO: We could support x ** b where b is a bool by generating it
        # as "if b then x else 1", and that's of the same type as x. This
        # would allow us to generate:
        # B ** B --> B
        # N ** B --> N
        #
        # TODO: We could support b ** n where b is bool and n is a natural
        # constant. If n is the constant zero then the result is just
        # the Boolean constant true; if n is a non-zero constant then
        # b ** n is simply b.
        #
        # NOTE: We CANNOT support b ** n where b is bool and n is a
        # non-constant natural and the result is bool. That would
        # have the semantics of "if b then true else n == 0" but we do
        # not have an equality operator on naturals in BMG.
        #
        # NOTE: We CANNOT support n1 ** n2 where both are naturals, where
        # n2 is a constant and where the result is natural, because we
        # do not have a multiplication operation on naturals. The best
        # we can do is convert both to R+, which is what we'd have to
        # do for the multiplication.

        inf_base = supremum(self.left.inf_type, Probability)
        inf_exp = supremum(self.right.inf_type, PositiveReal)

        if inf_base == Tensor or inf_exp == Tensor:
            return Tensor
        if inf_base == Probability and inf_exp == Real:
            return PositiveReal
        return inf_base

    def _supported_in_bmg(self) -> bool:
        return True

    @property
    def graph_type(self) -> type:
        # Figure out which of these seven cases we are in; otherwise
        # return Malformed.

        # T ** T   --> T
        # P ** R+  --> P
        # P ** R   --> R+
        # R+ ** R+ --> R+
        # R+ ** R  --> R+
        # R ** R+  --> R
        # R ** R   --> R
        lt = self.left.graph_type
        rt = self.right.graph_type
        if lt == Tensor and rt == Tensor:
            return Tensor
        if lt == Tensor or rt == Tensor:
            return Malformed
        if lt != Probability and lt != PositiveReal and lt != Real:
            return Malformed
        if rt != PositiveReal and rt != Real:
            return Malformed
        if lt == Probability and rt == Real:
            return PositiveReal
        return lt

    @property
    def requirements(self) -> List[Requirement]:
        # T ** T
        # P ** R+
        # P ** R
        # R+ ** R+
        # R+ ** R
        # R ** R+
        # R ** R

        # TODO: We could support x ** b where b is a bool by generating it
        # as "if b then x else 1", and that's of the same type as x.

        inf_base = supremum(self.left.inf_type, Probability)
        inf_exp = supremum(self.left.inf_type, PositiveReal)

        if inf_base == Tensor or inf_exp == Tensor:
            return [Tensor, Tensor]
        return [inf_base, inf_exp]

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) ** torch.zeros(self.right.size)).size()

    def __str__(self) -> str:
        return "(" + str(self.left) + "**" + str(self.right) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el ** ar for el in self.left.support() for ar in self.right.support()
        )


# ####
# #### Unary operators
# ####


class UnaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    """This is the base type of unary operator nodes."""

    edges = ["operand"]
    operator_type: OperatorType

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

    @property
    def operand(self) -> BMGNode:
        return self.children[0]

    @operand.setter
    def operand(self, p: BMGNode) -> None:
        self.children[0] = p

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        if not self._supported_in_bmg():
            raise InternalError(f"{str(type(self))} is not supported in BMG.")
        return g.add_operator(self.operator_type, [d[self.operand]])

    def _to_python(self, d: Dict[BMGNode, int]) -> str:
        if not self._supported_in_bmg():
            raise InternalError(f"{str(type(self))} is not supported in BMG.")
        n = d[self]
        o = d[self.operand]
        ot = str(self.operator_type)
        return f"n{n} = g.add_operator(graph.{ot}, [n{o}])"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        if not self._supported_in_bmg():
            raise InternalError(f"{str(type(self))} is not supported in BMG.")
        n = d[self]
        o = d[self.operand]
        # Since OperatorType is not actually an enum, there is no
        # name attribute to use.
        ot = str(self.operator_type).replace(".", "::")
        return (
            f"uint n{n} = g.add_operator(\n"
            + f"  graph::{ot}, std::vector<uint>({{n{o}}}));"
        )


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
    def inf_type(self) -> type:
        if self.operand.inf_type == Tensor:
            return Tensor
        return PositiveReal

    @property
    def graph_type(self) -> type:
        ot = self.operand.graph_type
        if ot == Tensor:
            return Tensor
        if ot == Real or ot == PositiveReal:
            return PositiveReal
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # If the operand is a tensor or real, the requirement
        # has been met; if not, we require that it be converted
        # to positive real.
        return [supremum(self.operand.inf_type, PositiveReal)]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Exp(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Not always a tensor.
        return SetOfTensors(torch.exp(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class LogNode(UnaryOperatorNode):
    """This represents a log operation; it is generated when
a model contains calls to Tensor.log or math.log."""

    operator_type = OperatorType.LOG

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    # The log node is a bit odd in that it requires *either* a positive
    # real *or* a tensor input, but does not accept real, which is between
    # those two. This then leads to some unusual code when we are working
    # out the smallest type this is convertible to, the type it really is,
    # and the requirements on the operand.

    @property
    def inf_type(self) -> type:
        ot = self.operand.inf_type
        if ot == Tensor or ot == Real:
            return Tensor
        return Real

    @property
    def graph_type(self) -> type:
        ot = self.operand.graph_type
        if ot == Tensor:
            return Tensor
        if ot == PositiveReal:
            return Real
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        it = self.operand.inf_type
        if it == Tensor or it == Real:
            return [Tensor]
        return [PositiveReal]

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

    def _supported_in_bmg(self) -> bool:
        return True


# BMG supports three different kinds of negation:

# * The "complement" node with a Boolean operand has the semantics
#   of logical negation.  The input and output are both bool.
#
# * The "complement" node with a probability operand has the semantics
#   of (1 - p). The input and output are both probability.
#
# * The "negate" node has the semantics of (0 - x). The input and output
#   are both real or both tensor.
#
# Note that there is no subtraction operator in BMG; to express x - y
# we generate nodes as though (x + (-y)) was written; that is, the
# sum of x and a real-number negation of y.
#
# This presents several problems when accumulating a graph while executing
# a Python model, and then turning said graph into a valid BMG, particularly
# during type analysis.
#
# Our strategy is:
#
# * When we accumulate the graph we will create nodes for addition
#   (AdditionNode), unary negation (NegationNode) and the "not"
#   operator (NotNode).  We will not generate "complement" nodes
#   directly from Python source.
#
# * After accumulating the graph we will do type analysis and use
#   that to drive a rewriting pass. The rewriting pass will perform
#   these tasks:
#
#   (1) "not" nodes whose operands are bool will be converted into
#       "complement" nodes.
#
#   (2) "not" nodes whose operands are not bool will produce an error.
#       (The "not" operator applied to a non-bool x in Python has the
#       semantics of "x == 0" and we do not have any way to represent
#       these semantics in BMG.
#
#   (3) Call a constant "one-like" if it is True, 1, 1.0, or a single-
#       valued tensor with a one-like value. If we have a one-like node,
#       call it 1 for short, then we will look for patterns in the
#       accumulated graph such as
#
#       1 + (-p)
#       (-p) + 1
#       -(p + -1)
#       -(-1 + p)
#
#       and replace them with "complement" nodes.
#
#   (4) Other usages of binary + and unary - in the Python model will
#       be converted to BMG following the rules for addition and negation
#       in BMG: negation must be real or tensor valued, and so on.


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
    def inf_type(self) -> type:
        return supremum(self.operand.inf_type, Real)

    @property
    def graph_type(self) -> type:
        return self.operand.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        # We require that the input type be identical to the output type,
        # and the smallest possible output type is the infimum type of
        # the input.
        return [self.inf_type]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "-" + str(self.operand)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(-o for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class NotNode(UnaryOperatorNode):
    """This represents a logical not that appears in the Python model."""

    # TODO: We do not support NOT in BMG yet; when we do, update this.
    operator_type = OperatorType.NEGATE  # TODO

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def inf_type(self) -> type:
        # TODO: When we support this node in BMG, revisit this code.
        return bool

    @property
    def graph_type(self) -> type:
        return self.operand.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: When we support this node in BMG, revisit this code.
        return [self.inf_type]

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
    def inf_type(self) -> type:
        # The infimum type of a sample is that of its distribution.
        return self.operand.inf_type

    @property
    def graph_type(self) -> type:
        return self.operand.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        return [self.inf_type]

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

    def _supported_in_bmg(self) -> bool:
        return True


class ToRealNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_REAL

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def graph_type(self) -> type:
        return Real

    @property
    def inf_type(self) -> type:
        # A ToRealNode's output is always real
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        # A ToRealNode's input must be real or smaller.
        return [upper_bound(Real)]

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

    def _supported_in_bmg(self) -> bool:
        return True


class ToPositiveRealNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_POS_REAL

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def graph_type(self) -> type:
        return PositiveReal

    @property
    def inf_type(self) -> type:
        # A ToPositiveRealNode's output is always PositiveReal
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        # A ToPositiveRealNode's input must be PositiveReal or smaller.
        return [upper_bound(PositiveReal)]

    @property
    def label(self) -> str:
        return "ToPosReal"

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "ToPosReal(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(float(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class ToTensorNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_TENSOR

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def label(self) -> str:
        return "ToTensor"

    @property
    def inf_type(self) -> type:
        # The output of a ToTensorNode is always a tensor.
        return Tensor

    @property
    def graph_type(self) -> type:
        return Tensor

    @property
    def requirements(self) -> List[Requirement]:
        # A ToTensorNode's input must be Tensor or smaller.
        return [upper_bound(Tensor)]

    def __str__(self) -> str:
        return "ToTensor(" + str(self.operand) + ")"

    @property
    def size(self) -> torch.Size:
        # TODO: Is this correct?
        return torch.Size([1])

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.tensor(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


# ####
# #### Marker nodes
# ####

# TODO: Do we also need to represent factors?


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
    def inf_type(self) -> type:
        # TODO: Since an observation node is never consumed, it's not actually
        # meaningful to compute its type, but we can potentially use this
        # to check for errors; for example, if we have an observation with
        # value 0.5 on an operation known to be of type Natural then we can
        # flag that as a likely error.
        return self.observed.inf_type

    @property
    def graph_type(self) -> type:
        return self.observed.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        return [self.inf_type]

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

    def _supported_in_bmg(self) -> bool:
        return True

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

We represent queries in models with the @bm.functional annotation;
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
    def graph_type(self) -> type:
        return self.operator.graph_type

    @property
    def inf_type(self) -> type:
        return self.operator.inf_type

    @property
    def requirements(self) -> List[Requirement]:
        return [self.inf_type]

    @property
    def size(self) -> torch.Size:
        return self.operator.size

    @property
    def label(self) -> str:
        return "Query"

    def __str__(self) -> str:
        return "Query(" + str(self.operator) + ")"

    def _supported_in_bmg(self) -> bool:
        return True

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        g.query(d[self.operator])
        return -1

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return f"g.query(n{d[self.operator]})"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return f"g.query(n{d[self.operator]});"

    def support(self) -> Iterator[Any]:
        return []
