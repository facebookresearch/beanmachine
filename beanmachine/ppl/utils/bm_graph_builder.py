# Copyright (c) Facebook, Inc. and its affiliates.
# from beanmachine.graph import Graph
"""A builder for the BeanMachine Graph language"""

import collections.abc
import functools
import itertools
import math
import operator
import sys
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterator, List

import torch

# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this mModuleNotFoundError
# pyre-ignore-all-errors
from beanmachine.graph import AtomicType, DistributionType, Graph, OperatorType
from beanmachine.ppl.utils.dotbuilder import DotBuilder
from beanmachine.ppl.utils.memoize import memoize
from torch import Tensor, tensor
from torch.distributions import Bernoulli, Beta, Categorical, Normal, Uniform


def prod(x):
    return functools.reduce(operator.mul, x, 1)


builtin_function_or_method = type(abs)


class BMGNode(ABC):
    children: List["BMGNode"]
    edges: List[str]

    def __init__(self, children: List["BMGNode"]):
        self.children = children

    @abstractmethod
    def node_type(self) -> Any:
        pass

    @abstractmethod
    def size(self) -> torch.Size:
        pass

    @abstractmethod
    def label(self) -> str:
        pass

    @abstractmethod
    def _add_to_graph(self, g: Graph, d: Dict["BMGNode", int]) -> int:
        pass

    @abstractmethod
    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        pass

    @abstractmethod
    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        pass

    @abstractmethod
    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        pass

    @abstractmethod
    def support(self) -> Iterator[Any]:
        pass


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


class KnownFunction:
    receiver: BMGNode
    function: Any

    def __init__(self, receiver: BMGNode, function: Any) -> None:
        self.receiver = receiver
        self.function = function


class ConstantNode(BMGNode, metaclass=ABCMeta):
    edges = []
    value: Any

    def __init__(self):
        BMGNode.__init__(self, [])

    @abstractmethod
    def _value_to_cpp(self) -> str:
        pass

    @abstractmethod
    def _value_to_python(self) -> str:
        pass

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = self._value_to_cpp()
        return f"uint n{n} = g.add_constant({v});"

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        v = self._value_to_python()
        return f"n{n} = g.add_constant({v})"

    def support(self) -> Iterator[Any]:
        yield self.value


class BooleanNode(ConstantNode):
    value: bool

    def __init__(self, value: bool):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def node_type(self) -> Any:
        return bool

    def size(self) -> torch.Size:
        return torch.Size([])

    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(bool(self.value))

    def _value_to_python(self) -> str:
        return str(bool(self.value))

    def _value_to_cpp(self) -> str:
        return str(bool(self.value)).lower()

    def __bool__(self) -> bool:
        return self.value


class RealNode(ConstantNode):
    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    # TODO: We may need to represent "positive real" and "real between 0 and 1"
    # TODO: in the type system; how are we to do that?
    def node_type(self) -> Any:
        return float

    def size(self) -> torch.Size:
        return torch.Size([])

    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(float(self.value))

    def _value_to_python(self) -> str:
        return str(float(self.value))

    def _value_to_cpp(self) -> str:
        return str(float(self.value))

    def __bool__(self) -> bool:
        return bool(self.value)


# TODO: Delete the list node; replace it with the map node.
class ListNode(BMGNode):
    # TODO: A list node does not appear in the final graph at this time; it
    # TODO: is useful when constructing the final graph from a BM program that
    # TODO: uses lists. Consider if we want a list type in the graph.
    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)
        self.edges = [str(x) for x in range(len(children))]

    # TODO: Determine the list type by observing the types of the members.
    def node_type(self) -> Any:
        return List[Any]

    def size(self) -> torch.Size:
        raise NotImplementedError()

    def label(self) -> str:
        return "list"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: list nodes are not currently part of the graph
        return -1

    def _to_python(self, d: Dict[BMGNode, int]) -> str:
        # TODO: list nodes are not currently part of the graph
        return ""

    def _to_cpp(self, d: Dict[BMGNode, int]) -> str:
        # TODO: list nodes are not currently part of the graph
        return ""

    def __getitem__(self, key) -> BMGNode:
        if not isinstance(key, RealNode):
            raise ValueError(
                "BeanMachine list must be indexed with a known numeric value"
            )
        return self.children[int(key.value)]

    def support(self) -> Iterator[Any]:
        return []
        # TODO: Do we need this?


class MapNode(BMGNode):
    # TODO: Add the map node to the C++ implementation of the graph.
    # TODO: Do the values all have to be the same type?
    """A map has an even number of children. Even number children are
    keys, odd numbered children are values. The keys must be constants.
    The values can be any node."""

    def __init__(self, children: List[BMGNode]):
        # TODO: Check that keys are all constant nodes.
        # TODO: Check that there is one value for each key.
        # TODO: Verify that there is at least one pair.
        BMGNode.__init__(self, children)
        self.edges = [str(x) for x in range(len(children))]

    def node_type(self) -> Any:
        return Dict

    def size(self) -> torch.Size:
        return self.children(1).size()

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


class TensorNode(ConstantNode):
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
    def node_type(self) -> Any:
        return Tensor

    def size(self) -> torch.Size:
        return self.value.size()

    def label(self) -> str:
        return TensorNode._tensor_to_label(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(self.value)

    def _value_to_python(self) -> str:
        t = TensorNode._tensor_to_python(self.value)
        return f"tensor({t})"

    def _value_to_cpp(self) -> str:
        values = ",".join(str(element) for element in self.value.storage())
        dims = ",".join(str(dim) for dim in self.value.shape)
        return f"torch::from_blob((float[]){{{values}}}, {{{dims}}})"

    def __bool__(self) -> bool:
        return bool(self.value)


class DistributionNode(BMGNode, metaclass=ABCMeta):
    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)

    @abstractmethod
    def sample_type(self) -> Any:
        pass


class BernoulliNode(DistributionNode):
    edges = ["probability"]
    is_logits: bool

    def __init__(self, probability: BMGNode, is_logits: bool = False):
        self.is_logits = is_logits
        DistributionNode.__init__(self, [probability])

    def probability(self) -> BMGNode:
        return self.children[0]

    # TODO: Do we need a generic type for "distribution of X"?
    def node_type(self) -> Any:
        return Bernoulli

    def size(self) -> torch.Size:
        return self.probability().size()

    def sample_type(self) -> Any:
        return self.probability().node_type()

    def label(self) -> str:
        return "Bernoulli" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability()) + ")"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: Handle case where child is logits
        return g.add_distribution(
            DistributionType.BERNOULLI, AtomicType.BOOLEAN, [d[self.probability()]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        return (
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.probability()]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.probability()]}}}));"
        )

    def support(self) -> Iterator[Any]:
        s = self.size()
        return (tensor(i).view(s) for i in itertools.product(*([[0.0, 1.0]] * prod(s))))


class CategoricalNode(DistributionNode):
    """ Graph generator for categorical distributions"""

    edges = ["probability"]
    is_logits: bool

    def __init__(self, probability: BMGNode, is_logits: bool = False):
        self.is_logits = is_logits
        DistributionNode.__init__(self, [probability])

    def probability(self) -> BMGNode:
        return self.children[0]

    # TODO: Do we need a generic type for "distribution of X"?
    def node_type(self) -> Any:
        return Categorical

    def size(self) -> torch.Size:
        return self.probability().size()[0:-1]

    def sample_type(self) -> Any:
        # TODO: When we support bounded integer types
        # TODO: this should indicate that it is a tensor
        # TODO: of bound integers.
        return self.probability().node_type()

    def label(self) -> str:
        return "Categorical" + ("(logits)" if self.is_logits else "")

    def __str__(self) -> str:
        return "Categorical(" + str(self.probability()) + ")"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        # TODO: Handle case where child is logits
        # TODO: This is incorrect.
        return g.add_distribution(
            DistributionType.BERNOULLI, AtomicType.BOOLEAN, [d[self.probability()]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        # TODO: This is incorrect.
        return (
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.probability()]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        # TODO: Handle case where child is logits
        # TODO: This is incorrect.
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.probability()]}}}));"
        )

    def support(self) -> Iterator[Any]:
        s = self.probability().size()
        r = list(range(s[-1]))
        sr = s[:-1]
        return (tensor(i).view(sr) for i in itertools.product(*([r] * prod(sr))))


class NormalNode(DistributionNode):
    edges = ["mu", "sigma"]

    def __init__(self, mu: BMGNode, sigma: BMGNode):
        DistributionNode.__init__(self, [mu, sigma])

    def mu(self) -> BMGNode:
        return self.children[0]

    def sigma(self) -> BMGNode:
        return self.children[1]

    # TODO: Do we need a generic type for "distribution of X"?
    def node_type(self) -> Any:
        return Normal

    def size(self) -> torch.Size:
        return self.mu().size()

    def sample_type(self) -> Any:
        return self.mu().node_type()

    def label(self) -> str:
        return "Normal"

    def __str__(self) -> str:
        return f"Normal({str(self.mu())},{str(self.sigma())})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            # TODO: Fix this when we add the node type to BMG
            DistributionType.BERNOULLI,
            AtomicType.BOOLEAN,
            [d[self.mu()]],
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.mu()]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.mu()]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a normal.
        raise ValueError(f"Normal distribution does not have finite support.")


class UniformNode(DistributionNode):
    edges = ["low", "high"]

    def __init__(self, low: BMGNode, high: BMGNode):
        DistributionNode.__init__(self, [low, high])

    def low(self) -> BMGNode:
        return self.children[0]

    def high(self) -> BMGNode:
        return self.children[1]

    # TODO: Do we need a generic type for "distribution of X"?
    def node_type(self) -> Any:
        return Uniform

    def sample_type(self) -> Any:
        return self.low().node_type()

    def size(self) -> torch.Size:
        return self.low().size()

    def label(self) -> str:
        return "Uniform"

    def __str__(self) -> str:
        return f"Uniform({str(self.low())},{str(self.high())})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            # TODO: Fix this when we add the node type to BMG
            DistributionType.BERNOULLI,
            AtomicType.BOOLEAN,
            [d[self.low()]],
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.low()]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.low()]}}}));"
        )

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a normal.
        raise ValueError(f"Uniform distribution does not have finite support.")


class BetaNode(DistributionNode):
    edges = ["alpha", "beta"]

    def __init__(self, alpha: BMGNode, beta: BMGNode):
        DistributionNode.__init__(self, [alpha, beta])

    def alpha(self) -> BMGNode:
        return self.children[0]

    def beta(self) -> BMGNode:
        return self.children[1]

    # TODO: Do we need a generic type for "distribution of X"?
    def node_type(self) -> Any:
        return Beta

    def sample_type(self) -> Any:
        return self.alpha().node_type()

    def size(self) -> torch.Size:
        return self.alpha().size()

    def label(self) -> str:
        return "Beta"

    def __str__(self) -> str:
        return f"Beta({str(self.alpha())},{str(self.beta())})"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            # TODO: Fix this when we add the node type to BMG
            DistributionType.BERNOULLI,
            AtomicType.BOOLEAN,
            [d[self.probability()]],
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.probability()]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            # TODO: Fix this when we add the node type to BMG
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.probability()]}}}));"
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
    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)


class BinaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    edges = ["left", "right"]
    operator_type: OperatorType

    def __init__(self, left: BMGNode, right: BMGNode):
        OperatorNode.__init__(self, [left, right])

    # TODO: Improve this
    def node_type(self) -> Any:
        if self.left().node_type() == Tensor or self.right().node_type() == Tensor:
            return Tensor
        return float

    def left(self) -> BMGNode:
        return self.children[0]

    def right(self) -> BMGNode:
        return self.children[1]

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_operator(self.operator_type, [d[self.left()], d[self.right()]])

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        ot = self.operator_type
        left = d[self.left()]
        right = d[self.right()]
        return f"n{n} = g.add_operator(graph.{ot}, [n{left}, n{right}])"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        ot = str(self.operator_type).replace(".", "::")
        left = d[self.left()]
        right = d[self.right()]
        return (
            f"uint n{d[self]} = g.add_operator(\n"
            + f"  graph::{ot}, std::vector<uint>({{n{left}, n{right}}}));"
        )


class AdditionNode(BinaryOperatorNode):
    operator_type = OperatorType.ADD

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def size(self) -> torch.Size:
        return (
            torch.zeros(self.left().size()) + torch.zeros(self.right().size())
        ).size()

    def label(self) -> str:
        return "+"

    def __str__(self) -> str:
        return "(" + str(self.left()) + "+" + str(self.right()) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l + r for l in self.left().support() for r in self.right().support()
        )


class MultiplicationNode(BinaryOperatorNode):
    operator_type = OperatorType.MULTIPLY

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def label(self) -> str:
        return "*"

    def size(self) -> torch.Size:
        return (
            torch.zeros(self.left().size()) * torch.zeros(self.right().size())
        ).size()

    def __str__(self) -> str:
        return "(" + str(self.left()) + "*" + str(self.right()) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l * r for l in self.left().support() for r in self.right().support()
        )


class MatrixMultiplicationNode(BinaryOperatorNode):
    # TODO: Fix this.
    operator_type = OperatorType.MULTIPLY

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def label(self) -> str:
        return "*"

    def size(self) -> torch.Size:
        return (
            torch.zeros(self.left().size()).mm(torch.zeros(self.right().size())).size()
        )

    def __str__(self) -> str:
        return "(" + str(self.left()) + "*" + str(self.right()) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            torch.mm(l, r)
            for l in self.left().support()
            for r in self.right().support()
        )


class DivisionNode(BinaryOperatorNode):
    # TODO: We're going to represent division as Mult(x, Power(y, -1)) so
    # TODO: we can remove this class.
    operator_type = OperatorType.MULTIPLY  # TODO

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def label(self) -> str:
        return "/"

    def size(self) -> torch.Size:
        return (
            torch.zeros(self.left().size()) / torch.zeros(self.right().size())
        ).size()

    def __str__(self) -> str:
        return "(" + str(self.left()) + "/" + str(self.right()) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Filter out division by zero?
        return SetOfTensors(
            l / r for l in self.left().support() for r in self.right().support()
        )


class IndexNode(BinaryOperatorNode):
    operator_type = OperatorType.MULTIPLY  # TODO

    def __init__(self, left: MapNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def label(self) -> str:
        return "index"

    def size(self) -> torch.Size:
        return self.left().size()

    def __str__(self) -> str:
        return str(self.left()) + "[" + str(self.right()) + "]"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l for r in self.right().support() for l in self.left()[r].support()
        )


class PowerNode(BinaryOperatorNode):
    # TODO: We haven't added power to the C++ implementation of BMG yet.
    # TODO: When we do, update this.
    operator_type = OperatorType.MULTIPLY  # TODO

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def label(self) -> str:
        return "**"

    def size(self) -> torch.Size:
        return (
            torch.zeros(self.left().size()) ** torch.zeros(self.right().size())
        ).size()

    def __str__(self) -> str:
        return "(" + str(self.left()) + "**" + str(self.right()) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            l ** r for l in self.left().support() for r in self.right().support()
        )


class UnaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    edges = ["operand"]
    operator_type: OperatorType

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

    # TODO: Improve this
    def node_type(self) -> Any:
        return self.operand().node_type()

    def operand(self) -> BMGNode:
        return self.children[0]

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_operator(self.operator_type, [d[self.operand()]])

    def _to_python(self, d: Dict[BMGNode, int]) -> str:
        n = d[self]
        o = d[self.operand()]
        ot = str(self.operator_type)
        return f"n{n} = g.add_operator(graph.{ot}, [n{o}])"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        n = d[self]
        o = d[self.operand()]
        # Since OperatorType is not actually an enum, there is no
        # name attribute to use.
        ot = str(self.operator_type).replace(".", "::")
        return (
            f"uint n{n} = g.add_operator(\n"
            + f"  graph::{ot}, std::vector<uint>({{n{o}}}));"
        )


class NegateNode(UnaryOperatorNode):
    operator_type = OperatorType.NEGATE

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def label(self) -> str:
        return "-"

    def size(self) -> torch.Size:
        return self.operand().size()

    def __str__(self) -> str:
        return "-" + str(self.operand())

    def support(self) -> Iterator[Any]:
        return SetOfTensors(-o for o in self.operand().support())


class NotNode(UnaryOperatorNode):
    # TODO: We do not support NOT in BMG yet; when we do, update this.
    operator_type = OperatorType.NEGATE  # TODO

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def node_type(self) -> Any:
        return bool

    def size(self) -> torch.Size:
        return self.operand().size()

    def label(self) -> str:
        return "not"

    def __str__(self) -> str:
        return "not " + str(self.operand())

    def support(self) -> Iterator[Any]:
        return SetOfTensors(not o for o in self.operand().support())


class ToRealNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_REAL

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def node_type(self) -> Any:
        return float

    def label(self) -> str:
        return "ToReal"

    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "ToReal(" + str(self.operand()) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(float(o) for o in self.operand().support())


class ToTensorNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_TENSOR

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def label(self) -> str:
        return "ToTensor"

    def __str__(self) -> str:
        return "ToTensor(" + str(self.operand()) + ")"

    def size(self) -> torch.Size:
        # TODO: Is this correct?
        return torch.Size([1])

    def node_type(self) -> Any:
        return Tensor

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.tensor(o) for o in self.operand().support())


class ExpNode(UnaryOperatorNode):
    operator_type = OperatorType.EXP

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def label(self) -> str:
        return "Exp"

    def size(self) -> torch.Size:
        return self.operand().size()

    def __str__(self) -> str:
        return "Exp(" + str(self.operand()) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Not always a tensor.
        return SetOfTensors(torch.exp(o) for o in self.operand().support())


class LogNode(UnaryOperatorNode):
    # TODO: We do not support LOG in BMG yet; when we do, update this:
    operator_type = OperatorType.EXP  # TODO

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def label(self) -> str:
        return "Log"

    def size(self) -> torch.Size:
        return self.operand().size()

    def __str__(self) -> str:
        return "Log(" + str(self.operand()) + ")"

    def support(self) -> Iterator[Any]:
        # TODO: Not always a tensor.
        return SetOfTensors(torch.log(o) for o in self.operand().support())


class SampleNode(UnaryOperatorNode):
    operator_type = OperatorType.SAMPLE

    def __init__(self, operand: DistributionNode):
        UnaryOperatorNode.__init__(self, operand)

    def node_type(self) -> Any:
        return self.operand().sample_type()

    def size(self) -> torch.Size:
        return self.operand().size()

    def operand(self) -> DistributionNode:
        c = self.children[0]
        assert isinstance(c, DistributionNode)
        return c

    def label(self) -> str:
        return "Sample"

    def __str__(self) -> str:
        return "Sample(" + str(self.operand()) + ")"

    def support(self) -> Iterator[Any]:
        return self.operand().support()


class Observation(BMGNode):
    edges = ["operand", "value"]

    def __init__(self, observed: SampleNode, value: ConstantNode):
        BMGNode.__init__(self, [observed, value])

    def observed(self) -> SampleNode:
        c = self.children[0]
        assert isinstance(c, SampleNode)
        return c

    def value(self) -> ConstantNode:
        c = self.children[1]
        assert isinstance(c, ConstantNode)
        return c

    def node_type(self) -> Any:
        return type(self.value())

    def size(self) -> torch.Size:
        return self.value().size()

    def label(self) -> str:
        return "Observation"

    def __str__(self) -> str:
        return str(self.observed()) + "=" + str(self.value())

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        v = self.value().value
        g.observe(d[self.observed()], v)
        return -1

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        v = self.value().value
        return f"g.observe(n{d[self.observed()]}, {v})"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        v = self.value()._value_to_cpp()
        return f"g.observe([n{d[self.observed()]}], {v});"

    def support(self) -> Iterator[Any]:
        return []


class Query(BMGNode):
    edges = ["operator"]

    def __init__(self, operator: OperatorNode):
        BMGNode.__init__(self, [operator])

    def operator(self) -> OperatorNode:
        c = self.children[0]
        assert isinstance(c, OperatorNode)
        return c

    def node_type(self) -> Any:
        return self.operator().node_type()

    def size(self) -> torch.Size:
        return self.operator().size()

    def label(self) -> str:
        return "Query"

    def __str__(self) -> str:
        return "Query(" + str(self.operator()) + ")"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        g.query(d[self.operator()])
        return -1

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return f"g.query(n{d[self.operator()]})"

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return f"g.query(n{d[self.operator()]});"

    def support(self) -> Iterator[Any]:
        return []


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
            Normal: self.handle_normal,
            Uniform: self.handle_uniform,
        }

    def remove_orphans(self, roots: List[BMGNode]) -> None:
        nodes = self.nodes.keys()
        self.nodes = {}
        for root in roots:
            self.add_node(root)
        for node in nodes:
            if isinstance(node, SampleNode):
                self.add_node(node)

    def add_node(self, node: BMGNode) -> None:
        # Maintain the invariant that children are always before parents
        # in the list.
        # TODO: If we are ever in a situation where we need to make nodes
        # TODO: and then add the edges later, we'll have to instead do
        # TODO: a deterministic topo sort.
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

    # TODO: Do NOT memoize add_list; if we eventually add a list node to the
    # TODO: underlying graph, revisit this decision.
    # TODO: Remove this; replace lists with maps
    def add_list(self, elements: List[BMGNode]) -> ExpNode:
        node = ListNode(elements)
        self.add_node(node)
        return node

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
        if isinstance(operand, Categorical):
            b = self.handle_categorical(operand.probs)
            return self.add_sample(b)
        if isinstance(operand, Normal):
            b = self.handle_normal(operand.mean, operand.stddev)
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
            if operand.node_type() == Tensor:
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

    def add_observation(self, observed: SampleNode, value: ConstantNode) -> Observation:
        node = Observation(observed, value)
        self.add_node(node)
        return node

    def add_query(self, operator: OperatorNode) -> Query:
        node = Query(operator)
        self.add_node(node)
        return node

    def to_dot(self) -> str:
        db = DotBuilder()
        for node, index in self.nodes.items():
            n = "N" + str(index)
            db.with_node(n, node.label())
            for (child, label) in zip(node.children, node.edges):
                db.with_edge(n, "N" + str(self.nodes[child]), label)
        return str(db)

    def to_bmg(self) -> Graph:
        g = Graph()
        d: Dict[BMGNode, int] = {}
        for node in self.nodes:
            d[node] = node._add_to_graph(g, d)
        return g

    def to_python(self) -> str:
        header = """from beanmachine import graph
from torch import tensor
g = graph.Graph()
"""
        return header + "\n".join(n._to_python(self.nodes) for n in self.nodes)

    def to_cpp(self) -> str:
        return "graph::Graph g;\n" + "\n".join(
            n._to_cpp(self.nodes) for n in self.nodes
        )
