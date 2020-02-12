# Copyright (c) Facebook, Inc. and its affiliates.
# from beanmachine.graph import Graph
"""A builder for the BeanMachine Graph language"""

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List

# TODO: For reasons unknown, Pyre is unable to find type information about
# TODO: beanmachine.graph from beanmachine.ppl.  I'll figure out why later;
# TODO: for now, we'll just turn off error checking in this mModuleNotFoundError
# pyre-ignore-all-errors
from beanmachine.graph import AtomicType, DistributionType, Graph, OperatorType
from beanmachine.ppl.utils.dotbuilder import DotBuilder
from torch import Tensor


class BMGNode(ABC):
    children: List["BMGNode"]
    edges: List[str]

    def __init__(self, children: List["BMGNode"]):
        self.children = children

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


class BooleanNode(ConstantNode):
    value: bool

    def __init__(self, value: bool):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(bool(self.value))

    def _value_to_python(self) -> str:
        return str(bool(self.value))

    def _value_to_cpp(self) -> str:
        return str(bool(self.value)).lower()


class RealNode(ConstantNode):
    value: float

    def __init__(self, value: float):
        ConstantNode.__init__(self)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def label(self) -> str:
        return str(self.value)

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_constant(float(self.value))

    def _value_to_python(self) -> str:
        return str(float(self.value))

    def _value_to_cpp(self) -> str:
        return str(float(self.value))


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


class DistributionNode(BMGNode, metaclass=ABCMeta):
    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)


class BernoulliNode(DistributionNode):
    edges = ["probability"]

    def __init__(self, probability: BMGNode):
        DistributionNode.__init__(self, [probability])

    def probability(self) -> BMGNode:
        return self.children[0]

    def label(self) -> str:
        return "Bernoulli"

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability()) + ")"

    def _add_to_graph(self, g: Graph, d: Dict[BMGNode, int]) -> int:
        return g.add_distribution(
            DistributionType.BERNOULLI, AtomicType.BOOLEAN, [d[self.probability()]]
        )

    def _to_python(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"n{d[self]} = g.add_distribution(graph.DistributionType.BERNOULLI, "
            + f"graph.AtomicType.BOOLEAN, [n{d[self.probability()]}])"
        )

    def _to_cpp(self, d: Dict["BMGNode", int]) -> str:
        return (
            f"uint n{d[self]} = g.add_distribution(\n"
            + "  graph::DistributionType::BERNOULLI,\n"
            + "  graph::AtomicType::BOOLEAN,\n"
            + f"  std::vector<uint>({{n{d[self.probability()]}}}));"
        )


class OperatorNode(BMGNode, metaclass=ABCMeta):
    def __init__(self, children: List[BMGNode]):
        BMGNode.__init__(self, children)


class BinaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    edges = ["left", "right"]
    operator_type: OperatorType

    def __init__(self, left: BMGNode, right: BMGNode):
        OperatorNode.__init__(self, [left, right])

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

    def label(self) -> str:
        return "+"

    def __str__(self) -> str:
        return "(" + str(self.left()) + "+" + str(self.right()) + ")"


class MultiplicationNode(BinaryOperatorNode):
    operator_type = OperatorType.MULTIPLY

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def label(self) -> str:
        return "*"

    def __str__(self) -> str:
        return "(" + str(self.left()) + "*" + str(self.right()) + ")"


class UnaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    edges = ["operand"]
    operator_type: OperatorType

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

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

    def __str__(self) -> str:
        return "-" + str(self.operand())


class ToRealNode(UnaryOperatorNode):
    operator_type = OperatorType.TO_REAL

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def label(self) -> str:
        return "ToReal"

    def __str__(self) -> str:
        return "ToReal(" + str(self.operand()) + ")"


class ExpNode(UnaryOperatorNode):
    operator_type = OperatorType.EXP

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def label(self) -> str:
        return "Exp"

    def __str__(self) -> str:
        return "Exp(" + str(self.operand()) + ")"


class SampleNode(UnaryOperatorNode):
    operator_type = OperatorType.SAMPLE

    def __init__(self, operand: DistributionNode):
        UnaryOperatorNode.__init__(self, operand)

    def operand(self) -> DistributionNode:
        c = self.children[0]
        assert isinstance(c, DistributionNode)
        return c

    def label(self) -> str:
        return "Sample"

    def __str__(self) -> str:
        return "Sample(" + str(self.operand()) + ")"


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


class Query(BMGNode):
    edges = ["operator"]

    def __init__(self, operator: OperatorNode):
        BMGNode.__init__(self, [operator])

    def operator(self) -> OperatorNode:
        c = self.children[0]
        assert isinstance(c, OperatorNode)
        return c

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


class BMGraphBuilder:

    # Note that Python 3.7 guarantees that dictionaries maintain insertion order.
    nodes: Dict[BMGNode, int]

    def __init__(self):
        self.nodes = {}

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

    def add_real(self, value: float) -> RealNode:
        node = RealNode(value)
        self.add_node(node)
        return node

    def add_boolean(self, value: bool) -> BooleanNode:
        node = BooleanNode(value)
        self.add_node(node)
        return node

    def add_tensor(self, value: Tensor) -> TensorNode:
        node = TensorNode(value)
        self.add_node(node)
        return node

    def add_bernoulli(self, probability: BMGNode) -> BernoulliNode:
        node = BernoulliNode(probability)
        self.add_node(node)
        return node

    def add_addition(self, left: BMGNode, right: BMGNode) -> AdditionNode:
        node = AdditionNode(left, right)
        self.add_node(node)
        return node

    def add_multiplication(self, left: BMGNode, right: BMGNode) -> MultiplicationNode:
        node = MultiplicationNode(left, right)
        self.add_node(node)
        return node

    def add_negate(self, operand: BMGNode) -> NegateNode:
        node = NegateNode(operand)
        self.add_node(node)
        return node

    def add_to_real(self, operand: BMGNode) -> ToRealNode:
        node = ToRealNode(operand)
        self.add_node(node)
        return node

    def add_exp(self, operand: BMGNode) -> ExpNode:
        node = ExpNode(operand)
        self.add_node(node)
        return node

    def add_sample(self, operand: DistributionNode) -> SampleNode:
        node = SampleNode(operand)
        self.add_node(node)
        return node

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
