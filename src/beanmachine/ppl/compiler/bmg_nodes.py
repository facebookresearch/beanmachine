# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, ABCMeta
from typing import Any, Iterable, List

import beanmachine.ppl.compiler.bmg_types as bt
import torch
from beanmachine.ppl.utils.item_counter import ItemCounter
from torch import Tensor


# Note that we're not going to subclass list or UserList here because we
# only need to use the most basic list operations: initialization, getting
# an item, and setting an item. We never want to delete items, append to
# the end, and so on.


class InputList:
    node: "BMGNode"
    inputs: List["BMGNode"]

    def __init__(self, node: "BMGNode", inputs: List["BMGNode"]) -> None:
        assert isinstance(inputs, list)
        self.node = node
        self.inputs = inputs
        for i in inputs:
            i.outputs.add_item(node)

    def __setitem__(self, index: int, value: "BMGNode") -> None:
        # If this is a no-op, do nothing.
        old_value = self.inputs[index]
        if old_value is value:
            return

        # Start by maintaining correctness of the input/output relationships.
        #
        # (1) The node is no longer an output of the current input at the index.
        # (2) The node is now an output of the new input at the index.
        #
        old_value.outputs.remove_item(self.node)
        self.inputs[index] = value
        value.outputs.add_item(self.node)

    def __getitem__(self, index: int) -> "BMGNode":
        return self.inputs[index]

    def __iter__(self):
        return iter(self.inputs)

    def __len__(self) -> int:
        return len(self.inputs)


class BMGNode(ABC):
    """The base class for all graph nodes."""

    # A Bayesian network is a acyclic graph in which each node represents
    # a value or operation; directed edges represent the inputs and
    # outputs of each node.
    #
    # We have a small nomenclature problem here; when describing the shape
    # of, say, a multiplication in an abstract syntax tree we would say that
    # the multiplication operator is the "parent" and the pair of operands
    # are the left and right "children".  However, in Bayesian networks
    # the tradition is to consider the input values as "parents" of the
    # multiplication, and nodes which consume the product are its "children".
    #
    # To avoid this confusion, in this class we will explicitly call out
    # that the edges represent inputs.

    inputs: InputList
    outputs: ItemCounter

    def __init__(self, inputs: List["BMGNode"]):
        assert isinstance(inputs, list)
        self.inputs = InputList(self, inputs)
        self.outputs = ItemCounter()

    @property
    def is_leaf(self) -> bool:
        return len(self.outputs.items) == 0


# ####
# #### Nodes representing constant values
# ####


class ConstantNode(BMGNode, metaclass=ABCMeta):
    """This is the base type for all nodes representing constants.
    Note that every constant node has an associated type in the
    BMG type system; nodes that represent the "real" 1.0,
    the "positive real" 1.0, the "probability" 1.0 and the
    "natural" 1 are all different nodes and are NOT deduplicated."""

    value: Any

    def __init__(self):
        BMGNode.__init__(self, [])


class UntypedConstantNode(ConstantNode):
    def __init__(self, value: Any) -> None:
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class BooleanNode(ConstantNode):
    """A Boolean constant"""

    value: bool

    def __init__(self, value: bool):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class NaturalNode(ConstantNode):
    """An integer constant restricted to non-negative values"""

    value: int

    def __init__(self, value: int):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class PositiveRealNode(ConstantNode):
    """A real constant restricted to non-negative values"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class NegativeRealNode(ConstantNode):
    """A real constant restricted to non-positive values"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class ProbabilityNode(ConstantNode):
    """A real constant restricted to values from 0.0 to 1.0"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class RealNode(ConstantNode):
    """An unrestricted real constant"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class ConstantTensorNode(ConstantNode):
    """A tensor constant"""

    value: Tensor

    def __init__(self, value: Tensor):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)


class ConstantPositiveRealMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class ConstantRealMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class ConstantNegativeRealMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class ConstantProbabilityMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class ConstantSimplexMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class ConstantNaturalMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class ConstantBooleanMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)


class TensorNode(BMGNode):
    """A tensor whose elements are graph nodes."""

    _size: torch.Size

    def __init__(self, items: List[BMGNode], size: torch.Size):
        assert isinstance(items, list)
        self._size = size
        BMGNode.__init__(self, items)

    def __str__(self) -> str:
        return "TensorNode"


# ####
# #### Nodes representing distributions
# ####


class DistributionNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all nodes that represent
    probability distributions."""

    def __init__(self, inputs: List[BMGNode]):
        BMGNode.__init__(self, inputs)


class BernoulliBase(DistributionNode):
    def __init__(self, probability: BMGNode):
        DistributionNode.__init__(self, [probability])

    @property
    def probability(self) -> BMGNode:
        return self.inputs[0]


class BernoulliNode(BernoulliBase):
    """The Bernoulli distribution is a coin flip; it takes
    a probability and each sample is either 0.0 or 1.0."""

    def __init__(self, probability: BMGNode):
        BernoulliBase.__init__(self, probability)

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability) + ")"


class BernoulliLogitNode(BernoulliBase):
    """The Bernoulli distribution is a coin flip; it takes
    a probability and each sample is either 0.0 or 1.0."""

    def __init__(self, probability: BMGNode):
        BernoulliBase.__init__(self, probability)

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability) + ")"


class BetaNode(DistributionNode):
    """The beta distribution samples are values between 0.0 and 1.0, and
    so is useful for creating probabilities."""

    def __init__(self, alpha: BMGNode, beta: BMGNode):
        DistributionNode.__init__(self, [alpha, beta])

    @property
    def alpha(self) -> BMGNode:
        return self.inputs[0]

    @property
    def beta(self) -> BMGNode:
        return self.inputs[1]

    def __str__(self) -> str:
        return f"Beta({str(self.alpha)},{str(self.beta)})"


class PoissonNode(DistributionNode):
    """The Poisson distribution samples are non-negative integer valued."""

    def __init__(self, rate: BMGNode):
        DistributionNode.__init__(self, [rate])

    @property
    def rate(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return f"Poisson({str(self.rate)})"


class BinomialNodeBase(DistributionNode):
    def __init__(self, count: BMGNode, probability: BMGNode):
        DistributionNode.__init__(self, [count, probability])

    @property
    def count(self) -> BMGNode:
        return self.inputs[0]

    @property
    def probability(self) -> BMGNode:
        return self.inputs[1]

    def __str__(self) -> str:
        return f"Binomial({self.count}, {self.probability})"

    def support(self) -> Iterable[Any]:
        raise ValueError("Support of binomial is not yet implemented.")


class BinomialNode(BinomialNodeBase):
    """The Binomial distribution is the extension of the
    Bernoulli distribution to multiple flips. The input
    is the count of flips and the probability of each
    coming up heads; each sample is the number of heads
    after "count" flips."""

    def __init__(self, count: BMGNode, probability: BMGNode, is_logits: bool = False):
        BinomialNodeBase.__init__(self, count, probability)


class BinomialLogitNode(BinomialNodeBase):
    """The Binomial distribution is the extension of the
    Bernoulli distribution to multiple flips. The input
    is the count of flips and the probability of each
    coming up heads; each sample is the number of heads
    after "count" flips."""

    # TODO: We do not yet have a BMG node for Binomial
    # with logits. When we do, add support for it.

    def __init__(self, count: BMGNode, probability: BMGNode):
        BinomialNodeBase.__init__(self, count, probability)


class CategoricalNodeBase(DistributionNode):
    """The categorical distribution is the extension of the
    Bernoulli distribution to multiple outcomes; rather
    than flipping an unfair coin, this is rolling an unfair
    n-sided die.

    The input is the probability of each of n possible outcomes,
    and each sample is drawn from 0, 1, 2, ... n-1."""

    # TODO: we may wish to add bounded integers to the BMG type system.

    def __init__(self, probability: BMGNode):
        DistributionNode.__init__(self, [probability])

    @property
    def probability(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return "Categorical(" + str(self.probability) + ")"


class CategoricalNode(CategoricalNodeBase):
    def __init__(self, probability: BMGNode):
        DistributionNode.__init__(self, [probability])


class CategoricalLogitNode(CategoricalNodeBase):
    def __init__(self, probability: BMGNode):
        DistributionNode.__init__(self, [probability])


class Chi2Node(DistributionNode):
    """The chi2 distribution is a distribution of positive
    real numbers; it is a special case of the gamma distribution."""

    def __init__(self, df: BMGNode):
        DistributionNode.__init__(self, [df])

    @property
    def df(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return f"Chi2({str(self.df)})"


class DirichletNode(DistributionNode):
    """The Dirichlet distribution generates simplexs -- vectors
    whose members are probabilities that add to 1.0, and
    so it is useful for generating inputs to the categorical
    distribution."""

    def __init__(self, concentration: BMGNode):
        DistributionNode.__init__(self, [concentration])

    @property
    def concentration(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return f"Dirichlet({str(self.concentration)})"


class FlatNode(DistributionNode):

    """The Flat distribution the standard uniform distribution from 0.0 to 1.0."""

    def __init__(self):
        DistributionNode.__init__(self, [])

    def __str__(self) -> str:
        return "Flat()"


class GammaNode(DistributionNode):
    """The gamma distribution is a distribution of positive
    real numbers characterized by positive real concentration and rate
    parameters."""

    def __init__(self, concentration: BMGNode, rate: BMGNode):
        DistributionNode.__init__(self, [concentration, rate])

    @property
    def concentration(self) -> BMGNode:
        return self.inputs[0]

    @property
    def rate(self) -> BMGNode:
        return self.inputs[1]

    def __str__(self) -> str:
        return f"Gamma({str(self.concentration)}, {str(self.rate)})"


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

    def __init__(self, scale: BMGNode):
        DistributionNode.__init__(self, [scale])

    @property
    def scale(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return f"HalfCauchy({str(self.scale)})"


class NormalNode(DistributionNode):

    """The normal (or "Gaussian") distribution is a bell curve with
    a given mean and standard deviation."""

    def __init__(self, mu: BMGNode, sigma: BMGNode):
        DistributionNode.__init__(self, [mu, sigma])

    @property
    def mu(self) -> BMGNode:
        return self.inputs[0]

    @property
    def sigma(self) -> BMGNode:
        return self.inputs[1]

    def __str__(self) -> str:
        return f"Normal({str(self.mu)},{str(self.sigma)})"


class HalfNormalNode(DistributionNode):

    """The half-normal distribution is a half bell curve with
    a given standard deviation. Mean (for the underlying normal)
    is taken to be zero."""

    def __init__(self, sigma: BMGNode):
        DistributionNode.__init__(self, [sigma])

    @property
    def sigma(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return f"HalfNormal({str(self.sigma)})"


class StudentTNode(DistributionNode):
    """The Student T distribution is a bell curve with zero mean
    and a heavier tail than the normal distribution. It is
    useful in statistical analysis because a common situation
    is to have observations of a normal process but to not
    know the true mean. Samples from the T distribution can
    be used to represent the difference between an observed mean
    and the true mean."""

    def __init__(self, df: BMGNode, loc: BMGNode, scale: BMGNode):
        DistributionNode.__init__(self, [df, loc, scale])

    @property
    def df(self) -> BMGNode:
        return self.inputs[0]

    @property
    def loc(self) -> BMGNode:
        return self.inputs[1]

    @property
    def scale(self) -> BMGNode:
        return self.inputs[2]

    def __str__(self) -> str:
        return f"StudentT({str(self.df)},{str(self.loc)},{str(self.scale)})"


class UniformNode(DistributionNode):

    """The Uniform distribution is a "flat" distribution of values
    between 0.0 and 1.0."""

    # TODO: We do not yet have an implementation of the uniform
    # distribution as a BMG node. When we do, implement the
    # feature here.

    def __init__(self, low: BMGNode, high: BMGNode):
        DistributionNode.__init__(self, [low, high])

    @property
    def low(self) -> BMGNode:
        return self.inputs[0]

    @property
    def high(self) -> BMGNode:
        return self.inputs[1]

    def __str__(self) -> str:
        return f"Uniform({str(self.low)},{str(self.high)})"


# ####
# #### Operators
# ####


class OperatorNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all operators.
    The inputs are the operands of each operator."""

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        BMGNode.__init__(self, inputs)


# ####
# #### Multiary operators
# ####


class AdditionNode(OperatorNode):
    """This represents an addition of values."""

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        OperatorNode.__init__(self, inputs)

    def __str__(self) -> str:
        return "(" + "+".join([str(inp) for inp in self.inputs]) + ")"


class MultiplicationNode(OperatorNode):
    """This represents multiplication of values."""

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        OperatorNode.__init__(self, inputs)

    def __str__(self) -> str:
        return "(" + "*".join([str(inp) for inp in self.inputs]) + ")"


# We have three kinds of logsumexp nodes.
#
# * LogSumExpTorchNode represents a call to logsumexp in the original
#   Python model. It has three operands: the tensor being summed,
#   the dimension along which it is summed, and a flag giving the shape.
#
# * LogSumExpNode represents a BMG LOGSUMEXP node. It is an n-ary operator
#   and produces a real; each of the inputs is one of the summands.
#
# * LogSumExpVectorNode represents a BMG LOGSUMEXP_VECTOR node. It is a unary
#   operator that takes a single-column matrix.
#
# We transform LogSumExpTorchNode into LogSumExpNode or LogSumExpVectorNode
# as appropriate.


class LogSumExpTorchNode(OperatorNode):
    def __init__(self, operand: BMGNode, dim: BMGNode, keepdim: BMGNode):
        OperatorNode.__init__(self, [operand, dim, keepdim])

    def __str__(self) -> str:
        return "LogSumExp"


class LogSumExpNode(OperatorNode):
    """This class represents the LogSumExp operation: for values v_1, ..., v_n
    we compute log(exp(v_1) + ... + exp(v_n))"""

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        OperatorNode.__init__(self, inputs)

    def __str__(self) -> str:
        return "LogSumExp"


class ToMatrixNode(OperatorNode):
    """A 2-d tensor whose elements are graph nodes."""

    def __init__(self, rows: NaturalNode, columns: NaturalNode, items: List[BMGNode]):
        # The first two elements are the row and column counts; they must
        # be constant naturals.
        assert isinstance(items, list)
        assert len(items) >= 1
        rc: List[BMGNode] = [rows, columns]
        BMGNode.__init__(self, rc + items)

    @property
    def rows(self) -> NaturalNode:
        return self.inputs[0]  # pyre-ignore

    @property
    def columns(self) -> NaturalNode:
        return self.inputs[1]  # pyre-ignore

    def __str__(self) -> str:
        return "ToMatrix"


# ####
# #### Control flow operators
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

    def __init__(self, condition: BMGNode, consequence: BMGNode, alternative: BMGNode):
        OperatorNode.__init__(self, [condition, consequence, alternative])

    @property
    def condition(self) -> BMGNode:
        return self.inputs[0]

    @property
    def consequence(self) -> BMGNode:
        return self.inputs[1]

    @property
    def alternative(self) -> BMGNode:
        return self.inputs[2]

    def __str__(self) -> str:
        i = str(self.condition)
        t = str(self.consequence)
        e = str(self.alternative)
        return f"(if {i} then {t} else {e})"


class ChoiceNode(OperatorNode):
    """This class represents a stochastic choice between n options, where
    the condition is a natural."""

    # See comments in SwitchNode for more details.

    def __init__(self, condition: BMGNode, items: List[BMGNode]):
        assert isinstance(items, list)
        # We should not generate a choice node if there is only one choice.
        assert len(items) >= 2
        c: List[BMGNode] = [condition]
        BMGNode.__init__(self, c + items)

    def __str__(self) -> str:
        return "Choice"


# ####
# #### Binary operators
# ####


class BinaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    """This is the base class for all binary operators."""

    def __init__(self, left: BMGNode, right: BMGNode):
        OperatorNode.__init__(self, [left, right])

    @property
    def left(self) -> BMGNode:
        return self.inputs[0]

    @property
    def right(self) -> BMGNode:
        return self.inputs[1]


class ComparisonNode(BinaryOperatorNode, metaclass=ABCMeta):
    """This is the base class for all comparison operators."""

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class GreaterThanNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def __str__(self) -> str:
        return f"({str(self.left)}>{str(self.right)})"


class GreaterThanEqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def __str__(self) -> str:
        return f"({str(self.left)}>={str(self.right)})"


class LessThanNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def __str__(self) -> str:
        return f"({str(self.left)}<{str(self.right)})"


class LessThanEqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def __str__(self) -> str:
        return f"({str(self.left)}<={str(self.right)})"


class EqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def __str__(self) -> str:
        return f"({str(self.left)}=={str(self.right)})"


class NotEqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def __str__(self) -> str:
        return f"({str(self.left)}!={str(self.right)})"


class IsNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)


class IsNotNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)


class InNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)


class NotInNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)


class BitAndNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class BitOrNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class BitXorNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class DivisionNode(BinaryOperatorNode):
    """This represents a division."""

    # There is no division node in BMG; we will replace
    # x / y with x * (y ** (-1)) during the "fix problems"
    # phase.

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return "(" + str(self.left) + "/" + str(self.right) + ")"


class FloorDivNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class LShiftNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class ModNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class RShiftNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)


class SwitchNode(BMGNode):

    """This class represents a point in a program where there are
    multiple control flows based on the value of a stochastic node."""

    # For example, suppose we have this contrived model:
    #
    #   @bm.random_variable def weird(i):
    #     if i == 0:
    #       return Normal(3.0, 4.0)
    #     return Normal(5.0, 6.0)
    #
    #   @bm.random_variable def flips():
    #     return Binomial(2, 0.5)
    #
    #   @bm.random_variable def really_weird():
    #     return Normal(weird(flips()), 7.0)
    #
    # There are three possibilities for weird(flips()) on the last line;
    # what we need to represent in the graph is:
    #
    # * sample once from Normal(0.0, 1.0), call this weird(0)
    # * sample twice from Normal(1.0, 1.0), call these weird(1) and weird(2)
    # * sample once from flips()
    # * choose one of weird(i) based on the sample from flips().
    #
    # We represent this with a switch node.
    #
    #   2  0.5 3   4     5   6
    #    \ /    \ /       \ /
    #     B      N         N
    #     |      |       /    \
    #     ~  0   ~    1  ~  2  ~
    #      \  \   \  /  /  /  /
    #            switch
    #                 \   7
    #                  \ /
    #                   N
    #                   |
    #                   ~
    #
    # That is, inputs[0] of the switch is the quantity that makes the choice:
    # a sample from B(2, 0.5).  We then have a series of case/value pairs:
    #
    # * inputs[c] for c = 1, 3, 5, ... are always constants.
    # * inputs[v] for v = 2, 4, 6, ... are the values chosen when inputs[0]
    #   takes on the value of the corresponding constant.
    #
    # Note that we do not have a generalized switch in BMG. Rather, we have
    # the simpler cases of (1) the IfThenElse node, where the leftmost input
    # is a Boolean quantity and the other two inputs are the values, and
    # (2) a ChoiceNode, which takes a natural and then chooses from amongst
    # n possible values.
    #
    # TODO: Should we implement a general switch node in BMG?
    #
    # The runtime creates the switch based on the support of flips(), the first
    # input to the switch. In this case the support is {0, 1, 2} but there is
    # no reason why they could not have been 1, 10, 100 instead, if for instance
    # we had something like "weird(10 ** flips())".

    def __init__(self, inputs: List[BMGNode]):
        # TODO: Check that cases are all constant nodes.
        # TODO: Check that there is one value for each case.
        BMGNode.__init__(self, inputs)


# This represents an indexing operation in the original source code.
# It will be replaced by a VectorIndexNode or ColumnIndexNode in the
# problem fixing phase.
class IndexNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return str(self.left) + "[" + str(self.right) + "]"


class ItemNode(OperatorNode):
    """Represents torch.Tensor.item() conversion from tensor to scalar."""

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

    def __str__(self) -> str:
        return str(self.inputs[0]) + ".item()"


class VectorIndexNode(BinaryOperatorNode):
    """This represents a stochastic index into a vector. The left operand
    is the vector and the right operand is the index."""

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return str(self.left) + "[" + str(self.right) + "]"


class ColumnIndexNode(BinaryOperatorNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return "ColumnIndex"


class MatrixMultiplicationNode(BinaryOperatorNode):
    """This represents a matrix multiplication."""

    # TODO: We now have matrix multiplication in BMG; finish this implementation

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return "(" + str(self.left) + "*" + str(self.right) + ")"


class MatrixScaleNode(BinaryOperatorNode):
    """This represents a matrix scaling."""

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return "(" + str(self.left) + "*" + str(self.right) + ")"


class PowerNode(BinaryOperatorNode):
    """This represents an x-to-the-y operation."""

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def __str__(self) -> str:
        return "(" + str(self.left) + "**" + str(self.right) + ")"


# ####
# #### Unary operators
# ####


class UnaryOperatorNode(OperatorNode, metaclass=ABCMeta):
    """This is the base type of unary operator nodes."""

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

    @property
    def operand(self) -> BMGNode:
        return self.inputs[0]


class ExpNode(UnaryOperatorNode):
    """This represents an exponentiation operation; it is generated when
    a model contains calls to Tensor.exp or math.exp."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "Exp(" + str(self.operand) + ")"


class ExpM1Node(UnaryOperatorNode):
    """This represents the operation exp(x) - 1; it is generated when
    a model contains calls to Tensor.expm1."""

    # TODO: If we have exp(x) - 1 in the graph and x is known to be of type
    # positive real or negative real then the expression as a whole is of
    # type real. If we convert such expressions in the graph to expm1(x)
    # then we can make the type more specific, and also possibly reduce
    # the number of nodes in the graph.

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ExpM1(" + str(self.operand) + ")"


class LogisticNode(UnaryOperatorNode):
    """This represents the operation 1/(1+exp(x)); it is generated when
    a model contains calls to Tensor.sigmoid."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "Logistic(" + str(self.operand) + ")"


class LogNode(UnaryOperatorNode):
    """This represents a log operation; it is generated when
    a model contains calls to Tensor.log or math.log."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "Log(" + str(self.operand) + ")"


# TODO: replace "log" with "log1mexp" as needed below and update defs


class Log1mexpNode(UnaryOperatorNode):
    """This represents a log1mexp operation; it is generated when
    a model contains calls to log1mexp or math_log1mexp."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "Log1mexp(" + str(self.operand) + ")"


# BMG supports three different kinds of negation:

# * The "complement" node with a Boolean operand has the semantics
#   of logical negation.  The input and output are both bool.
#
# * The "complement" node with a probability operand has the semantics
#   of (1 - p). The input and output are both probability.
#
# * The "negate" node has the semantics of (0 - x). The input must be
#   real, positive real or negative real, and the output is
#   real, negative real or positive real respectively.
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
#       and replace them with "complement" nodes (where p is a probability or
#       Boolean expression).
#
#   (4) Other usages of binary + and unary - in the Python model will
#       be converted to BMG following the rules for addition and negation
#       in BMG: negation must be real valued, and so on.


class NegateNode(UnaryOperatorNode):

    """This represents a unary minus."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "-" + str(self.operand)


class NotNode(UnaryOperatorNode):
    """This represents a logical not that appears in the Python model."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "not " + str(self.operand)


class ComplementNode(UnaryOperatorNode):
    """This represents a complement of a Boolean or probability
    value."""

    # See notes above NegateNode for details

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "complement " + str(self.operand)


# This operator is not supported in BMG.  We accumulate it into
# the graph in order to produce a good error message.
class InvertNode(UnaryOperatorNode):
    """This represents a bit inversion (~)."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "~" + str(self.operand)


class PhiNode(UnaryOperatorNode):
    """This represents a phi operation; that is, the cumulative
    distribution function of the standard normal."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "Phi(" + str(self.operand) + ")"


class SampleNode(UnaryOperatorNode):
    """This represents a single unique sample from a distribution;
    if a graph has two sample nodes both taking input from the same
    distribution, each sample is logically distinct. But if a graph
    has two nodes that both input from the same sample node, we must
    treat those two uses of the sample as though they had identical
    values."""

    def __init__(self, operand: DistributionNode):
        UnaryOperatorNode.__init__(self, operand)

    @property
    def operand(self) -> DistributionNode:
        c = self.inputs[0]
        assert isinstance(c, DistributionNode)
        return c

    def __str__(self) -> str:
        return "Sample(" + str(self.operand) + ")"


class ToRealNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToReal(" + str(self.operand) + ")"


class ToIntNode(UnaryOperatorNode):
    """This represents an integer truncation operation; it is generated
    when a model contains calls to Tensor.int() or int()."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToInt(" + str(self.operand) + ")"


class ToRealMatrixNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToRealMatrix(" + str(self.operand) + ")"


class ToPositiveRealMatrixNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToPosRealMatrix(" + str(self.operand) + ")"


class ToPositiveRealNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToPosReal(" + str(self.operand) + ")"


class ToProbabilityNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToProb(" + str(self.operand) + ")"


class ToNegativeRealNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "ToNegReal(" + str(self.operand) + ")"


class LogSumExpVectorNode(UnaryOperatorNode):
    # BMG supports a log-sum-exp operator that takes a one-column tensor.
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def __str__(self) -> str:
        return "LogSumExpVector"


# ####
# #### Marker nodes
# ####


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
    # models.) Consider how we might extract the graph from a model
    # without knowing the observations ahead of time.
    value: Any

    def __init__(self, observed: BMGNode, value: Any):
        # The observed node is required to be a sample by BMG,
        # but during model transformations it is possible for
        # an observation to temporarily observe a non-sample.
        # TODO: Consider adding a verification pass which ensures
        # this invariant is maintained by the rewriters.
        self.value = value
        BMGNode.__init__(self, [observed])

    @property
    def observed(self) -> BMGNode:
        return self.inputs[0]

    def __str__(self) -> str:
        return str(self.observed) + "=" + str(self.value)


class Query(BMGNode):
    """A query is a marker on a node in the graph that indicates
    to the inference engine that the user is interested in
    getting a distribution of values of that node."""

    # TODO: BMG requires that the target of a query be classified
    # as an operator and that queries be unique; that is, every node
    # is queried *exactly* zero or one times. Rather than making
    # those restrictions here, instead detect bad queries in the
    # problem fixing phase and report accordingly.

    # TODO: As with observations, properly speaking there is no
    # need to represent a query as a *node*, and BMG does not
    # do so. We might wish to follow this pattern as well.

    def __init__(self, operator: BMGNode):
        BMGNode.__init__(self, [operator])

    @property
    def operator(self) -> BMGNode:
        c = self.inputs[0]
        return c

    def __str__(self) -> str:
        return "Query(" + str(self.operator) + ")"


# The basic idea of the Metropolis algorithm is: each possible state of
# the graph is assigned a "score" proportional to the probability density
# of that state. We do not know the proportionality constant but we do not
# need to because we take the ratio of the current state's score to a proposed
# new state's score, and accept or reject the proposal based on the ratio.
#
# The idea of a "factor" node is that we also multiply the score by a real number
# which is high for "more likely" states and low for "less likely" states. By
# carefully choosing a factor function we can express our additional knowledge of
# the model.
#
# Factors (like observations and queries) are never used as inputs even though they
# compute a value.


class FactorNode(BMGNode, metaclass=ABCMeta):
    """This is the base class for all factors.
    The inputs are the operands of each factor."""

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        BMGNode.__init__(self, inputs)


# The ExpProduct factor takes one or more inputs, computes their product,
# and then multiplies the score by exp(product), so if the product is large
# then the factor will be very large; if the product is zero then the factor
# will be one, and if the product is negative then the factor will be small.


class ExpProductFactorNode(FactorNode):
    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        FactorNode.__init__(self, inputs)

    def __str__(self) -> str:
        return "ExpProduct"


def is_zero(n: BMGNode) -> bool:
    return isinstance(n, ConstantNode) and bt.is_zero(n.value)


def is_one(n: BMGNode) -> bool:
    return isinstance(n, ConstantNode) and bt.is_one(n.value)
