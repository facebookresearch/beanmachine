# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import functools
import itertools
import operator
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Iterator, List

import torch
from beanmachine.ppl.compiler.bmg_types import (
    AnyRequirement,
    BMGLatticeType,
    BMGMatrixType,
    Boolean,
    Malformed,
    Natural,
    NegativeReal,
    One,
    OneHotMatrix,
    PositiveReal,
    Probability,
    Real,
    Requirement,
    SimplexMatrix,
    Tensor as BMGTensor,
    ZeroMatrix,
    always_matrix,
    supremum,
    type_of_value,
    upper_bound,
)
from beanmachine.ppl.utils.item_counter import ItemCounter
from torch import Tensor, tensor
from torch.distributions import Normal
from torch.distributions.utils import broadcast_all


# TODO: There are numerous pyre errors in this module; fix them.
# pyre-ignore-all-errors

# TODO: Extract inf type, graph type and requirements computations to own module
# TODO: Add assertions which ensure that type requirements are never "fake" types
# like Zero or OneHot.  Also assert that a requirement is never Malformed.


def prod(x):
    """Compute the product of a sequence of values of arbitrary length"""
    return functools.reduce(operator.mul, x, 1)


positive_infinity = float("inf")


def _recompute_types(node: "BMGNode") -> None:

    # If a node is mutated -- say, by changing one of its inputs -- then
    # its type might change. This can then cause the types of its outputs
    # to change, and the change can thereby propagate through the graph.
    # The path along which that change propagates can be arbitrarily long in
    # large graphs, which means that we can exceed Python's recursion limit.
    # We therefore need this algorithm to be iterative, not recursive.
    #
    # We depend on several invariants for this algorithm to be correct.
    # First, of course, we require that the graph is acyclic. We also
    # require that the types of all inputs are already correct. Given
    # those invariants, what we can do when we believe that the type of
    # a node might be wrong is: recompute the type of this node; if the
    # new type is the same as the old type, we're done. If not, then
    # we set the type of this node to the correct type and then propagate
    # that change to its outputs, which might then need to recompute their
    # type information in turn.
    #

    work = [node]

    while len(work) != 0:
        current = work.pop()
        it = current._compute_inf_type()
        gt = current._compute_graph_type()
        changed = it != current._inf_type or gt != current._graph_type
        if changed:
            current._inf_type = it
            current._graph_type = gt
            # Types are now correct and cached; we can propagate that
            # change if necessary to our outputs.
            for o in current.outputs.items:
                work.append(o)


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

        # Some additional invariants that we must maintain for performance reasons are:
        #
        # (3) Every node caches its own graph and inf types.
        # (4) The types of the inputs to a node are always correct.
        #
        # This means that when the graph mutates such that an input of a node
        # changes, we need to recompute the type of the node so that it stays
        # correct. That could in turn cause the type of the node's outputs to
        # change; the call to _recompute_types propagates the change to all
        # descendant outputs.
        _recompute_types(self.node)

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

    # See comments in InputList above for invariants we maintain on these members.
    _inf_type: BMGLatticeType
    _graph_type: BMGLatticeType

    def __init__(self, inputs: List["BMGNode"]):
        assert isinstance(inputs, list)
        self.inputs = InputList(self, inputs)
        self.outputs = ItemCounter()

        # The inf type and graph type of a node may depend upon the
        # type of the nodes inputs. However we cannot compute them
        # lazily using a standard recursive algorithm, because that
        # could do a recursive traversal of the entire graph. If the
        # graph has a path longer than the Python recursion limit
        # then a recursive algorithm can crash. We therefore maintain
        # the invariant that types of inputs are already known, and
        # types of their outputs are computed on construction.
        #
        # See notes above for how types are recomputed upon a graph mutation.
        #
        # Note: We're doing virtual calls from inside a base class
        # constructor, which requires some care. We require that the
        # derived types have already set up all the state they need
        # to successfully make these calls at this time!

        # TODO: We will probably have to do the same thing for
        # computing the tensor shape, since at present that algorithm
        # is recursive.

        self._inf_type = self._compute_inf_type()
        self._graph_type = self._compute_graph_type()

    @abstractmethod
    def _compute_graph_type(self) -> BMGLatticeType:
        """The type of the node in the graph type system."""
        pass

    @property
    def graph_type(self) -> BMGLatticeType:
        """The type of the node in the graph type system."""
        return self._graph_type

    @abstractmethod
    def _compute_inf_type(self) -> BMGLatticeType:
        pass

    @property
    def inf_type(self) -> BMGLatticeType:
        """BMG nodes have type requirements on their inputs; the *infimum type* of
        a node is the *smallest* BMG type that a node may be converted to if required by
        an input."""
        return self._inf_type

    @property
    def is_matrix(self) -> BMGLatticeType:
        """Is this node classified as a matrix in BMG?"""
        return False

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

    # Many of the node types defined in this module have no direct counterparts
    # in BMG; they are generated during execution of the model and accumulated
    # into a graph builder. They must then be transformed into semantically-
    # equivalent supported nodes.

    def _supported_in_bmg(self) -> bool:
        return False

    @abstractmethod
    def support(self) -> Iterator[Any]:
        """To build the graph of all possible control flows through
        the model we need to know for any given node what are
        all the possible values it could attain; we require that
        the set be finite and will throw an exception if it is not."""
        pass

    def support_size(self) -> float:
        # It can be expensive to construct the support if it is large
        # and we might wish to merely know how big it is.  By default
        # assume that every node has infinite support and override this
        # in nodes which have smaller support.
        #
        # Note that this is the *approximate* support size. For example,
        # if we have a Boolean node then its support size is two. If we
        # have the sum of two distinct Boolean nodes then the true size
        # of the support of the sum node is 3 because the result will be
        # 0, 1 or 2.  But we assume that there are two possibilities on
        # the left, two on the right, so four possible outcomes. We can
        # therefore over-estimate; we should however not under-estimate.
        return positive_infinity


# When constructing the support of various nodes we are often
# having to remove duplicates from a set of possible values.
# Unfortunately, it is not easy to do so with torch tensors.
# This helper class implements a set of tensors.

# TODO: Move this to its own module.


class SetOfTensors(collections.abc.Set):
    """Tensors cannot be put into a normal set because tensors that compare as
    equal do not hash to equal hashes. This is a linear-time set implementation.
    Most of the time the sets will be very small."""

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

    value: Any

    def __init__(self):
        BMGNode.__init__(self, [])

    def _compute_inf_type(self) -> BMGLatticeType:
        # The infimum type of a constant is derived from the value,
        # not from the kind of constant node we have. For instance,
        # a NaturalNode containing zero and a ConstantTensorNode containing
        # tensor([[[0.0]]]) both have infimum type "Boolean" because
        # we can convert zero to False, and Boolean is the smallest type
        # in the lattice. Remember, the infimum type answers the question
        # "what types can this node be converted to?" and not "what is the
        # type of this node?"
        return type_of_value(self.value)

    @property
    def requirements(self) -> List[Requirement]:
        return []

    def _supported_in_bmg(self) -> bool:
        return True

    # The support of a constant is just the value.
    def support(self) -> Iterator[Any]:
        yield self.value

    def support_size(self) -> float:
        return 1.0


class BooleanNode(ConstantNode):
    """A Boolean constant"""

    value: bool

    def __init__(self, value: bool):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Boolean

    @property
    def size(self) -> torch.Size:
        return torch.Size([])


class NaturalNode(ConstantNode):
    """An integer constant restricted to non-negative values"""

    value: int

    def __init__(self, value: int):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Natural

    @property
    def size(self) -> torch.Size:
        return torch.Size([])


class PositiveRealNode(ConstantNode):
    """A real constant restricted to non-negative values"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return PositiveReal

    @property
    def size(self) -> torch.Size:
        return torch.Size([])


class NegativeRealNode(ConstantNode):
    """A real constant restricted to non-positive values"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return NegativeReal

    @property
    def size(self) -> torch.Size:
        return torch.Size([])


class ProbabilityNode(ConstantNode):
    """A real constant restricted to values from 0.0 to 1.0"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Probability

    @property
    def size(self) -> torch.Size:
        return torch.Size([])


class RealNode(ConstantNode):
    """An unrestricted real constant"""

    value: float

    def __init__(self, value: float):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real

    @property
    def size(self) -> torch.Size:
        return torch.Size([])


class ConstantTensorNode(ConstantNode):
    """A tensor constant"""

    value: Tensor

    def __init__(self, value: Tensor):
        self.value = value
        ConstantNode.__init__(self)

    def __str__(self) -> str:
        return str(self.value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return BMGTensor

    @property
    def size(self) -> torch.Size:
        return self.value.size()


class ConstantPositiveRealMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return PositiveReal.with_size(self.size)

    @property
    def is_matrix(self) -> bool:
        return True


class ConstantRealMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real.with_size(self.size)

    @property
    def is_matrix(self) -> bool:
        return True


class ConstantNegativeRealMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return NegativeReal.with_size(self.size)

    @property
    def is_matrix(self) -> bool:
        return True


class ConstantProbabilityMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Probability.with_size(self.size)

    @property
    def is_matrix(self) -> bool:
        return True


class ConstantNaturalMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Natural.with_size(self.size)

    @property
    def is_matrix(self) -> bool:
        return True


class ConstantBooleanMatrixNode(ConstantTensorNode):
    def __init__(self, value: Tensor):
        assert len(value.size()) <= 2
        ConstantTensorNode.__init__(self, value)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Boolean.with_size(self.size)

    @property
    def is_matrix(self) -> bool:
        return True


class TensorNode(BMGNode):
    """A tensor whose elements are graph nodes."""

    _size: torch.Size

    def __init__(self, items: List[BMGNode], size: torch.Size):
        assert isinstance(items, list)
        self._size = size
        BMGNode.__init__(self, items)

    def __str__(self) -> str:
        return "TensorNode"

    def _compute_graph_type(self) -> BMGLatticeType:
        # TODO: When eventually we get a representation of these in BMG, we will
        # need to compute the graph type of this node. When that happens we should
        # impose the invariant that the graph types of each input must be the same;
        # we can have the problem fixer insert conversions on input edges as necessary.
        # Once we have all the input types the same, we can produce a matrix type
        # here based on the size and the graph type of the input elements.
        return BMGTensor

    def _compute_inf_type(self) -> BMGLatticeType:
        # TODO: When eventually we get a representation of these in BMG, we will
        # need to compute the inf type of this node.  When that happens we should
        # first check the size; if it is not one or two dimensional then we can
        # just return BMGTensor. If it is one or two dimensional then we can
        # take the supremum of the inf types of all the inputs, and use that with
        # the size to create the appropriate matrix type. But for now we'll
        # just return tensor and worry about it later.
        return BMGTensor

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: When eventually we get a representation of these in BMG, we will
        # need to make every input required to be the supremum of the inf types of
        # all the inputs. Until then, just make every input required to be whatever
        # it already is.
        return [i.inf_type for i in self.inputs]

    @property
    def size(self) -> torch.Size:
        return self._size

    def support(self) -> Iterator[Any]:
        s = self.size
        return (
            tensor(c).view(s)
            for c in itertools.product(*(i.support() for i in self.inputs))
        )

    def support_size(self) -> float:
        return prod(i.support_size() for i in self.inputs)


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

    @probability.setter
    def probability(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Boolean

    def _compute_inf_type(self) -> BMGLatticeType:
        return Boolean

    @property
    def size(self) -> torch.Size:
        return self.probability.size

    def _supported_in_bmg(self) -> bool:
        return True

    def support(self) -> Iterator[Any]:
        s = self.size
        return (tensor(i).view(s) for i in itertools.product(*([[0.0, 1.0]] * prod(s))))

    def support_size(self) -> float:
        return 2.0 ** prod(self.size)


class BernoulliNode(BernoulliBase):
    """The Bernoulli distribution is a coin flip; it takes
    a probability and each sample is either 0.0 or 1.0."""

    def __init__(self, probability: BMGNode):
        BernoulliBase.__init__(self, probability)

    @property
    def requirements(self) -> List[Requirement]:
        return [Probability]

    def __str__(self) -> str:
        return "Bernoulli(" + str(self.probability) + ")"


class BernoulliLogitNode(BernoulliBase):
    """The Bernoulli distribution is a coin flip; it takes
    a probability and each sample is either 0.0 or 1.0."""

    def __init__(self, probability: BMGNode):
        BernoulliBase.__init__(self, probability)

    @property
    def requirements(self) -> List[Requirement]:
        return [Real]

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

    @alpha.setter
    def alpha(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def beta(self) -> BMGNode:
        return self.inputs[1]

    @beta.setter
    def beta(self, p: BMGNode) -> None:
        self.inputs[1] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Probability

    def _compute_inf_type(self) -> BMGLatticeType:
        return Probability

    @property
    def requirements(self) -> List[Requirement]:
        # Both inputs to a beta must be positive reals
        return [PositiveReal, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.alpha.size

    def __str__(self) -> str:
        return f"Beta({str(self.alpha)},{str(self.beta)})"

    def _supported_in_bmg(self) -> bool:
        return True

    def support(self) -> Iterator[Any]:
        # TODO: Make a better exception type.
        # TODO: Catch this error during graph generation and produce a better
        # TODO: error message that diagnoses the problem more exactly for
        # TODO: the user.  This would happen if we did something like
        # TODO: x(n()) where x() is a sample that takes a finite index but
        # TODO: n() is a sample that returns a beta.
        raise ValueError("Beta distribution does not have finite support.")


class BinomialNodeBase(DistributionNode):
    def __init__(self, count: BMGNode, probability: BMGNode):
        DistributionNode.__init__(self, [count, probability])

    @property
    def count(self) -> BMGNode:
        return self.inputs[0]

    @count.setter
    def count(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def probability(self) -> BMGNode:
        return self.inputs[1]

    @probability.setter
    def probability(self, p: BMGNode) -> None:
        self.inputs[1] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Natural

    def _compute_inf_type(self) -> BMGLatticeType:
        return Natural

    @property
    def size(self) -> torch.Size:
        return broadcast_all(
            torch.zeros(self.count.size), torch.zeros(self.probability.size)
        ).size()

    def __str__(self) -> str:
        return f"Binomial({self.count}, {self.probability})"

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


class BinomialNode(BinomialNodeBase):
    """The Binomial distribution is the extension of the
    Bernoulli distribution to multiple flips. The input
    is the count of flips and the probability of each
    coming up heads; each sample is the number of heads
    after "count" flips."""

    def __init__(self, count: BMGNode, probability: BMGNode, is_logits: bool = False):
        BinomialNodeBase.__init__(self, count, probability)

    @property
    def requirements(self) -> List[Requirement]:
        return [Natural, Probability]

    def _supported_in_bmg(self) -> bool:
        return True


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

    @property
    def requirements(self) -> List[Requirement]:
        return [Natural, Real]

    def _supported_in_bmg(self) -> bool:
        return False


# TODO: Split this into two distributions as with binomial and Bernoulli.
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

    is_logits: bool

    def __init__(self, probability: BMGNode, is_logits: bool = False):
        self.is_logits = is_logits
        DistributionNode.__init__(self, [probability])

    @property
    def probability(self) -> BMGNode:
        return self.inputs[0]

    @probability.setter
    def probability(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Natural

    def _compute_inf_type(self) -> BMGLatticeType:
        return Natural

    @property
    def requirements(self) -> List[Requirement]:
        # The input to a categorical must be a tensor.
        # TODO: We do not yet support categoricals in BMG and when we do,
        # we will likely need to implement a simplex type rather than
        # a tensor. Fix this up when categoricals are implemented.
        return [BMGTensor]

    @property
    def size(self) -> torch.Size:
        return self.probability.size[0:-1]

    def __str__(self) -> str:
        return "Categorical(" + str(self.probability) + ")"

    def support(self) -> Iterator[Any]:
        s = self.probability.size
        r = list(range(s[-1]))
        sr = s[:-1]
        return (tensor(i).view(sr) for i in itertools.product(*([r] * prod(sr))))

    def support_size(self) -> float:
        s = self.probability.size
        return s[-1] ** prod(s[:-1])


class Chi2Node(DistributionNode):
    """The chi2 distribution is a distribution of positive
    real numbers; it is a special case of the gamma distribution."""

    def __init__(self, df: BMGNode):
        DistributionNode.__init__(self, [df])

    @property
    def df(self) -> BMGNode:
        return self.inputs[0]

    @df.setter
    def df(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return PositiveReal

    def _compute_inf_type(self) -> BMGLatticeType:
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        return [PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.df.size

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

    def __init__(self, concentration: BMGNode):
        DistributionNode.__init__(self, [concentration])

    @property
    def concentration(self) -> BMGNode:
        return self.inputs[0]

    @concentration.setter
    def concentration(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return SimplexMatrix(1, self._required_columns)

    def _compute_inf_type(self) -> BMGLatticeType:
        return SimplexMatrix(1, self._required_columns)

    @property
    def _required_columns(self) -> int:
        # The "max" is needed to handle the degenerate case of
        # Dirichlet(tensor([])) -- in this case we will say that we require
        # a single positive real and that the requirement cannot be met.
        size = self.size
        dimensions = len(size)
        return max(1, size[dimensions - 1]) if dimensions > 0 else 1

    @property
    def requirements(self) -> List[Requirement]:
        # BMG's Dirichlet node requires that the input be exactly one
        # vector of positive reals, and the length of the vector is
        # the number of elements in the simplex we produce. We can
        # express that restriction as a positive real matrix with
        # row count equal to 1 and column count equal to the final
        # dimension of the size.
        #
        # A degenerate case here is Dirichlet(tensor([1.0])); we would
        # normally generate the input as a positive real constant, but
        # we require that it be a positive real constant 1x1 *matrix*,
        # which is a different node. The "always matrix" requirement
        # forces the problem fixer to ensure that the input node is
        # always considered by BMG to be a matrix.
        #
        # TODO: BMG requires it to be a *broadcast* matrix; what happens
        # if we feed one Dirichlet into another?  That would be a simplex,
        # not a broadcast matrix. Do some research here; do we actually
        # need the semantics of "always a broadcast matrix" ?

        required_rows = 1
        required_columns = self._required_columns
        t = PositiveReal.with_dimensions(required_rows, required_columns)
        return [always_matrix(t)]

    @property
    def size(self) -> torch.Size:
        return self.concentration.size

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

    def _supported_in_bmg(self) -> bool:
        return True


class FlatNode(DistributionNode):

    """The Flat distribution the standard uniform distribution from 0.0 to 1.0."""

    def __init__(self):
        DistributionNode.__init__(self, [])

    def _compute_graph_type(self) -> BMGLatticeType:
        return Probability

    def _compute_inf_type(self) -> BMGLatticeType:
        return Probability

    @property
    def requirements(self) -> List[Requirement]:
        return []

    def _supported_in_bmg(self) -> bool:
        return True

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "Flat()"

    def support(self) -> Iterator[Any]:
        raise ValueError("Flat distribution does not have finite support.")


class GammaNode(DistributionNode):
    """The gamma distribution is a distribution of positive
    real numbers characterized by positive real concentration and rate
    parameters."""

    def __init__(self, concentration: BMGNode, rate: BMGNode):
        DistributionNode.__init__(self, [concentration, rate])

    @property
    def concentration(self) -> BMGNode:
        return self.inputs[0]

    @concentration.setter
    def concentration(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def rate(self) -> BMGNode:
        return self.inputs[1]

    @rate.setter
    def rate(self, p: BMGNode) -> None:
        self.inputs[1] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return PositiveReal

    def _compute_inf_type(self) -> BMGLatticeType:
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        return [PositiveReal, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.concentration.size

    def __str__(self) -> str:
        return f"Gamma({str(self.concentration)}, {str(self.rate)})"

    def _supported_in_bmg(self) -> bool:
        return True

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

    def __init__(self, scale: BMGNode):
        DistributionNode.__init__(self, [scale])

    @property
    def scale(self) -> BMGNode:
        return self.inputs[0]

    @scale.setter
    def scale(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return PositiveReal

    def _compute_inf_type(self) -> BMGLatticeType:
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        # The input to a HalfCauchy must be a positive real.
        return [PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.scale.size

    def __str__(self) -> str:
        return f"HalfCauchy({str(self.scale)})"

    def _supported_in_bmg(self) -> bool:
        return True

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

    def __init__(self, mu: BMGNode, sigma: BMGNode):
        DistributionNode.__init__(self, [mu, sigma])

    @property
    def mu(self) -> BMGNode:
        return self.inputs[0]

    @mu.setter
    def mu(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def sigma(self) -> BMGNode:
        return self.inputs[1]

    @sigma.setter
    def sigma(self, p: BMGNode) -> None:
        self.inputs[1] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real

    def _compute_inf_type(self) -> BMGLatticeType:
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        # The mean of a normal must be a real; the standard deviation
        # must be a positive real.
        return [Real, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.mu.size

    def __str__(self) -> str:
        return f"Normal({str(self.mu)},{str(self.sigma)})"

    def _supported_in_bmg(self) -> bool:
        return True

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

    def __init__(self, df: BMGNode, loc: BMGNode, scale: BMGNode):
        DistributionNode.__init__(self, [df, loc, scale])

    @property
    def df(self) -> BMGNode:
        return self.inputs[0]

    @df.setter
    def df(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def loc(self) -> BMGNode:
        return self.inputs[1]

    @loc.setter
    def loc(self, p: BMGNode) -> None:
        self.inputs[1] = p

    @property
    def scale(self) -> BMGNode:
        return self.inputs[2]

    @scale.setter
    def scale(self, p: BMGNode) -> None:
        self.inputs[2] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real

    def _compute_inf_type(self) -> BMGLatticeType:
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        return [PositiveReal, Real, PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.df.size

    def __str__(self) -> str:
        return f"StudentT({str(self.df)},{str(self.loc)},{str(self.scale)})"

    def _supported_in_bmg(self) -> bool:
        return True

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

    def __init__(self, low: BMGNode, high: BMGNode):
        DistributionNode.__init__(self, [low, high])

    @property
    def low(self) -> BMGNode:
        return self.inputs[0]

    @low.setter
    def low(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def high(self) -> BMGNode:
        return self.inputs[1]

    @high.setter
    def high(self, p: BMGNode) -> None:
        self.inputs[1] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real

    def _compute_inf_type(self) -> BMGLatticeType:
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
    The inputs are the operands of each operator."""

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        BMGNode.__init__(self, inputs)


# ####
# #### Multiary operators
# ####


class MultiAdditionNode(OperatorNode):
    """This represents an addition of values."""

    # TODO: Do the same for multiplication
    # TODO: Consider a base class for multi add, logsumexp, and so on.

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        OperatorNode.__init__(self, inputs)

    def _compute_graph_type(self) -> BMGLatticeType:
        # We require:
        # * at least two values
        # * all values the same graph type
        # * that type is R, R+ or R-.
        if len(self.inputs) <= 1:
            return Malformed
        gt = self.inputs[0].graph_type
        if gt not in {Real, NegativeReal, PositiveReal}:
            return Malformed
        if any(i.graph_type != gt for i in self.inputs):
            return Malformed
        return gt

    def _compute_inf_type(self) -> BMGLatticeType:
        op_type = supremum(*[i.inf_type for i in self.inputs])
        if supremum(op_type, NegativeReal) == NegativeReal:
            return NegativeReal
        if supremum(op_type, PositiveReal) == PositiveReal:
            return PositiveReal
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        s = supremum(*[i.inf_type for i in self.inputs])
        if s not in {Real, NegativeReal, PositiveReal}:
            s = Real
        return [s] * len(self.inputs)

    @property
    def size(self) -> torch.Size:
        return self.inputs[0].size

    def support(self) -> Iterator[Any]:
        raise ValueError("support of multiary addition not yet implemented")

    def _supported_in_bmg(self) -> bool:
        return True

    def __str__(self) -> str:
        return "MultiAdd"


class LogSumExpNode(OperatorNode):
    """This class represents the LogSumExp operation: for values v_1, ..., v_n
    we compute log(exp(v_1) + ... + exp(v_n))"""

    # TODO: The original Python model allows the developer to choose which
    # dimension of the tensor to apply the logsumexp operation to. Right now
    # we only support single-dimensional tensors which are created using the
    # tensor constructor; we will expand this to more scenarios later.
    # For example, we might want to support compiling something like
    #
    # @rv def n(): return Normal(tensor(0., 0.), tensor(1., 1.))
    # y = n().logsumexp(dim=0)
    #
    # the same way we would compile the semantically equivalent model:
    #
    # @rv def n(x): return Normal(0., 1.)
    # y = tensor([n(0), n(1)]).logsumexp(dim=0)
    #
    # TODO: Similarly we might want to support logsumexp on dimensions other than 0.

    def __init__(self, inputs: List[BMGNode]):
        assert isinstance(inputs, list)
        OperatorNode.__init__(self, inputs)

    def _compute_graph_type(self) -> BMGLatticeType:
        # We require:
        # * at least two values
        # * all values the same graph type
        # * that type is R, R+ or R-.
        #
        # If these conditions are met then the graph type is real,
        # otherwise it is malformed

        if len(self.inputs) <= 1:
            return Malformed
        input_gt = self.inputs[0].graph_type
        if input_gt not in {Real, NegativeReal, PositiveReal}:
            return Malformed
        if any(i.graph_type != input_gt for i in self.inputs):
            return Malformed
        return Real

    def _compute_inf_type(self) -> BMGLatticeType:
        # No matter what, logsumexp produces a real.

        # TODO: If we support multiple dimensional tensors someday we will
        # need to revisit this.

        # TODO: We could do slightly better in the degenerate case of a singleton.
        # If we have something like flip().logsumexp() where flip() produces a single
        # bool, then that's equivalent to just plain flip(), and we could know that
        # is a bool, not a real.  But this scenario is likely to be a bug in the
        # model in the first place, so there's no reason to prioritize creating an
        # optimization for its type analysis.

        return Real

    @property
    def requirements(self) -> List[Requirement]:
        s = supremum(*[i.inf_type for i in self.inputs])
        if s not in {Real, NegativeReal, PositiveReal}:
            s = Real
        return [s] * len(self.inputs)

    @property
    def size(self) -> torch.Size:
        # TODO: If we ever support logsumexp on a dimension other than 1
        # then we will need to update this.
        return torch.Size([])

    def __str__(self) -> str:
        return "LogSumExp"

    def support(self) -> Iterator[Any]:
        raise ValueError("support of LogSumExp not yet implemented")

    def _supported_in_bmg(self) -> bool:
        return True


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

    def __init__(self, condition: BMGNode, consequence: BMGNode, alternative: BMGNode):
        OperatorNode.__init__(self, [condition, consequence, alternative])

    def _compute_graph_type(self) -> BMGLatticeType:
        if self.consequence.graph_type == self.alternative.graph_type:
            return self.consequence.graph_type
        return Malformed

    def _compute_inf_type(self) -> BMGLatticeType:
        return supremum(self.consequence.inf_type, self.alternative.inf_type)

    @property
    def requirements(self) -> List[Requirement]:
        # We require that the consequence and alternative types of the if-then-else
        # be exactly the same. In order to minimize the output type of the node
        # we will take the supremum of the infimums of the input types, and then
        # require that the inputs each be of that type.
        # The condition input must be a bool.
        it = self.inf_type
        return [Boolean, it, it]

    @property
    def condition(self) -> BMGNode:
        return self.inputs[0]

    @condition.setter
    def condition(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def consequence(self) -> BMGNode:
        return self.inputs[1]

    @consequence.setter
    def consequence(self, p: BMGNode) -> None:
        self.inputs[1] = p

    @property
    def alternative(self) -> BMGNode:
        return self.inputs[2]

    @alternative.setter
    def alternative(self, p: BMGNode) -> None:
        self.inputs[2] = p

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

    def _supported_in_bmg(self) -> bool:
        return True


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

    @left.setter
    def left(self, p: BMGNode) -> None:
        self.inputs[0] = p

    @property
    def right(self) -> BMGNode:
        return self.inputs[1]

    @right.setter
    def right(self, p: BMGNode) -> None:
        self.inputs[1] = p

    def support_size(self) -> float:
        return self.left.support_size() * self.right.support_size()


class ComparisonNode(BinaryOperatorNode, metaclass=ABCMeta):
    """This is the base class for all comparison operators."""

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def _compute_inf_type(self) -> BMGLatticeType:
        return Boolean

    def _compute_graph_type(self) -> BMGLatticeType:
        return Boolean

    @property
    def requirements(self) -> List[Requirement]:
        return [self.left.inf_type, self.right.inf_type]

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) < torch.zeros(self.right.size)).size()

    def support_size(self) -> float:
        return 2.0 ** prod(self.size)


class GreaterThanNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el > ar for el in self.left.support() for ar in self.right.support()
        )

    def __str__(self) -> str:
        return f"({str(self.left)}>{str(self.right)})"


class GreaterThanEqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el >= ar for el in self.left.support() for ar in self.right.support()
        )

    def __str__(self) -> str:
        return f"({str(self.left)}>={str(self.right)})"


class LessThanNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el < ar for el in self.left.support() for ar in self.right.support()
        )

    def __str__(self) -> str:
        return f"({str(self.left)}<{str(self.right)})"


class LessThanEqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el <= ar for el in self.left.support() for ar in self.right.support()
        )

    def __str__(self) -> str:
        return f"({str(self.left)}<={str(self.right)})"


class EqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el == ar for el in self.left.support() for ar in self.right.support()
        )

    def __str__(self) -> str:
        return f"({str(self.left)}=={str(self.right)})"


class NotEqualNode(ComparisonNode):
    def __init__(self, left: BMGNode, right: BMGNode):
        ComparisonNode.__init__(self, left, right)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el != ar for el in self.left.support() for ar in self.right.support()
        )

    def __str__(self) -> str:
        return f"({str(self.left)}!={str(self.right)})"


class AdditionNode(BinaryOperatorNode):
    """This represents an addition of values."""

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    @property
    def can_be_complement(self) -> bool:
        if self.left.inf_type == One:
            other = self.right
            if isinstance(other, NegateNode):
                it = other.operand.inf_type
                if supremum(it, Probability) == Probability:
                    return True
        if self.right.inf_type == One:
            other = self.left
            if isinstance(other, NegateNode):
                it = other.operand.inf_type
                if supremum(it, Probability) == Probability:
                    return True
        return False

    def _compute_inf_type(self) -> BMGLatticeType:
        # The BMG addition node requires:
        # * the operands and the result type to be the same
        # * that type must be R+, R- or R.
        #
        # However, we can make transformations during the problem-fixing phase
        # that enable other combinations of types:
        #
        # * If one operand is the constant 1.0 and the other is
        #   a negate operator applied to a B or P, then we can turn
        #   the whole thing into a complement node of type B or P.
        #
        # TODO:
        # * If one operand is a constant N or B and the other is
        #   any B, we can generate an if-then-else of type N.

        if self.can_be_complement:
            if self.left.inf_type == One:
                other = self.right
            else:
                other = self.left
            assert isinstance(other, NegateNode)
            return other.operand.inf_type

        # There is no way to turn this into a complement. Can we make both
        # operands into a negative real?
        op_type = supremum(self.left.inf_type, self.right.inf_type)
        if supremum(op_type, NegativeReal) == NegativeReal:
            return NegativeReal
        # Can we make both operands into a positive real?
        if supremum(op_type, PositiveReal) == PositiveReal:
            return PositiveReal
        return Real

    def _compute_graph_type(self) -> BMGLatticeType:
        if self.left.graph_type != self.right.graph_type:
            return Malformed
        t = self.left.graph_type
        if t in {PositiveReal, NegativeReal, Real}:
            return t
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # If we have 1 + (-P), 1 + (-B), (-P) + 1 or (-B) + 1, then
        # the nodes already meet their requirements and we will convert
        # this to a complement.

        if self.can_be_complement:
            return [self.left.inf_type, self.right.inf_type]

        # We require that the input types of an addition be exactly the same.
        # In order to minimize the output type of the node we will take the
        # supremum of the infimums of the input types, and then require that
        # the inputs each be of that type.
        it = self.inf_type
        return [it, it]

    @property
    def size(self) -> torch.Size:
        return (torch.zeros(self.left.size) + torch.zeros(self.right.size)).size()

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

    # There is no division node in BMG; we will replace
    # x / y with x * (y ** (-1)) during the "fix problems"
    # phase.

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

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

    def _compute_inf_type(self) -> BMGLatticeType:
        # Both operands must be R, R+ or T, they must be the same,
        # and the result type is that type.
        return supremum(self.left.inf_type, self.right.inf_type, PositiveReal)

    def _compute_graph_type(self) -> BMGLatticeType:
        # Both operands must be R, R+ or T, they must be the same,
        # and the result type is that type.
        lgt = self.left.graph_type
        if lgt != self.right.graph_type:
            return Malformed
        if lgt != PositiveReal and lgt != Real and lgt != BMGTensor:
            return Malformed
        return lgt

    @property
    def requirements(self) -> List[Requirement]:
        # We require that both inputs be the same as the output.
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
    # * Even numbered inputs are keys
    # * All keys are constant nodes
    # * Odd numbered inputs are values associated with the previous
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

    def __init__(self, inputs: List[BMGNode]):
        # TODO: Check that keys are all constant nodes.
        # TODO: Check that there is one value for each key.
        # TODO: Verify that there is at least one pair.
        BMGNode.__init__(self, inputs)

    def _compute_inf_type(self) -> BMGLatticeType:
        # The inf type of a map is the supremum of the types of all
        # its inputs.
        return supremum(
            *[self.inputs[i * 2 + 1].inf_type for i in range(len(self.inputs) // 2)]
        )

    def _compute_graph_type(self) -> BMGLatticeType:
        first = self.inputs[0].graph_type
        for i in range(len(self.inputs) // 2):
            if self.inputs[i * 2 + 1].graph_type != first:
                return Malformed
        return first

    @property
    def requirements(self) -> List[Requirement]:
        it = self.inf_type
        # TODO: This isn't quite right; when we support this kind of node
        # in BMG, fix this.
        return [upper_bound(BMGTensor), it] * (len(self.inputs) // 2)

    @property
    def size(self) -> torch.Size:
        return self.inputs[1].size

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
        for i in range(len(self.inputs) // 2):
            if self.inputs[i * 2].value == k:
                return self.inputs[i * 2 + 1]
        raise ValueError("Key not found in map")


class IndexNodeDeprecated(BinaryOperatorNode):

    # TODO: The index / map node combination that we originally envisioned
    # does not work well with BMGs indexing operator; we will eventually
    # remove it. Until then, just mark it as deprecated to minimize
    # disruption while we get indexing support working.

    """This represents a stochastic choice of multiple options; the left
    operand must be a map, and the right is the stochastic value used to
    choose an element from the map."""

    # See notes on MapNode for an explanation of this code.

    def __init__(self, left: MapNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def _compute_inf_type(self) -> BMGLatticeType:
        # The inf type of an index is that of its map.
        return self.left.inf_type

    def _compute_graph_type(self) -> BMGLatticeType:
        return self.left.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        it = self.inf_type
        # TODO: This isn't quite right; when we support this kind of node
        # in BMG, fix this.
        return [upper_bound(BMGTensor), it]

    @property
    def size(self) -> torch.Size:
        return self.left.size

    def __str__(self) -> str:
        return str(self.left) + "[" + str(self.right) + "]"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(
            el for ar in self.right.support() for el in self.left[ar].support()
        )


class IndexNode(BinaryOperatorNode):
    """This represents a stochastic index into a vector. The left operand
    is the vector and the right operand is the index."""

    def __init__(self, left: MapNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def _compute_inf_type(self) -> BMGLatticeType:
        # The inf type of an index is derived from the inf type of
        # the vector, but it's not as straightforward as just
        # shrinking the type down to a 1x1 matrix. The elements of
        # a one-hot vector are bools, for instance, not all one.
        # The elements of a simplex are probabilities.
        lt = self.left.inf_type
        if isinstance(lt, OneHotMatrix):
            return Boolean
        # See notes in "requirements", below.
        if isinstance(lt, ZeroMatrix):
            return Boolean
        if isinstance(lt, SimplexMatrix):
            return Probability
        if isinstance(lt, BMGMatrixType):
            return lt.with_dimensions(1, 1)
        # The only other possibility is that we have a tensor, so let's say
        # its elements are reals.
        return Real

    def _compute_graph_type(self) -> BMGLatticeType:
        lt = self.left.graph_type
        # These should be impossible
        assert not isinstance(lt, OneHotMatrix)
        assert not isinstance(lt, ZeroMatrix)
        if isinstance(lt, SimplexMatrix):
            return Probability
        if isinstance(lt, BMGMatrixType):
            return lt.with_dimensions(1, 1)
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # The index operator introduces an interesting wrinkle into the
        # "requirements" computation. Until now we have always had the property
        # that queries and observations are "sinks" of the graph, and the transitive
        # closure of the inputs to the sinks can have their requirements checked in
        # order going from the nodes farthest from the sinks down to the sinks.
        # That is, each node can meet its input requirements *before* its output
        # nodes meet their requirements. We now have a case where doing so creates
        # potential inefficiencies.
        #
        # B is the constant vector [0, 1, 1]
        # N is any node of type natural.
        # I is an index
        # F is Bernoulli.
        #
        #   B N
        #   | |
        #    I
        #    |
        #    F
        #
        # The requirement on edge I->F is Probability
        # The requirement on edge N->I is Natural.
        # What is the requirement on the B->I edge?
        #
        # If we say that it is Boolean[1, 3], its inf type, then the graph we end up
        # generating is
        #
        # b = const_bool_matrix([0, 1, 1])  # bool matrix
        # n = whatever                      # natural
        # i = index(b, i)                   # bool
        # z = const_prob(0)                 # prob
        # o = const_prob(1)                 # prob
        # c = if_then_else(i, o, z)         # prob
        # f = Bernoulli(c)                  # bool
        #
        # But it would be arguably better to produce
        #
        # b = const_prob_matrix([0, 1, 1])  # prob matrix
        # n = whatever                      # natural
        # i = index(b, i)                   # prob
        # f = Bernoulli(i)                  # bool
        #
        # TODO: We might consider an optimization pass which does so.
        #
        # However there is an even worse situation. Suppose we have
        # this unlikely-but-legal graph:
        #
        # Z is [0, 0, 0]
        # N is any natural
        # I is an index
        # C requires a Boolean input
        # L requires a NegativeReal input
        #
        #    Z   N
        #     | |
        #      I
        #     | |
        #    C   L
        #
        # The inf type of Z is Zero[1, 3].
        # The I->C edge requirement is Boolean
        # The I->L edge requirement is NegativeReal
        #
        # Now what requirement do we impose on the Z->I edge? We have our choice
        # of "matrix of negative reals" or "matrix of bools", and whichever we
        # pick will disappoint someone.
        #
        # Fortunately for us, this situation is unlikely; a model writer who
        # contrives a situation where they are making a stochastic choice where
        # all choices are all zero AND that zero needs to be used as both
        # false and a negative number is not writing realistic models.
        #
        # What we will do in this unlikely situation is decide that the intended
        # output type is Boolean and therefore the vector is a vector of bools.
        #
        # -----
        #
        # We require:
        # * the vector must be one row
        # * the vector must be a matrix, not a single value
        # * the vector must be either a simplex, or a matrix where the element
        #   type is the output type of the indexing operation
        # * the index must be a natural
        #

        lt = self.left.inf_type

        # If we have a tensor that has more than two dimensions, who can
        # say what the column count should be?

        # TODO: We need a better error message for that scenario.
        # It will be common for people to use tensors that are too high
        # dimension for BMG to handle and we should say that clearly.

        required_columns = lt.columns if isinstance(lt, BMGMatrixType) else 1
        required_rows = 1

        if isinstance(lt, SimplexMatrix):
            vector_req = lt.with_dimensions(required_rows, required_columns)
        else:
            it = self.inf_type
            assert isinstance(it, BMGMatrixType)
            vector_req = it.with_dimensions(required_rows, required_columns)

        return [always_matrix(vector_req), Natural]

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return str(self.left) + "[" + str(self.right) + "]"

    def support(self) -> Iterator[Any]:
        raise NotImplementedError("support of index operator not implemented")

    def _supported_in_bmg(self) -> bool:
        return True


class MatrixMultiplicationNode(BinaryOperatorNode):
    """"This represents a matrix multiplication."""

    # TODO: We now have matrix multiplication in BMG; finish this implementation

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def _compute_inf_type(self) -> BMGLatticeType:
        # TODO: Fix this
        return supremum(self.left.inf_type, self.right.inf_type)

    def _compute_graph_type(self) -> BMGLatticeType:
        # TODO: Fix this
        if self.left.graph_type == self.right.graph_type:
            return self.left.graph_type
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: Fix this
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

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def _compute_graph_type(self) -> BMGLatticeType:
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
        if lgt == Boolean or lgt == Natural:
            return Malformed
        return lgt

    def _compute_inf_type(self) -> BMGLatticeType:
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
        if lit == Boolean:
            return rit
        if rit == Boolean:
            return lit

        # If neither type is Boolean then the best we can do is the sup
        # of the left type, the right type, and Probability.
        return supremum(self.left.inf_type, self.right.inf_type, Probability)

    @property
    def requirements(self) -> List[Requirement]:
        # As noted above, we have special handling if an operand to a multiplication
        # is a Boolean. In those cases, we can simply impose a requirement that can
        # always be met: that the operands be of their inf types.
        lit = self.left.inf_type
        rit = self.right.inf_type
        if lit == Boolean or rit == Boolean:
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

    def __init__(self, left: BMGNode, right: BMGNode):
        BinaryOperatorNode.__init__(self, left, right)

    def _compute_inf_type(self) -> BMGLatticeType:
        # Given the inf types of the operands, what is the smallest
        # possible type we could make the result?
        #
        # BMG supports a power node that has these possible combinations of
        # base and exponent type:
        #
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
        # The smallest type we can make the return if we generate a BMG power
        # node can be found by:
        #
        # * treat the base type as the larger of its inf type and Probability
        # * treat the exp type as the larger of its inf type and Positive Real.
        # * return the best match from the table above.
        #
        # However, there are some cases where we can generate a smaller
        # result type:
        #
        # * We generate x ** b where b is bool as "if b then x else 1", which
        #   is of the same type as x.
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

        inf_base = self.left.inf_type
        inf_exp = self.right.inf_type

        if inf_base == Tensor or inf_exp == Tensor:
            return Tensor

        if supremum(inf_exp, Boolean) == Boolean:
            return inf_base

        inf_base = supremum(inf_base, Probability)

        if inf_base == Probability and inf_exp == Real:
            return PositiveReal

        return inf_base

    def _supported_in_bmg(self) -> bool:
        return True

    def _compute_graph_type(self) -> BMGLatticeType:
        # Figure out which of these cases we are in; otherwise
        # return Malformed.

        # P ** R+  --> P
        # P ** R   --> R+
        # R+ ** R+ --> R+
        # R+ ** R  --> R+
        # R ** R+  --> R
        # R ** R   --> R
        lt = self.left.graph_type
        rt = self.right.graph_type
        if lt != Probability and lt != PositiveReal and lt != Real:
            return Malformed
        if rt != PositiveReal and rt != Real:
            return Malformed
        if lt == Probability and rt == Real:
            return PositiveReal
        return lt

    @property
    def requirements(self) -> List[Requirement]:
        # P ** R+
        # P ** R
        # R+ ** R+
        # R+ ** R
        # R ** R+
        # R ** R

        inf_base = self.left.inf_type
        inf_exp = supremum(Boolean, self.right.inf_type)

        if inf_base == Tensor or inf_exp == Tensor:
            return [Tensor, Tensor]

        if inf_exp == Boolean:
            return [inf_base, inf_exp]

        return [supremum(inf_base, Probability), supremum(inf_exp, PositiveReal)]

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

    def __init__(self, operand: BMGNode):
        OperatorNode.__init__(self, [operand])

    @property
    def operand(self) -> BMGNode:
        return self.inputs[0]

    @operand.setter
    def operand(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def support_size(self) -> float:
        return self.operand.support_size()


class ExpNode(UnaryOperatorNode):
    """This represents an exponentiation operation; it is generated when
    a model contains calls to Tensor.exp or math.exp."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_inf_type(self) -> BMGLatticeType:
        ot = self.operand.inf_type
        if supremum(ot, NegativeReal) == NegativeReal:
            return Probability
        return PositiveReal

    def _compute_graph_type(self) -> BMGLatticeType:
        ot = self.operand.graph_type
        if ot == Real or ot == PositiveReal:
            return PositiveReal
        if ot == NegativeReal:
            return Probability
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        ot = self.operand.inf_type
        if supremum(ot, NegativeReal) == NegativeReal:
            return [NegativeReal]
        if supremum(ot, PositiveReal) == PositiveReal:
            return [PositiveReal]
        return [Real]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Exp(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.exp(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


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

    def _compute_inf_type(self) -> BMGLatticeType:
        # ExpM1 takes a real, positive real or negative real. Its return has
        # the same type as its input.
        ot = self.operand.inf_type
        if supremum(ot, PositiveReal) == PositiveReal:
            return PositiveReal
        if supremum(ot, NegativeReal) == NegativeReal:
            return NegativeReal
        return Real

    def _compute_graph_type(self) -> BMGLatticeType:
        ot = self.operand.graph_type
        if ot in {Real, PositiveReal, NegativeReal}:
            return ot
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        return [self.inf_type]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "ExpM1(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.expm1(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class LogisticNode(UnaryOperatorNode):
    """This represents the operation 1/(1+exp(x)); it is generated when
    a model contains calls to Tensor.sigmoid."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_inf_type(self) -> BMGLatticeType:
        return Probability

    def _compute_graph_type(self) -> BMGLatticeType:
        ot = self.operand.graph_type
        if ot == Real:
            return Probability
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        return [Real]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Logistic(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.sigmoid(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class LogNode(UnaryOperatorNode):
    """This represents a log operation; it is generated when
    a model contains calls to Tensor.log or math.log."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    # The log node can either:
    # * Take a positive real and produce a real
    # * Take a probability and produce a negative real

    def _compute_inf_type(self) -> BMGLatticeType:
        # If the operand is convertible to probability then
        # the smallest we can make the log is negative real.
        ot = supremum(self.operand.inf_type, Probability)
        if ot == Probability:
            return NegativeReal
        return Real

    def _compute_graph_type(self) -> BMGLatticeType:
        ot = self.operand.graph_type
        if ot == Probability:
            return NegativeReal
        if ot == PositiveReal:
            return Real
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # If the operand is less than or equal to probability,
        # require that it be converted to probability. Otherwise,
        # convert it to positive real.
        ot = supremum(self.operand.inf_type, Probability)
        if ot == Probability:
            return [Probability]
        return [PositiveReal]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Log(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.log(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


# TODO: replace "log" with "log1mexp" as needed below and update defs


class Log1mexpNode(UnaryOperatorNode):
    """This represents a log1mexp operation; it is generated when
    a model contains calls to log1mexp or math_log1mexp."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    # The log1mexp node can only:
    # * Take a negative real and produce a negative real

    def _compute_inf_type(self) -> BMGLatticeType:
        return NegativeReal

    def _compute_graph_type(self) -> BMGLatticeType:
        ot = self.operand.graph_type
        if ot == NegativeReal:
            return NegativeReal
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        return [NegativeReal]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Log1mexp(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(torch.log(1 - torch.exp(o)) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


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

    def _compute_inf_type(self) -> BMGLatticeType:
        ot = self.operand.inf_type
        if supremum(ot, PositiveReal) == PositiveReal:
            return NegativeReal
        if supremum(ot, NegativeReal) == NegativeReal:
            return PositiveReal
        return Real

    def _compute_graph_type(self) -> BMGLatticeType:
        ot = self.operand.graph_type
        if ot == PositiveReal:
            return NegativeReal
        if ot == NegativeReal:
            return PositiveReal
        if ot == Real:
            return Real
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # Find the smallest type that can be used as an
        # input requirement.
        ot = self.operand.inf_type
        if supremum(ot, PositiveReal) == PositiveReal:
            return [PositiveReal]
        if supremum(ot, NegativeReal) == NegativeReal:
            return [NegativeReal]
        return [Real]

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

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_inf_type(self) -> BMGLatticeType:
        # TODO: When we support this node in BMG, revisit this code.
        return Boolean

    def _compute_graph_type(self) -> BMGLatticeType:
        return self.operand.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: When we support this node in BMG, revisit this code.
        return [self.inf_type]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "not " + str(self.operand)

    def support(self) -> Iterator[Any]:
        return SetOfTensors(not o for o in self.operand.support())


class ComplementNode(UnaryOperatorNode):
    """This represents a complement of a Boolean or probability
    value."""

    # See notes above NegateNode for details

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_inf_type(self) -> BMGLatticeType:
        # Note that we should not be generating complement nodes in the graph
        # unless we can type them correctly. The inf type should always
        # be probability or bool. If somehow it is not -- perhaps because
        # we are running a unit test -- then treat the inf type as probability
        # if the operand type is not correct. The inf type should always be
        # a genuine type, not Malformed.
        it = self.operand.inf_type
        if supremum(it, Boolean) == Boolean:
            return Boolean
        return Probability

    def _compute_graph_type(self) -> BMGLatticeType:
        # Note that we should not be generating complement nodes in the graph
        # unless we can type them correctly. The graph type should always
        # be probability or bool. If somehow it is not -- perhaps because
        # we are running a unit test -- then treat the node as malformed.
        t = self.operand.graph_type
        if t == Boolean or t == Probability:
            return t
        return Malformed

    @property
    def requirements(self) -> List[Requirement]:
        # We require that the input type be the same as the output type.
        return [self.inf_type]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "complement " + str(self.operand)

    def support(self) -> Iterator[Any]:
        # This should never be called because we never generate
        # a complement node while executing the model to accumulate
        # the graph.
        return [1 - p for p in self.operand.support]

    def _supported_in_bmg(self) -> bool:
        return True


class PhiNode(UnaryOperatorNode):
    """This represents a phi operation; that is, the cumulative
    distribution function of the standard normal."""

    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    # The log node only takes a real and only produces a probability

    def _compute_inf_type(self) -> BMGLatticeType:
        return Probability

    def _compute_graph_type(self) -> BMGLatticeType:
        return Probability

    @property
    def requirements(self) -> List[Requirement]:
        return [Real]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    def __str__(self) -> str:
        return "Phi(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        cdf = Normal(0.0, 1.0).cdf
        return SetOfTensors(cdf(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class SampleNode(UnaryOperatorNode):
    """This represents a single unique sample from a distribution;
    if a graph has two sample nodes both taking input from the same
    distribution, each sample is logically distinct. But if a graph
    has two nodes that both input from the same sample node, we must
    treat those two uses of the sample as though they had identical
    values."""

    def __init__(self, operand: DistributionNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_inf_type(self) -> BMGLatticeType:
        # The infimum type of a sample is that of its distribution.
        return self.operand.inf_type

    def _compute_graph_type(self) -> BMGLatticeType:
        return self.operand.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        return [self.inf_type]

    @property
    def size(self) -> torch.Size:
        return self.operand.size

    @property
    def operand(self) -> DistributionNode:
        c = self.inputs[0]
        assert isinstance(c, DistributionNode)
        return c

    @operand.setter
    def operand(self, p: DistributionNode) -> None:
        self.inputs[0] = p

    def __str__(self) -> str:
        return "Sample(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return self.operand.support()

    def _supported_in_bmg(self) -> bool:
        return True


class ToRealNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real

    def _compute_inf_type(self) -> BMGLatticeType:
        # A ToRealNode's output is always real
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        # A ToRealNode's input must be real or smaller.
        return [upper_bound(Real)]

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
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_graph_type(self) -> BMGLatticeType:
        return PositiveReal

    def _compute_inf_type(self) -> BMGLatticeType:
        # A ToPositiveRealNode's output is always PositiveReal
        return PositiveReal

    @property
    def requirements(self) -> List[Requirement]:
        # A ToPositiveRealNode's input must be PositiveReal or smaller.
        return [upper_bound(PositiveReal)]

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "ToPosReal(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return SetOfTensors(float(o) for o in self.operand.support())

    def _supported_in_bmg(self) -> bool:
        return True


class ToProbabilityNode(UnaryOperatorNode):
    def __init__(self, operand: BMGNode):
        UnaryOperatorNode.__init__(self, operand)

    def _compute_graph_type(self) -> BMGLatticeType:
        return Probability

    def _compute_inf_type(self) -> BMGLatticeType:
        return Probability

    @property
    def requirements(self) -> List[Requirement]:
        # TODO: A ToProbabilityNode's input must be real, positive real
        # or probability, but we don't have a bound for that. However,
        # we do not need one, since this node is only ever added when
        # a requirement violation exists in the graph! We will only
        # add this node when it is legal to do so; it does not matter
        # what requirement we give here.
        return [upper_bound(Real)]

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "ToProb(" + str(self.operand) + ")"

    def support(self) -> Iterator[Any]:
        return self.operand.support()

    def _supported_in_bmg(self) -> bool:
        return True


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
    # models.)  Moreover it is not yet clear how exactly
    # observation nodes are to be generated by the compiler;
    # from what model source code, if any, do we generate these
    # nodes?  This code is only used right now for testing purposes
    # and we need to do significant design work to figure out
    # how users wish to generate observations of models that have
    # been compiled into BMG.
    value: Any

    def __init__(self, observed: SampleNode, value: Any):
        self.value = value
        BMGNode.__init__(self, [observed])

    @property
    def observed(self) -> SampleNode:
        c = self.inputs[0]
        assert isinstance(c, SampleNode)
        return c

    @observed.setter
    def operand(self, p: SampleNode) -> None:
        self.inputs[0] = p

    def _compute_inf_type(self) -> BMGLatticeType:
        # Since an observation node is never consumed it is not actually
        # meaningful to compute its type. However, we can use this to check for
        # errors; for example, if we have an observation with value 0.5 on
        # an operation known to be of type Natural then we flag that as an
        # error.
        return self.observed.inf_type

    def _compute_graph_type(self) -> BMGLatticeType:
        return self.observed.graph_type

    @property
    def requirements(self) -> List[Requirement]:
        return [AnyRequirement()]

    @property
    def size(self) -> torch.Size:
        if isinstance(self.value, Tensor):
            return self.value.size()
        return torch.Size([])

    def __str__(self) -> str:
        return str(self.observed) + "=" + str(self.value)

    def _supported_in_bmg(self) -> bool:
        return True

    def support(self) -> Iterator[Any]:
        return []


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

    @operator.setter
    def operator(self, p: BMGNode) -> None:
        self.inputs[0] = p

    def _compute_graph_type(self) -> BMGLatticeType:
        return self.operator.graph_type

    def _compute_inf_type(self) -> BMGLatticeType:
        return self.operator.inf_type

    @property
    def requirements(self) -> List[Requirement]:
        return [AnyRequirement()]

    @property
    def size(self) -> torch.Size:
        return self.operator.size

    def __str__(self) -> str:
        return "Query(" + str(self.operator) + ")"

    def _supported_in_bmg(self) -> bool:
        return True

    def support(self) -> Iterator[Any]:
        return []


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

    def _compute_graph_type(self) -> BMGLatticeType:
        return Real

    def _compute_inf_type(self) -> BMGLatticeType:
        return Real

    @property
    def requirements(self) -> List[Requirement]:
        # Each input to an exp-power is required to be a
        # real, negative real, positive real or probability.
        return [
            i.inf_type
            if i.inf_type in {Real, NegativeReal, PositiveReal, Probability}
            else Real
            for i in self.inputs
        ]

    @property
    def size(self) -> torch.Size:
        return torch.Size([])

    def __str__(self) -> str:
        return "ExpProduct"

    def support(self) -> Iterator[Any]:
        # Factors never produce output so it is not meaningful to compute
        # their support
        return []

    def _supported_in_bmg(self) -> bool:
        return True
