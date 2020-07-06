# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Union

from beanmachine.ppl.utils.memoize import memoize
from torch import Tensor


"""This module contains type definitions used as markers
for BMG types not represented in the Python type system.

When we construct a graph we know all the "storage" types of
the nodes -- Boolean, integer, float, tensor, and so on.
But Bean Machine Graph requires that we ensure that "semantic"
type associations are made to each node in the graph. The
types in the BMG type system are:

Unknown       -- we largely do not need to worry about this one,
                 and it is more "undefined" than "unknown"
Boolean       -- we can just use bool
Real          -- we can just use float, but we'll make an alias Real
Tensor        -- we can just use Tensor
Probability   -- a real between 0.0 and 1.0
Positive Real -- what it says on the tin
Natural       -- a non-negative integer

The type definitions are the objects which represent the last three.

During construction of a graph we may create nodes which need to be
"fixed up" later; for example, a multiplication node with a tensor
on one side and a real on the other cannot be represented in the BMG
type system. We will mark such nodes as having the "Malformed" type.
"""

# TODO: We might also need:
# * Bounded natural -- a sample from a categorical
# * Simplex         -- a vector of probabilities that adds to 1.0,
#                      for the input to a categorical.


class Probability:
    pass


class PositiveReal:
    pass


class Natural:
    pass


class Malformed:
    pass


Real = float


"""
When converting from an accumulated graph that uses Python types, we
can express the rules concisely by defining a *type lattice*. A type
lattice is a DAG which meets these conditions:

* Nodes are types
* Edges are directed from "larger" types to "smaller" types.
  (Note that there is no requirement for a total order.)
* There is a function called "supremum" which takes two types
  and returns the unique type that is the smallest type that
  is bigger than both arguments.
* There is a function called "infimum" which similarly is
  the unique largest type smaller than both arguments.
  (Right now we do not actually need this function in our
  type analysis so it is not implemented.)

The type lattice of the BMG type system is:

          tensor
            |
          real
            |
         posreal
         |     |
        nat   prob
          |   |
          bool

where bool is the smallest type and tensor is the largest.

Why is this useful?

Our goal is to generate a graph that meets all the requirements of the BMG
type system. In order to do so, we compute a *requirement object* for each
*edge* in the graph.

There are two kinds of requirements: "input must be exactly type X" and
"input must be type X or smaller".  The first kind of requirement we
call an "exact requirement" and the latter is an "upper bound requirement".

(We do not at this time have any scenarios that need "lower bound
requirements" but if we need them we can add them.)

Once we know the requirements on the *incoming* edges to a node, we can
check to see if the requirements are met by the nodes.
* If they are, then we're good.
* If not then we can do a graph mutation which causes the requirements
to be met.
* If there is no such mutation then we can report an error.

How do we compute the requirements for an edge? It depends on the kind
of graph node that the edge is attached to. A Bernoulli node, for example,
requires that its input be "exactly Probability". A "to real" node requires
that its input be "real or smaller".

Some nodes however introduce more complex requirements. The requirements
of a multiplication node, for instance, are:

* the input types must be greater than or equal to Probability
* the input types must be exactly the same

We do not have a requirement object for either "greater than or equal", or
for "this edge must be the same as that edge". And we have one more
factor to throw in: the output type of a multiplication is the same as its
input types, so we wish to find the *smallest possible* restriction on the
input types so that the multiplication's type is minimized.

(That is: if we have a multiplication of a natural by a probability, we do
not wish to require that both be converted to reals, because then the
multiplication node could not be used in a context where a positive real
was required.)

How are we going to do this?

We generate the edge requirements for a multiplication as follows:

The requirement on both input edges is that they be *exactly* equal
to the *supremum* of the *infimum types* of the two inputs.

The infimum type is the *smallest* type that each *node* in the graph
could possibly be. We call this type the "infimum type" of the node
because it is the infimum -- the *greatest lower bound* -- in the lattice.

It might not be clear why that works. Let's work through an example.

Suppose we have a sample from a beta; it's infimum type is Probability because
that's the only type a sample from a beta can be.  And suppose we have a sample
from a binomial; it's infimum type is Natural, again, because that's the only
type it can be.

Now suppose we multiply them. What restrictions go on the left and right
input edges of the multiplication?

The supremum of Natural and Probability is PositiveReal, so we put an "exactly
PositiveReal" restriction on the two edges. During the "fix problems" phase,
we see that we have an edge from a multiplication to a sample of type Probability,
but there is a requirement that it be a PositiveReal, so we insert a
ToPositiveRealNode between the multiplication and the sample. And then similarly
for the sample of type Natural.

The result is that we have a multiplication node that meets its requirements;
the input types are the same, and the output type is the smallest type it
could possibly be: PositiveReal.
"""

# This computes the supremum of two types.

# There are only 36 pairs; we can just list them.
_lookup = {
    (bool, bool): bool,
    (bool, Natural): Natural,
    (bool, Probability): Probability,
    (bool, PositiveReal): PositiveReal,
    (bool, Real): Real,
    (bool, Tensor): Tensor,
    (Natural, bool): Natural,
    (Natural, Natural): Natural,
    (Natural, Probability): PositiveReal,
    (Natural, PositiveReal): PositiveReal,
    (Natural, Real): Real,
    (Natural, Tensor): Tensor,
    (Probability, bool): Probability,
    (Probability, Natural): PositiveReal,
    (Probability, Probability): Probability,
    (Probability, PositiveReal): PositiveReal,
    (Probability, Real): Real,
    (Probability, Tensor): Tensor,
    (PositiveReal, bool): PositiveReal,
    (PositiveReal, Natural): PositiveReal,
    (PositiveReal, Probability): PositiveReal,
    (PositiveReal, PositiveReal): PositiveReal,
    (PositiveReal, Real): Real,
    (PositiveReal, Tensor): Tensor,
    (Real, bool): Real,
    (Real, Natural): Real,
    (Real, Probability): Real,
    (Real, PositiveReal): Real,
    (Real, Real): Real,
    (Real, Tensor): Tensor,
    (Tensor, bool): Tensor,
    (Tensor, Natural): Tensor,
    (Tensor, Probability): Tensor,
    (Tensor, PositiveReal): Tensor,
    (Tensor, Real): Tensor,
    (Tensor, Tensor): Tensor,
}


def _supremum(t: type, u: type) -> type:
    """Takes two BMG types; returns the smallest type that is
greater than or equal to both."""
    if (t, u) in _lookup:
        return _lookup[(t, u)]
    raise ValueError("Invalid arguments to _supremum")


# We can extend the two-argument supremum function to any number of arguments:
def supremum(*ts: type) -> type:
    """Takes any number of BMG types; returns the smallest type that is
greater than or equal to all of them."""
    result = bool
    for t in ts:
        result = _supremum(result, t)
    return result


def type_of_value(v: Any) -> type:
    """This computes the smallest BMG type that a given value fits into."""
    if isinstance(v, Tensor):
        if v.numel() == 1:
            return type_of_value(float(v))  # pyre-fixme
        return Tensor
    if isinstance(v, bool):
        return bool
    if isinstance(v, int):
        if v == 0 or v == 1:
            return bool
        if v >= 2:
            return Natural
        return Real
    if isinstance(v, float):
        if v == int(v):
            return type_of_value(int(v))
        if 0.0 <= v:
            if v <= 1.0:
                return Probability
            return PositiveReal
        return Real
    raise ValueError("Unexpected value passed to type_of_value")


# We will need to be able to express requirements on inputs;
# for example the input to a Bernoulli must be *exactly* a
# Probability, but the input to a ToPositiveReal must have
# any type smaller than or equal to PositiveReal.
#
# That is to say, we need to express *exact bounds* and
# *upper bounds*. At this time we do not need to express
# *lower bounds*; if we do, we can implement it using
# the same technique as here.
#
# To express an upper bound, we will wrap a type object
# in an upper bound wrapper; notice that the upper_bound
# factory method is memoized, so we can do reference equality
# to check to see if two bounds are equal.
#
# To express an exact bound, we'll just use the unwrapped type
# object itself; the vast majority of bounds will be exact bounds
# and I do not want to litter the code with calls to an "exact"
# helper method.


class UpperBound:
    bound: type

    def __init__(self, bound: type) -> None:
        self.bound = bound


Requirement = Union[type, UpperBound]


@memoize
def upper_bound(bound: Requirement) -> UpperBound:
    if isinstance(bound, UpperBound):
        return bound
    return UpperBound(bound)


def meets_requirement(t: type, r: Requirement) -> bool:
    if isinstance(r, UpperBound):
        return _supremum(t, r.bound) == r.bound
    return t == r


_type_names = {
    bool: "bool",
    Malformed: "malformed",
    Natural: "natural",
    PositiveReal: "positive real",
    Real: "real",
    Probability: "probability",
    Tensor: "tensor",
}

_short_type_names = {
    bool: "B",
    Malformed: "M",
    Natural: "N",
    PositiveReal: "R+",
    Real: "R",
    Probability: "P",
    Tensor: "T",
}


def name_of_type(t: type) -> str:
    return _type_names[t]


def short_name_of_type(t: type) -> str:
    return _short_type_names[t]


def name_of_requirement(r: Requirement) -> str:
    if isinstance(r, UpperBound):
        return "<=" + name_of_requirement(r.bound)
    assert isinstance(r, type)
    return name_of_type(r)


def short_name_of_requirement(r: Requirement) -> str:
    if isinstance(r, UpperBound):
        return "<=" + short_name_of_requirement(r.bound)
    assert isinstance(r, type)
    return short_name_of_type(r)
