# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any

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
Real          -- we can just use float
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
"""

# There are only 36 pairs; we can just list them.
_lookup = {
    (bool, bool): bool,
    (bool, Natural): Natural,
    (bool, Probability): Probability,
    (bool, PositiveReal): PositiveReal,
    (bool, float): float,
    (bool, Tensor): Tensor,
    (Natural, bool): Natural,
    (Natural, Natural): Natural,
    (Natural, Probability): PositiveReal,
    (Natural, PositiveReal): PositiveReal,
    (Natural, float): float,
    (Natural, Tensor): Tensor,
    (Probability, bool): Probability,
    (Probability, Natural): PositiveReal,
    (Probability, Probability): Probability,
    (Probability, PositiveReal): PositiveReal,
    (Probability, float): float,
    (Probability, Tensor): Tensor,
    (PositiveReal, bool): PositiveReal,
    (PositiveReal, Natural): PositiveReal,
    (PositiveReal, Probability): PositiveReal,
    (PositiveReal, PositiveReal): PositiveReal,
    (PositiveReal, float): float,
    (PositiveReal, Tensor): Tensor,
    (float, bool): float,
    (float, Natural): float,
    (float, Probability): float,
    (float, PositiveReal): float,
    (float, float): float,
    (float, Tensor): Tensor,
    (Tensor, bool): Tensor,
    (Tensor, Natural): Tensor,
    (Tensor, Probability): Tensor,
    (Tensor, PositiveReal): Tensor,
    (Tensor, float): Tensor,
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
        return float
    if isinstance(v, float):
        if v == int(v):
            return type_of_value(int(v))
        if 0.0 <= v:
            if v <= 1.0:
                return Probability
            return PositiveReal
        return float
    raise ValueError("Unexpected value passed to type_of_value")
