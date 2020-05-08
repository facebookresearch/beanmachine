# Copyright (c) Facebook, Inc. and its affiliates.

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

The type definitions are the objects which represent the last three."""

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
