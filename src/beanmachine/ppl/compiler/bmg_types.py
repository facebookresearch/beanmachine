# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any, Union

import torch
from beanmachine.ppl.utils.memoize import MemoizedClass, memoize
from torch import Size


"""This module contains class definitions and helper functions
for describing the *types* of graph nodes and the *type restrictions*
on graph edges.

When we construct a graph we know all the "storage" types of
the nodes -- Boolean, integer, float, tensor -- as they were
in the original Python program. BMG requires that we ensure
that "semantic" type associations are made to each node in the graph.
The types in the BMG type system are as follows:

* Unknown: a pseudo-type used as a marker when the type of
  a node is undefined. We do not have to worry about this one.

There are five "scalar" types:

* Boolean (B)
* Probability (P) -- a real between 0.0 and 1.0
* Natural (N) -- a non-negative integer
* Positive Real (R+)
* Negative Real (R-)
* Real (R)

There are infinitely many "matrix" types, but they can be
divided into the following kinds:

* Matrix of booleans (MB[r, c])
* Matrix of naturals (MN[r, c])
* Matrix of probabilities (MP[r, c])
* Matrix of positive reals (MR+[r, c])
* Matrix of negative reals (MR-[r, c])
* Matrix of reals (MR[r, c])
* Row simplex (S[r, c]): A restriction on MP[r, c] such that every row adds to 1.0.

There are infinitely many because all matrix types track their number
of rows and columns. All matrix types are two-dimensional, and the
row and column counts are constants, not stochastic quantities.

Because a scalar and a 1x1 matrix are effectively the same type,
for the purposes of this analysis we will only consider matrix types.
That is, we will make "Real" and "Probability" and so on aliases for
"1x1 real matrix" and "1x1 probability matrix".

To facilitate analysis, we organize the infinite set of types into a *lattice*.
See below for further details.
"""

# TODO: We might also need:
# * Bounded natural -- a sample from a categorical
# * Positive definite -- a real matrix with positive eigenvalues


def _size_to_rc(size: Size):
    dimensions = len(size)
    assert dimensions <= 2
    if dimensions == 0:
        return 1, 1
    r = 1 if dimensions == 1 else size[0]
    c = size[0] if dimensions == 1 else size[1]
    return r, c


class BMGLatticeType:
    short_name: str
    long_name: str

    def __init__(self, short_name: str, long_name: str) -> None:
        self.short_name = short_name
        self.long_name = long_name

    def __str__(self) -> str:
        return self.short_name

    def is_singleton(self) -> bool:
        return False


class BMGElementType:
    short_name: str
    long_name: str

    def __init__(self, short_name: str, long_name: str) -> None:
        self.short_name = short_name
        self.long_name = long_name


bool_element = BMGElementType("B", "bool")
natural_element = BMGElementType("N", "natural")
probability_element = BMGElementType("P", "probability")
positive_real_element = BMGElementType("R+", "positive real")
negative_real_element = BMGElementType("R-", "negative real")
real_element = BMGElementType("R", "real")


class BMGMatrixType(BMGLatticeType):
    element_type: BMGElementType
    rows: int
    columns: int

    def __init__(
        self,
        element_type: BMGElementType,
        short_name: str,
        long_name: str,
        rows: int,
        columns: int,
    ) -> None:
        BMGLatticeType.__init__(self, short_name, long_name)
        self.element_type = element_type
        self.rows = rows
        self.columns = columns

    @abstractmethod
    def with_dimensions(self, rows: int, columns: int) -> "BMGMatrixType":
        pass

    def with_size(self, size: Size) -> "BMGMatrixType":
        # We store the values for a matrix in a tensor; tensors are
        # row-major.  But the BMG type system expects a column-major
        # matrix. Here we get the rows and columns of the tensor size,
        # and swap them to make the column-major type.
        r, c = _size_to_rc(size)
        return self.with_dimensions(c, r)

    def is_singleton(self) -> bool:
        return (isinstance(self, BMGMatrixType)) and (self.rows * self.columns == 1)


class BroadcastMatrixType(BMGMatrixType):
    def __init__(self, element_type: BMGElementType, rows: int, columns: int) -> None:
        short_name = (
            element_type.short_name
            if rows == 1 and columns == 1
            else f"M{element_type.short_name}[{rows},{columns}]"
        )
        long_name = (
            element_type.long_name
            if rows == 1 and columns == 1
            else f"{rows} x {columns} {element_type.long_name} matrix"
        )
        BMGMatrixType.__init__(self, element_type, short_name, long_name, rows, columns)


# Note that all matrix type constructors are memoized in their initializer
# arguments; that way we ensure that any two MR[2,2]s are reference equal,
# which is a nice property to have.


class BooleanMatrix(BroadcastMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BroadcastMatrixType.__init__(self, bool_element, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return BooleanMatrix(rows, columns)


class NaturalMatrix(BroadcastMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BroadcastMatrixType.__init__(self, natural_element, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return NaturalMatrix(rows, columns)


class ProbabilityMatrix(BroadcastMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BroadcastMatrixType.__init__(self, probability_element, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return ProbabilityMatrix(rows, columns)


class PositiveRealMatrix(BroadcastMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BroadcastMatrixType.__init__(self, positive_real_element, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return PositiveRealMatrix(rows, columns)


class NegativeRealMatrix(BroadcastMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BroadcastMatrixType.__init__(self, negative_real_element, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return NegativeRealMatrix(rows, columns)


class RealMatrix(BroadcastMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BroadcastMatrixType.__init__(self, real_element, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return RealMatrix(rows, columns)


class SimplexMatrix(BMGMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        BMGMatrixType.__init__(
            self,
            probability_element,
            f"S[{rows},{columns}]",
            f"{rows} x {columns} simplex matrix",
            rows,
            columns,
        )

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return SimplexMatrix(rows, columns)


class OneHotMatrix(BMGMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        short_name = "OH" if rows == 1 and columns == 1 else f"OH[{rows},{columns}]"
        long_name = (
            "one-hot"
            if rows == 1 and columns == 1
            else f"{rows} x {columns} one-hot matrix"
        )
        BMGMatrixType.__init__(self, bool_element, short_name, long_name, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return OneHotMatrix(rows, columns)


class ZeroMatrix(BMGMatrixType, metaclass=MemoizedClass):
    def __init__(self, rows: int, columns: int) -> None:
        short_name = "Z" if rows == 1 and columns == 1 else f"Z[{rows},{columns}]"
        long_name = (
            "zero" if rows == 1 and columns == 1 else f"{rows} x {columns} zero matrix"
        )
        BMGMatrixType.__init__(self, bool_element, short_name, long_name, rows, columns)

    def with_dimensions(self, rows: int, columns: int) -> BMGMatrixType:
        return ZeroMatrix(rows, columns)


bottom = BMGLatticeType("bottom", "bottom")
One = OneHotMatrix(1, 1)
Zero = ZeroMatrix(1, 1)
Boolean = BooleanMatrix(1, 1)
Natural = NaturalMatrix(1, 1)
Probability = ProbabilityMatrix(1, 1)
PositiveReal = PositiveRealMatrix(1, 1)
NegativeReal = NegativeRealMatrix(1, 1)
Real = RealMatrix(1, 1)
Tensor = BMGLatticeType("T", "tensor")
top = Tensor
# This is not a real lattice type; rather, this is a marker to indicate
# that the node cannot have a lattice type assigned to it in the first
# place because we lack BMG typing rules for it.
Untypable = BMGLatticeType("U", "untypeable")


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

For matrix types with a single column and any number of rows r,
and columns c, the type lattice is:

      T                      (Tensor unsupported by BMG)
      |
    MR[r,c]                  (Real matrix)
    |     |
    |      MR+[r,c]          (Positive real matrix)
MR-[r,c]   |     |           (Negative real matrix)
    |      |     |
    | MN[r,c]    |           (Natural matrix)
    |      |   MP[r,c]       (Probability matrix)
    |      |   |    |
    |    MB[r,c]    |        (Boolean matrix)
    |    |    |   S[r,c]     (Row-simplex matrix)
    |    |    |   |
    |    |  OH[r,c]          (One-hot matrix)
    |    |    |
    Z[r,c]    |              (All-zero matrix)
         |    |
         BOTTOM              (the bottom type)

OH -- one-hot -- is not a type in the BMG type system; we use
it only when analyzing the accumulated graph. The OH type is
used to track the situation where a constant matrix can be
converted to both a Boolean and a simplex matrix; if the
rows are "one hot" -- all false (or 0) except for one true
(or 1) -- then the matrix is convertable to both Boolean and
simplex.

Similarly, Z -- the all-zero matrix -- is not a type in the BMG
type system. We use it to track cases where a matrix is convertible
to both Boolean and negative real.

Similarly, T (tensor) -- the top type -- is not found in the BMG
type system. The top type is the type more general than all other
types, and is used for situations such as attempting to resolve
situations such as "what is the type that is more general than both
a 1x2 matrix and a 2x2 matrix?" or to represent matrix types not
supported in BMG such as 3-dimensional matrices.

The BOTTOM type is the type that has no values; it is similarly
used as a convenience when you need a type more specific than any
other type; it is not in the BMG type system.

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
requires that its input be "exactly P". A "to real" node requires
that its input be "R or smaller".

Some nodes however introduce more complex requirements. The requirements
of a multiplication node, for instance, are:

* the input types must be greater than or equal to P
* the input types must be exactly the same

We do not have a requirement object for either "greater than or equal", or
for "this edge must be the same as that edge". And we have one more
factor to throw in: the output type of a multiplication is the same as its
input types, so we wish to find the *smallest possible* restriction on the
input types so that the multiplication's type is minimized.

(That is: if we have a multiplication of a natural by a probability, we do
not wish to require that both be converted to reals, because then the
multiplication node could not be used in a context where a positive real
was required. We want the smallest type that works: positive real.)

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

# This is a map from (class, class) to (int, int)=>BMGMatrix
_lookup_table = None


def _lookup():
    global _lookup_table
    if _lookup_table is None:
        R = Real.__class__
        RP = PositiveReal.__class__
        RN = NegativeReal.__class__
        P = Probability.__class__
        S = SimplexMatrix(1, 1).__class__
        N = Natural.__class__
        B = Boolean.__class__
        OH = One.__class__
        Z = Zero.__class__
        _lookup_table = {
            (R, R): RealMatrix,
            (R, RP): RealMatrix,
            (R, RN): RealMatrix,
            (R, P): RealMatrix,
            (R, S): RealMatrix,
            (R, N): RealMatrix,
            (R, B): RealMatrix,
            (R, OH): RealMatrix,
            (R, Z): RealMatrix,
            (RP, R): RealMatrix,
            (RP, RP): PositiveRealMatrix,
            (RP, RN): RealMatrix,
            (RP, P): PositiveRealMatrix,
            (RP, S): PositiveRealMatrix,
            (RP, N): PositiveRealMatrix,
            (RP, B): PositiveRealMatrix,
            (RP, OH): PositiveRealMatrix,
            (RP, Z): PositiveRealMatrix,
            (RN, R): RealMatrix,
            (RN, RP): RealMatrix,
            (RN, RN): NegativeRealMatrix,
            (RN, P): RealMatrix,
            (RN, S): RealMatrix,
            (RN, N): RealMatrix,
            (RN, B): RealMatrix,
            (RN, OH): RealMatrix,
            (RN, Z): NegativeRealMatrix,
            (P, R): RealMatrix,
            (P, RP): PositiveRealMatrix,
            (P, RN): RealMatrix,
            (P, P): ProbabilityMatrix,
            (P, S): ProbabilityMatrix,
            (P, N): PositiveRealMatrix,
            (P, B): ProbabilityMatrix,
            (P, OH): ProbabilityMatrix,
            (P, Z): ProbabilityMatrix,
            (S, R): RealMatrix,
            (S, RP): PositiveRealMatrix,
            (S, RN): RealMatrix,
            (S, P): ProbabilityMatrix,
            (S, S): SimplexMatrix,
            (S, N): PositiveRealMatrix,
            (S, B): ProbabilityMatrix,
            (S, OH): SimplexMatrix,
            (S, Z): ProbabilityMatrix,
            (N, R): RealMatrix,
            (N, RP): PositiveRealMatrix,
            (N, RN): RealMatrix,
            (N, P): PositiveRealMatrix,
            (N, S): PositiveRealMatrix,
            (N, N): NaturalMatrix,
            (N, B): NaturalMatrix,
            (N, OH): NaturalMatrix,
            (N, Z): NaturalMatrix,
            (B, R): RealMatrix,
            (B, RP): PositiveRealMatrix,
            (B, RN): RealMatrix,
            (B, P): ProbabilityMatrix,
            (B, S): ProbabilityMatrix,
            (B, N): NaturalMatrix,
            (B, B): BooleanMatrix,
            (B, OH): BooleanMatrix,
            (B, Z): BooleanMatrix,
            (OH, R): RealMatrix,
            (OH, RP): PositiveRealMatrix,
            (OH, RN): RealMatrix,
            (OH, P): ProbabilityMatrix,
            (OH, S): SimplexMatrix,
            (OH, N): NaturalMatrix,
            (OH, B): BooleanMatrix,
            (OH, OH): OneHotMatrix,
            (OH, Z): BooleanMatrix,
            (Z, R): RealMatrix,
            (Z, RP): PositiveRealMatrix,
            (Z, RN): NegativeRealMatrix,
            (Z, P): ProbabilityMatrix,
            (Z, S): ProbabilityMatrix,
            (Z, N): NaturalMatrix,
            (Z, B): BooleanMatrix,
            (Z, OH): BooleanMatrix,
            (Z, Z): ZeroMatrix,
        }
    return _lookup_table


@memoize
def _supremum(t: BMGLatticeType, u: BMGLatticeType) -> BMGLatticeType:
    """Takes two BMG types; returns the smallest type that is
    greater than or equal to both."""
    assert t != Untypable and u != Untypable
    if t == u:
        return t
    if t == bottom:
        return u
    if u == bottom:
        return t
    if t == top or u == top:
        return top
    assert isinstance(t, BMGMatrixType)
    assert isinstance(u, BMGMatrixType)
    if t.rows != u.rows or t.columns != u.columns:
        return Tensor
    # If we've made it here, they are unequal types but have the
    # same dimensions, and both are matrix types.
    return _lookup()[(t.__class__, u.__class__)](t.rows, t.columns)


# We can extend the two-argument supremum function to any number of arguments:
def supremum(*ts: BMGLatticeType) -> BMGLatticeType:
    """Takes any number of BMG types; returns the smallest type that is
    greater than or equal to all of them."""
    result = bottom
    for t in ts:
        result = _supremum(result, t)
    return result


def is_convertible_to(source: BMGLatticeType, target: BMGLatticeType) -> bool:
    return _supremum(source, target) == target


simplex_precision = 1e-10


def _type_of_matrix(v: torch.Tensor) -> BMGLatticeType:
    elements = v.numel()

    # If we have tensor([]) then that is not useful as a value
    # or a matrix; just call it a tensor.
    if elements == 0:
        return Tensor

    # If we have a single element tensor no matter what its dimensionality,
    # treat it as a single value.

    if elements == 1:
        return type_of_value(float(v))  # pyre-fixme

    # We have more than one element. What's the shape?

    shape = v.shape
    dimensions = len(shape)

    # If we have more than two dimensions then we cannot make it a matrix.
    # CONSIDER: Suppose we have something like [[[10, 20]]]] which is 1 x 1 x 2.
    # We could reduce that to a 1 x 2 matrix if we needed to. We might discard
    # sizes on the right equal to one.

    # We have the rows and columns of the original tensor, which is row-major.
    # But in BMG, constant matrices are expressed in column-major form.
    # Therefore we swap rows and columns here.

    if dimensions > 2:
        return Tensor
    tensor_rows, tensor_cols = _size_to_rc(shape)

    # However, for the purposes of analysis below, we still do it row by
    # row because that is more convenient when working with tensors:
    v = v.view(tensor_rows, tensor_cols)

    c = tensor_rows
    r = tensor_cols

    # We've got the shape. What is the smallest type
    # that is greater than or equal to the smallest type of
    # all the elements?

    sup = supremum(*[type_of_value(element) for row in v for element in row])

    # We should get a 1x1 matrix out; there should be no way to get
    # top or bottom out.

    assert isinstance(sup, BMGMatrixType)
    assert sup.rows == 1
    assert sup.columns == 1

    if sup in {Real, PositiveReal, NegativeReal, Natural}:
        return sup.with_dimensions(r, c)

    # The only remaining possibilities are:
    #
    # * Every element was 0 -- sup is Zero
    # * Every element was 1 -- sup is One
    # * Every element was 0 or 1 -- sup is Boolean
    # * At least one element was between 0 and 1 -- sup is Probability
    #
    # In the first two cases, we might have a one-hot.
    # In the third case, it is possible that we have a simplex.

    assert sup in {Boolean, Zero, One, Probability}

    sums_to_one = all(abs(float(row.sum()) - 1.0) <= simplex_precision for row in v)
    if sums_to_one:
        if sup == Probability:
            return SimplexMatrix(r, c)
        return OneHotMatrix(r, c)

    # It is not a simplex or a one-hot. Is it a matrix of probabilities that
    # do not sum to one?

    if sup == Probability:
        return sup.with_dimensions(r, c)

    # The only remaining possibilities are all zeros, all ones,
    # or some mixture of zero and one.
    #
    # If we have all zeros then this could be treated as either a matrix
    # of Booleans or a matrix of negative reals, and we do not know which
    # we will need; matrix of zeros is the type smaller than both those,
    # so return it:

    if sup == Zero:
        return sup.with_dimensions(r, c)

    # The only remaining possibility is matrix of all ones, or matrix
    # of some zeros and some ones. Either way, the smallest type
    # left is matrix of Booleans.

    return BooleanMatrix(r, c)


def type_of_value(v: Any) -> BMGLatticeType:
    """This computes the smallest BMG type that a given value fits into."""
    if isinstance(v, torch.Tensor):
        return _type_of_matrix(v)
    if isinstance(v, bool):
        return One if v else Zero
    if isinstance(v, int):
        if v == 0:
            return Zero
        if v == 1:
            return One
        if v >= 2:
            return Natural
        return NegativeReal
    if isinstance(v, float):
        # TODO: add a range check to make sure it fits into the integer
        # size expected by BMG
        if v.is_integer():
            return type_of_value(int(v))
        if 0.0 <= v:
            if v <= 1.0:
                return Probability
            return PositiveReal
        return NegativeReal
    return Untypable


def is_zero(v: Any) -> bool:
    return type_of_value(v) == Zero


def is_one(v: Any) -> bool:
    return type_of_value(v) == One


def lattice_to_bmg(t: BMGLatticeType) -> BMGLatticeType:
    # There are situations where we have a Zero or One type in hand
    # but those are just used for type convertibility analysis;
    # we sometimes actually need a valid BMG type. In those
    # situations, choose Boolean.
    assert t is not Tensor
    assert t is not Untypable
    if isinstance(t, OneHotMatrix) or isinstance(t, ZeroMatrix):
        return BooleanMatrix(t.rows, t.columns)
    return t


# TODO: Move this to bmg_requirements.py

# We need to be able to express requirements on inputs;
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
#
# The fact that we have unified the types which mean "a single value
# of a given type" and "a 1x1 matrix of that type" leads to an unfortunate
# wrinkle: there are a small number of rare situations where we must
# distinguish between the two.  For example, it is bizarre to have the input
# tensor([1]) to a Dirichlet, but it is legal. When we generate the BMG code
# for that, we need to ensure that the corresponding constant node is created
# via add_constant_pos_matrix(), not add_constant_pos().
#
# Rather than change the type system so that it distinguishes more clearly
# between single values and 1x1 matrices, we will just add a "force it to
# be a matrix" requirement; the problem fixer can then use that to ensure
# that the correct node is generated.
#
# We also occasionally need to express that an input edge has no restriction
# on it whatsoever; we'll use a singleton object for that.

# We should never create a requirement of a "fake" type.

_invalid_requirement_types = {Zero, One, Untypable}

# TODO: Mark this as abstract
class BaseRequirement:
    short_name: str
    long_name: str

    def __init__(self, short_name: str, long_name: str) -> None:
        self.short_name = short_name
        self.long_name = long_name


# TODO: Memoize these, remove memoization of construction functions below.
class AnyRequirement(BaseRequirement):
    def __init__(self) -> None:
        BaseRequirement.__init__(self, "any", "any")


class UpperBound(BaseRequirement):
    bound: BMGLatticeType

    def __init__(self, bound: BMGLatticeType) -> None:
        assert bound not in _invalid_requirement_types
        self.bound = bound
        BaseRequirement.__init__(self, f"<={bound.short_name}", f"<={bound.long_name}")


class AlwaysMatrix(BaseRequirement):
    bound: BMGMatrixType

    def __init__(self, bound: BMGMatrixType) -> None:
        assert bound not in _invalid_requirement_types
        self.bound = bound
        # We won't bother to make these have a special representation
        # when we display requirements on edges in DOT.
        BaseRequirement.__init__(self, bound.short_name, bound.long_name)


Requirement = Union[BMGLatticeType, BaseRequirement]


@memoize
def upper_bound(bound: Requirement) -> BaseRequirement:
    if isinstance(bound, UpperBound):
        return bound
    if isinstance(bound, AlwaysMatrix):
        return upper_bound(bound.bound)
    if isinstance(bound, BMGLatticeType):
        return UpperBound(bound)
    assert isinstance(bound, AnyRequirement)
    return bound


@memoize
def always_matrix(bound: BMGMatrixType) -> Requirement:
    if bound.rows != 1 or bound.columns != 1:
        # No need for a special annotation if it already
        # is a multi-dimensional matrix.
        return bound
    return AlwaysMatrix(bound)


def requirement_to_type(r: Requirement) -> BMGLatticeType:
    if isinstance(r, UpperBound):
        return r.bound
    if isinstance(r, AlwaysMatrix):
        return r.bound
    assert isinstance(r, BMGLatticeType)
    return r


def must_be_matrix(r: Requirement) -> bool:
    """Does the requirement indicate that the edge must be a matrix?"""
    if isinstance(r, AnyRequirement):
        return False
    if isinstance(r, AlwaysMatrix):
        return True
    t = requirement_to_type(r)
    if isinstance(t, BMGMatrixType):
        return t.rows != 1 or t.columns != 1
    return False


def is_atomic(t: BMGLatticeType) -> bool:
    return (
        isinstance(t, BMGMatrixType)
        and t.rows == 1
        and t.columns == 1
        and not isinstance(t, SimplexMatrix)
    )
