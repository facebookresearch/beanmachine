# Copyright (c) Facebook, Inc. and its affiliates.

import collections
from typing import List

from torch import Tensor, tensor

# When constructing the support of various nodes we are often
# having to remove duplicates from a set of possible values.
# Unfortunately, it is not easy to do so with torch tensors.
# This helper class implements a set of tensors.

# TODO: This code is wrong and needs to be fixed. The "in" operator
# used in the code below throws an exception due to the way that
# tensors implement equality.  For example:
#
# tensor([1, 2]) in [ tensor([2, 1]) ]
#
# raises "Boolean value of Tensor with more than one value is ambiguous"
#
# We need to implement proper hashing and equality on tensors for the purposes
# of this set.
#
# We should also implement a sublinear set rather than using a list.


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
