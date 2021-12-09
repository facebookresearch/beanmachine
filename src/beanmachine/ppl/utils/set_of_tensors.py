# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from typing import Set, Tuple

from beanmachine.ppl.utils.memoize import tensor_to_tuple
from torch import Tensor, tensor

# When constructing the support of various nodes we often
# must remove duplicates from a set of possible values.
# Unfortunately, it is not easy to do so with torch tensors.
# This helper class implements a set of tensors using the same
# technique as is used in the function call memoizer: we encode
# the data in the tensor into a tuple with the same shape. The
# tuple implements hashing and equality correctly, so we can put
# it in a set.


class SetOfTensors(collections.abc.Set):
    _elements: Set[Tuple]

    def __init__(self, iterable):
        self._elements = set()
        for value in iterable:
            t = value if isinstance(value, Tensor) else tensor(value)
            self._elements.add(tensor_to_tuple(t))

    def __iter__(self):
        return (tensor(t) for t in self._elements)

    def __contains__(self, value):
        t = value if isinstance(value, Tensor) else tensor(value)
        return tensor_to_tuple(t) in self._elements

    def __len__(self):
        return len(self._elements)

    def __str__(self):
        return "\n".join(sorted(str(t) for t in self))
