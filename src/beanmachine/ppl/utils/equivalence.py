# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defines partition, a helper function to partition a set into
equivalence classes by an equivalence relation."""
from collections import defaultdict
from typing import Callable, Iterable, List, Set, TypeVar


_T = TypeVar("T")
_K = TypeVar("K")


def partition_by_relation(
    items: Iterable[_T], relation: Callable[[_T, _T], bool]
) -> List[Set[_T]]:
    # This is a quadratic algorithm, but n is likely to be small.
    result = []
    for item in items:
        eqv = next(filter((lambda s: relation(next(iter(s)), item)), result), None)
        if eqv is None:
            eqv = set()
            result.append(eqv)
        eqv.add(item)
    return result


def partition_by_kernel(
    items: Iterable[_T], kernel: Callable[[_T], _K]
) -> List[Set[_T]]:
    d = defaultdict(set)
    for item in items:
        d[kernel(item)].add(item)
    return list(d.values())
