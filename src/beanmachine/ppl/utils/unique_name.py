# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A helper class to give unique names to a set of objects."""
from typing import Any, Callable, Dict, Optional


def make_namer(
    namer: Optional[Callable[[Any], str]] = None, prefix: str = ""
) -> Callable[[Any], str]:
    if namer is None:
        un = UniqueNames(prefix)
        return lambda x: un.name(x)
    else:
        return namer


class UniqueNames(object):
    _map: Dict[Any, str]
    _prefix: str

    def __init__(self, prefix: str = ""):
        self._map = {}
        self._prefix = prefix

    def name(self, o: Any) -> str:
        if o.__hash__ is None:
            # This can lead to a situation where two objects are given the
            # same name; if the object is named, then freed, and then a different
            # object is allocated at the same address, the ID will be re-used.
            # Ideally, the instance of UniqueNames should be longer-lived than
            # any of the named objects.
            o = "unhashable " + str(id(o))
        if o not in self._map:
            self._map[o] = self._prefix + str(len(self._map))
        return self._map[o]
