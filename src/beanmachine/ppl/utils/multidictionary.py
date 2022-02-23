# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Set


class MultiDictionary:
    """A simple append-only multidictionary; values are deduplicated
    and must be hashable."""

    _d: Dict[Any, Set[Any]]

    def __init__(self) -> None:
        self._d = {}

    def add(self, key: Any, value: Any) -> None:
        if key not in self._d:
            self._d[key] = {value}
        else:
            self._d[key].add(value)

    def __getitem__(self, key: Any) -> Set[Any]:
        return self._d[key] if key in self else set()

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __contains__(self, key: Any):
        return key in self._d

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def __repr__(self) -> str:
        return (
            "{"
            + "\n".join(
                str(key) + ":{" + ",\n".join(sorted(str(v) for v in self[key])) + "}"
                for key in self
            )
            + "}"
        )
