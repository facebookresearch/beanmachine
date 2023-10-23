#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a simple implementation of an immutable, persistent map based on
the Hash Array Mapped Trie data structure described in Phil Bagwell's paper
_Ideal Hash Trees_."""

from typing import Tuple, Any


def _bit_count(mask: int) -> int:
    # Standard algorithm for counting number of 1 bits in an int.
    count = 0
    while mask != 0:
        mask &= mask - 1
        count += 1
    return count


def _bit_count_below(mask: int, bit: int) -> int:
    # How many 1 bits are there below the given bit?
    return _bit_count(mask & (0xFFFFFFFF >> (32 - bit)))


class Map32:
    """A Map32 is a sparse immutable map from the numbers 0 to 31 to Any."""

    # Values is a tuple of zero through 32 possible values.
    # Mask indicates which elements of the tuple correspond to which
    # keys.  For example, if we have the value tuple (10, 20, 30) and
    # the mask is binary number 0000_0100_0001_0010 then the keys are
    # the 1 bits (1, 4, 7), so this map is {1 : 10, 2 : 20, 7 : 30}
    _mask: int
    _values: Tuple

    def __init__(self, mask: int, values: Tuple) -> None:
        self._mask = mask
        self._values = values

    def __contains__(self, key: int) -> bool:
        # Implements the "in" and "not in" operators.
        return (0 <= key <= 31) and (self._mask & (1 << key) != 0)

    def __getitem__(self, key: int) -> Any:
        if key not in self:
            return None
        index = _bit_count_below(self._mask, key)
        assert len(self._values) > index
        return self._values[index]

    def __iter__(self):
        for key in range(0, 32):
            if key in self:
                yield key

    def insert(self, key: int, value: Any) -> "Map32":
        assert 0 <= key <= 31
        index = _bit_count_below(self._mask, key)
        if key in self:
            new_values = self._values[0:index] + (value,) + self._values[index + 1 :]
            new_mask = self._mask
            assert len(new_values) == len(self._values)
        else:
            new_mask = self._mask | (1 << key)
            new_values = self._values[0:index] + (value,) + self._values[index:]
            assert len(new_values) == len(self._values) + 1
        return Map32(new_mask, new_values)

    def __str__(self) -> str:
        return "{" + ",".join(f"{key}:{self[key]}" for key in self) + "}"


_empty_map32 = Map32(0, ())


def _hash_key_shift(key: Any, shift: int) -> int:
    # Extract five bits from the hash starting at shift to obtain a value
    # from 0 through 31.
    return (abs(hash(key)) >> shift) & 0x1F


class HAMTrie:
    """This is a simple implementation of an immutable, persistent map based on
    the Hash Array Mapped Trie data structure described in Phil Bagwell's paper
    _Ideal Hash Trees_."""

    # We have a trie with branching factor 32; each node is responsible for
    # 5 bits of the hash. The bit number of the bottom bit of those five is
    # given by _shift. _map is a sparse immutable map from integers 0-31 to
    # a map entry. An entry may be:
    # * A (key, value) tuple
    # * A [(key, value), ... ] bucket where every key has exactly the same hash.
    #   (These should be exceedingly rare as it requires unequal keys with
    #   identical hashes.)
    # * A subnode to handle the *next* five bits of the hash.
    _shift: int
    _map: Map32

    def __init__(self, shift: int, map32: Map32) -> None:
        self._shift = shift
        self._map = map32

    def _hash_key(self, key: Any) -> int:
        return _hash_key_shift(key, self._shift)

    def __contains__(self, key: Any) -> bool:
        # Implements the "in" and "not in" operators.
        h = self._hash_key(key)
        if h not in self._map:
            # No entry; that key is not in the
            return False
        entry = self._map[h]
        if isinstance(entry, HAMTrie):
            return key in entry
        if isinstance(entry, tuple):
            assert len(entry) == 2
            return key == entry[0]
        assert isinstance(entry, list)
        assert len(entry) >= 2
        return any(key == item[0] for item in entry)

    def __getitem__(self, key: Any) -> Any:
        h = self._hash_key(key)
        if h not in self._map:
            return None
        entry = self._map[h]
        if isinstance(entry, HAMTrie):
            return entry[key]
        if isinstance(entry, tuple):
            return entry[1] if key == entry[0] else None
        assert isinstance(entry, list)
        assert len(entry) >= 2
        for k, v in entry:
            if k == key:
                return v
        return None

    def __iter__(self):
        for index in self._map:
            entry = self._map[index]
            if isinstance(entry, HAMTrie):
                for key in entry:
                    yield key
            elif isinstance(entry, tuple):
                yield entry[0]
            elif isinstance(entry, list):
                for item in entry:
                    yield item[0]
            else:
                raise AssertionError("Unexpected entry in HAMTrie")

    def insert(self, key: Any, value: Any) -> "HAMTrie":
        h = self._hash_key(key)
        # We've obtained 5 bits of the hash.
        if h not in self._map:
            # If we have no entry in the map for that portion of the hash,
            # then we can insert a key-value pair at that entry.
            new_map = self._map.insert(h, (key, value))
            return HAMTrie(self._shift, new_map)

        # We already have something in that slot. What?
        entry = self._map[h]
        if isinstance(entry, HAMTrie):
            # It's a subnode. Recursively insert the key-value pair into
            # the subnode.
            new_map = self._map.insert(h, entry.insert(key, value))
            return HAMTrie(self._shift, new_map)

        if isinstance(entry, tuple):
            assert len(entry) == 2
            existing_key, existing_value = entry
            if existing_key == key:
                # If the existing key and the new key are the same, replace
                # the value.
                new_map = self._map.insert(h, (key, value))
                return HAMTrie(self._shift, new_map)

            if abs(hash(key)) == abs(hash(existing_key)):
                # We have unequal keys with identical hashes. This should
                # be rare if hashes are well distributed. Make a bucket of
                # key-value pairs.
                new_map = self._map.insert(
                    h, [(key, value), (existing_key, existing_value)]
                )
                return HAMTrie(self._shift, new_map)
            # We have two keys that collide on these five bits of the hash
            # but not on all bits of the hash. Make a subnode.
            new_node_shift = self._shift + 5
            new_node = (
                HAMTrie(new_node_shift, _empty_map32)
                .insert(existing_key, existing_value)
                .insert(key, value)
            )
            new_map = self._map.insert(h, new_node)
            return HAMTrie(self._shift, new_map)

        assert isinstance(entry, list)
        assert len(entry) >= 2

        for i in range(len(entry)):
            existing_key, existing_value = entry[i]
            if existing_key == key:
                # If the existing key and the new key are the same, replace
                # the value.
                new_list = entry[0:i] + [(key, value)] + entry[i + 1 :]
                new_map = self._map.insert(h, new_list)
                return HAMTrie(self._shift, new_map)

        # The key was unequal to every key in the bucket. Does this key have the
        # same hash as the first item in the bucket? (Remember, every key in the
        # bucket has the same hash so we only need to check one.)

        existing_key = entry[0][0]
        if abs(hash(key)) == abs(hash(existing_key)):
            new_map = self._map.insert(h, entry + [(key, value)])
            return HAMTrie(self._shift, new_map)

        # The key is unequal to every key in the bucket and has a different hash.
        # Make a subnode.
        new_node_shift = self._shift + 5
        existing_hash = _hash_key_shift(existing_key, new_node_shift)
        new_node_map = _empty_map32.insert(existing_hash, entry)
        new_node = HAMTrie(new_node_shift, new_node_map).insert(key, value)
        new_map = self._map.insert(h, new_node)
        return HAMTrie(self._shift, new_map)

    def __str__(self) -> str:
        return "{" + ",".join(f"{key}:{self[key]}" for key in sorted(self)) + "}"


empty_hamtrie = HAMTrie(0, _empty_map32)
