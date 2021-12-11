# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This is just a little wrapper class around a dictionary for quickly
# and easily counting how many of each item you've got.

from typing import Any, Dict


class ItemCounter:
    items: Dict[Any, int]

    def __init__(self) -> None:
        self.items = {}

    def add_item(self, item: Any) -> None:
        if item not in self.items:
            self.items[item] = 1
        else:
            self.items[item] = self.items[item] + 1

    def remove_item(self, item: Any) -> None:
        if item not in self.items:
            return
        count = self.items[item] - 1
        if count == 0:
            del self.items[item]
        else:
            assert count > 0
            self.items[item] = count
