# Copyright (c) Facebook, Inc. and its affiliates.

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
