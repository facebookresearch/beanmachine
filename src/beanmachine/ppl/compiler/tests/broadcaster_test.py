# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from beanmachine.ppl.compiler.broadcaster import broadcast_fnc
from torch import Size


class BroadcastTest(unittest.TestCase):
    def test_broadcast_success(self) -> None:
        input_sizes = [Size([3]), Size([3, 2, 1]), Size([1, 2, 1]), Size([2, 3])]
        expectations = [
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        ]
        target_size = Size([3, 2, 3])
        i = 0
        for input_size in input_sizes:
            broadcaster = broadcast_fnc(input_size, target_size)
            expectation = expectations[i]
            i = i + 1
            for j in range(0, 18):
                input_index = broadcaster(j)
                expected = expectation[j]
                self.assertEqual(input_index, expected)
