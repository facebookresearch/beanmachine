# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from beanmachine.ppl.utils.multidictionary import MultiDictionary


class MultiDictionaryTest(unittest.TestCase):
    def test_multidictionary(self) -> None:
        d = MultiDictionary()
        d.add(1, "alpha")
        d.add(1, "bravo")
        d.add(2, "charlie")
        d.add(2, "delta")
        self.assertEqual(2, len(d))
        self.assertEqual(2, len(d[1]))
        self.assertEqual(2, len(d[2]))
        self.assertEqual(0, len(d[3]))
        self.assertTrue("alpha" in d[1])
        self.assertTrue("alpha" not in d[2])
        expected = """
{1:{alpha,
bravo}
2:{charlie,
delta}}"""
        self.assertEqual(expected.strip(), str(d).strip())
