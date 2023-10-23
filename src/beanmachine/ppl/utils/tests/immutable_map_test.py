# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for immutable_map.py"""
import unittest

from beanmachine.ppl.utils.immutable_map import (
    _bit_count_below,
    _empty_map32,
    empty_hamtrie,
)


class ImmutableMapTest(unittest.TestCase):
    def test_bit_count(self) -> None:
        self.assertEqual(_bit_count_below(11, 0), 0)
        self.assertEqual(_bit_count_below(11, 1), 1)
        self.assertEqual(_bit_count_below(11, 2), 2)
        self.assertEqual(_bit_count_below(11, 3), 2)
        self.assertEqual(_bit_count_below(11, 4), 3)
        self.assertEqual(_bit_count_below(11, 5), 3)

    def test_map32(self) -> None:
        m0 = _empty_map32
        m1 = m0.insert(1, 10)
        m2 = m1.insert(30, 300)
        m3 = m2.insert(11, 110)
        # Map32 is immutable; inserting does not change previous map.
        self.assertEqual("{}", str(m0))
        self.assertEqual("{1:10}", str(m1))
        self.assertEqual("{1:10,30:300}", str(m2))
        self.assertEqual("{1:10,11:110,30:300}", str(m3))
        self.assertTrue(0 not in m0)
        self.assertTrue(0 not in m1)
        self.assertTrue(0 not in m2)
        self.assertTrue(0 not in m3)
        self.assertTrue(1 not in m0)
        self.assertTrue(1 in m1)
        self.assertTrue(1 in m2)
        self.assertTrue(1 in m3)
        self.assertTrue(30 not in m0)
        self.assertTrue(30 not in m1)
        self.assertTrue(30 in m2)
        self.assertTrue(30 in m3)
        self.assertTrue(11 not in m0)
        self.assertTrue(11 not in m1)
        self.assertTrue(11 not in m2)
        self.assertTrue(11 in m3)
        self.assertEqual(m0[0], None)
        self.assertEqual(m1[0], None)
        self.assertEqual(m2[0], None)
        self.assertEqual(m3[0], None)
        self.assertEqual(m0[1], None)
        self.assertEqual(m1[1], 10)
        self.assertEqual(m2[1], 10)
        self.assertEqual(m3[1], 10)
        self.assertEqual(m0[30], None)
        self.assertEqual(m1[30], None)
        self.assertEqual(m2[30], 300)
        self.assertEqual(m3[30], 300)
        self.assertEqual(m0[11], None)
        self.assertEqual(m1[11], None)
        self.assertEqual(m2[11], None)
        self.assertEqual(m3[11], 110)

    def test_hamtrie(self) -> None:
        # Let's start with some simple black box tests.
        m0 = empty_hamtrie
        m1 = m0.insert("a", "alpha")
        m2 = m1.insert("b", "bravo")
        m3 = m2.insert("c", "charlie")
        self.assertEqual("{}", str(m0))
        self.assertEqual("{a:alpha}", str(m1))
        self.assertEqual("{a:alpha,b:bravo}", str(m2))
        self.assertEqual("{a:alpha,b:bravo,c:charlie}", str(m3))
        self.assertTrue("a" not in m0)
        self.assertTrue("a" in m1)
        self.assertTrue("a" in m2)
        self.assertTrue("a" in m3)
        self.assertTrue("b" not in m0)
        self.assertTrue("b" not in m1)
        self.assertTrue("b" in m2)
        self.assertTrue("b" in m3)
        self.assertTrue("c" not in m0)
        self.assertTrue("c" not in m1)
        self.assertTrue("c" not in m2)
        self.assertTrue("c" in m3)
        self.assertTrue("d" not in m0)
        self.assertTrue("d" not in m1)
        self.assertTrue("d" not in m2)
        self.assertTrue("d" not in m3)
        self.assertEqual(m0["a"], None)
        self.assertEqual(m1["a"], "alpha")
        self.assertEqual(m2["a"], "alpha")
        self.assertEqual(m3["a"], "alpha")
        self.assertEqual(m0["b"], None)
        self.assertEqual(m1["b"], None)
        self.assertEqual(m2["b"], "bravo")
        self.assertEqual(m3["b"], "bravo")
        self.assertEqual(m0["c"], None)
        self.assertEqual(m1["c"], None)
        self.assertEqual(m2["c"], None)
        self.assertEqual(m3["c"], "charlie")
        self.assertEqual(m0["d"], None)
        self.assertEqual(m1["d"], None)
        self.assertEqual(m2["d"], None)
        self.assertEqual(m3["d"], None)

        # Now force some rare code paths with partial and total hash
        # collisions:

        class H:
            n: str
            h: int

            def __init__(self, n: str, h: int) -> None:
                self.n = n
                self.h = h

            def __hash__(self) -> int:
                return self.h

            def __str__(self) -> str:
                return self.n

            def __lt__(self, other) -> bool:
                # HAMTrie.__str__ sorts by key so that
                # we get the same output for each test.
                return self.n < other.n

        ha = H("a", 0x3F)  # 00001 11111
        hb = H("b", 0x7F)  # 00011 11111 Collides on bottom 5 bits only
        hc = H("c", 0x3F)  # 00001 11111 Collides on all bits
        hd = H("d", 0x3F)  # 00001 11111 Collides on all bits

        m1 = m0.insert(ha, "alpha")
        m2 = m1.insert(hb, "bravo")
        m3 = m2.insert(hc, "charlie")

        self.assertEqual("{}", str(m0))
        self.assertEqual("{a:alpha}", str(m1))
        self.assertEqual("{a:alpha,b:bravo}", str(m2))
        self.assertEqual("{a:alpha,b:bravo,c:charlie}", str(m3))

        self.assertTrue(ha not in m0)
        self.assertTrue(ha in m1)
        self.assertTrue(ha in m2)
        self.assertTrue(ha in m3)
        self.assertTrue(hb not in m0)
        self.assertTrue(hb not in m1)
        self.assertTrue(hb in m2)
        self.assertTrue(hb in m3)
        self.assertTrue(hc not in m0)
        self.assertTrue(hc not in m1)
        self.assertTrue(hc not in m2)
        self.assertTrue(hc in m3)
        self.assertTrue(hd not in m0)
        self.assertTrue(hd not in m1)
        self.assertTrue(hd not in m2)
        self.assertTrue(hd not in m3)
