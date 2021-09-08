# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

import torch
from beanmachine.ppl.compiler.bmg_nodes import NormalNode, RealNode, MultiplicationNode


class BMGNodesTest(unittest.TestCase):
    def test_RealNode(self) -> None:
        r42 = RealNode(42.0)
        self.assertEqual(r42.value, 42.0)
        self.assertEqual(r42.size, torch.Size([]))
        self.assertEqual(r42.support_size(), 1.0)
        self.assertEqual(list(r42.support()), [42.0])

    def test_MultiplicationNode(self) -> None:
        r2 = RealNode(2.0)
        r3 = RealNode(3.0)
        rx = MultiplicationNode([r2, r3])
        self.assertEqual(rx.size, torch.Size([]))
        self.assertEqual(rx.support_size(), 1.0)
        self.assertEqual(list(rx.support()), [6.0])

    def test_inputs_and_outputs(self) -> None:
        # We must maintain the invariant that the output set and the
        # input set of every node are consistent even when the graph
        # is edited.

        r1 = RealNode(1.0)
        self.assertEqual(len(r1.outputs.items), 0)
        n = NormalNode(r1, r1)
        # r1 has two outputs, both equal to n
        self.assertEqual(r1.outputs.items[n], 2)
        r2 = RealNode(2.0)
        n.inputs[0] = r2
        # r1 and r2 now each have one output
        self.assertEqual(r1.outputs.items[n], 1)
        self.assertEqual(r2.outputs.items[n], 1)
