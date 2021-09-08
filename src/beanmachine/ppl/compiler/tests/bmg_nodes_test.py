# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

import torch
from beanmachine.ppl.compiler.bmg_nodes import (
    NormalNode,
    RealNode,
    MultiplicationNode,
    ConstantTensorNode,
    ConstantRealMatrixNode,
    MatrixMultiplicationNode,
)


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

    def test_ConstantTensorNode_1d(self) -> None:
        v42 = torch.tensor([42, 43])
        t42 = ConstantTensorNode(v42)
        self.assertEqual(t42.value[0], v42[0])
        self.assertEqual(t42.value[1], v42[1])
        self.assertEqual(v42.size(), torch.Size([2]))
        self.assertEqual(t42.size, v42.size())
        self.assertEqual(t42.support_size(), 1.0)
        self.assertEqual(list(t42.support()), [v42])

    def test_ConstantTensorNode_2d(self) -> None:
        v42 = torch.tensor([[42, 43], [44, 45]])
        t42 = ConstantTensorNode(v42)
        self.assertEqual(t42.value[0, 0], v42[0, 0])
        self.assertEqual(t42.value[1, 0], v42[1, 0])
        self.assertEqual(v42.size(), torch.Size([2, 2]))
        self.assertEqual(t42.size, v42.size())
        self.assertEqual(t42.support_size(), 1.0)
        self.assertEqual(list(t42.support()), [v42])

    def test_ConstantRealMatrixNode_2d(self) -> None:
        v42 = torch.tensor([[42, 43], [44, 45]])
        t42 = ConstantRealMatrixNode(v42)
        self.assertEqual(t42.value[0, 0], v42[0, 0])
        self.assertEqual(t42.value[1, 0], v42[1, 0])
        self.assertEqual(v42.size(), torch.Size([2, 2]))
        self.assertEqual(t42.size, v42.size())
        self.assertEqual(t42.support_size(), 1.0)
        self.assertEqual(list(t42.support()), [v42])

    def test_MatrixMultiplicationNode(self) -> None:
        v42 = torch.tensor([[42, 43], [44, 45]])
        mv = torch.mm(v42, v42)
        t42 = ConstantRealMatrixNode(v42)
        mt = MatrixMultiplicationNode(t42, t42)
        # Note: Unlike constants, we cannot inspect the value directly
        #  self.assertEqual(mt.value[0, 0], mv[0, 0])
        #  self.assertEqual(mt.value[1, 0], mv[1, 0])
        # but shortly we will be able to inspect the support
        self.assertEqual(v42.size(), torch.Size([2, 2]))
        self.assertEqual(mt.size, mv.size())
        self.assertEqual(mt.support_size(), 1.0)
        support_list = list(mt.support())
        self.assertEqual(len(support_list), 1)
        support = support_list[0]
        self.assertEqual(support[0, 0], mv[0, 0])
        self.assertEqual(support[0, 1], mv[0, 1])
        self.assertEqual(support[1, 0], mv[1, 0])
        self.assertEqual(support[1, 1], mv[1, 1])

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
