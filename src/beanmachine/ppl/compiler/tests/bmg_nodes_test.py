# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from beanmachine.ppl.compiler.sizer import Sizer
from beanmachine.ppl.compiler.support import ComputeSupport


def support(n):
    return str(ComputeSupport()[n])


def size(n):
    return Sizer()[n]


class BMGNodesTest(unittest.TestCase):
    def test_RealNode(self) -> None:
        r42 = RealNode(42.0)
        self.assertEqual(r42.value, 42.0)
        self.assertEqual(size(r42), torch.Size([]))
        # Note that support always returns a set of tensors, even though this
        # node is technically scalar valued. In practice we never need to compute
        # the support of a RealNode, so fixing this minor oddity is unnecessary.
        self.assertEqual(support(r42), "tensor(42.)")

    def test_MultiplicationNode(self) -> None:
        r2 = RealNode(2.0)
        r3 = RealNode(3.0)
        rx = MultiplicationNode([r2, r3])
        self.assertEqual(size(rx), torch.Size([]))
        self.assertEqual(support(rx), "tensor(6.)")

    def test_ConstantTensorNode_1d(self) -> None:
        v42 = torch.tensor([42, 43])
        t42 = ConstantTensorNode(v42)
        self.assertEqual(t42.value[0], v42[0])
        self.assertEqual(t42.value[1], v42[1])
        self.assertEqual(v42.size(), torch.Size([2]))
        self.assertEqual(size(t42), v42.size())
        self.assertEqual(support(t42), "tensor([42, 43])")

    def test_ConstantTensorNode_2d(self) -> None:
        v42 = torch.tensor([[42, 43], [44, 45]])
        t42 = ConstantTensorNode(v42)
        self.assertEqual(t42.value[0, 0], v42[0, 0])
        self.assertEqual(t42.value[1, 0], v42[1, 0])
        self.assertEqual(v42.size(), torch.Size([2, 2]))
        self.assertEqual(size(t42), v42.size())
        expected = """
tensor([[42, 43],
        [44, 45]])"""
        self.assertEqual(support(t42).strip(), expected.strip())

    def test_ConstantRealMatrixNode_2d(self) -> None:
        v42 = torch.tensor([[42, 43], [44, 45]])
        t42 = ConstantRealMatrixNode(v42)
        self.assertEqual(t42.value[0, 0], v42[0, 0])
        self.assertEqual(t42.value[1, 0], v42[1, 0])
        self.assertEqual(v42.size(), torch.Size([2, 2]))
        self.assertEqual(size(t42), v42.size())
        expected = """
tensor([[42, 43],
        [44, 45]])"""
        self.assertEqual(support(t42).strip(), expected.strip())

    def test_MatrixMultiplicationNode(self) -> None:
        v42 = torch.tensor([[42, 43], [44, 45]])
        mv = torch.mm(v42, v42)
        t42 = ConstantRealMatrixNode(v42)
        mt = MatrixMultiplicationNode(t42, t42)
        self.assertEqual(v42.size(), torch.Size([2, 2]))
        self.assertEqual(size(mt), mv.size())
        expected = """
tensor([[3656, 3741],
        [3828, 3917]])
"""
        self.assertEqual(support(mt).strip(), expected.strip())

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
