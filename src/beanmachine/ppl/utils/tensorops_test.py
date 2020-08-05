# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for tensorops.cpp"""
import unittest

import torch
from beanmachine.ppl.utils import tensorops


class TensorOpsTest(unittest.TestCase):
    def test_gradients(self) -> None:
        for type_ in [torch.float32, torch.float64]:
            x = torch.randn(3, requires_grad=True, dtype=type_)
            prec = torch.Tensor([[1, 0.1, 0], [0.1, 2, 0.5], [0, 0.5, 3]]).to(type_)
            mu = torch.randn(3, dtype=type_)
            # first gradient is `-(x - mu) @ prec`, second gradient is `- prec`
            f = -(x - mu) @ prec @ (x - mu) / 2
            grad, hess = tensorops.gradients(f, x)
            self.assertTrue(grad.allclose(-(x - mu) @ prec))
            self.assertTrue(hess.allclose(-prec))
            self.assertEqual(grad.dtype, type_, "gradient dtype must match input")
            self.assertEqual(hess.dtype, type_, "hessian dtype must match input")

    def test_simplex_gradients(self) -> None:
        for type_ in [torch.float32, torch.float64]:
            x = torch.randn(3, requires_grad=True, dtype=type_)
            prec = torch.Tensor([[1, 0.1, 0], [0.1, 2, 0.5], [0, 0.5, 3]]).to(type_)
            prec_diag = torch.Tensor([1.0, 1.9, 3.0]).to(type_)
            mu = torch.randn(3, dtype=type_)
            # first gradient is `-(x - mu) @ prec`, second gradient is `- prec`
            f = -(x - mu) @ prec @ (x - mu) / 2
            grad, hess = tensorops.simplex_gradients(f, x)
            self.assertTrue(grad.allclose(-(x - mu) @ prec))
            self.assertTrue(hess.allclose(-prec_diag))
            self.assertEqual(grad.dtype, type_, "gradient dtype must match input")
            self.assertEqual(hess.dtype, type_, "hessian dtype must match input")

    def test_halfspace_gradients(self) -> None:
        for type_ in [torch.float32, torch.float64]:
            x = torch.randn(3, requires_grad=True, dtype=type_)
            prec = torch.Tensor([[1, 0.1, 0], [0.1, 2, 0.5], [0, 0.5, 3]]).to(type_)
            prec_diag = torch.Tensor([1.0, 2.0, 3.0]).to(type_)
            mu = torch.randn(3, dtype=type_)
            # first gradient is `-(x - mu) @ prec`, second gradient is `- prec`
            f = -(x - mu) @ prec @ (x - mu) / 2
            grad, hess = tensorops.halfspace_gradients(f, x)
            self.assertTrue(grad.allclose(-(x - mu) @ prec))
            self.assertTrue(hess.allclose(-prec_diag))
            self.assertEqual(grad.dtype, type_, "gradient dtype must match input")
            self.assertEqual(hess.dtype, type_, "hessian dtype must match input")

    def test_gradients_negative(self) -> None:
        # output must have one element
        x = torch.randn(3, requires_grad=True)
        with self.assertRaises(ValueError) as cm:
            tensorops.gradients(2 * x, x)
        self.assertTrue(
            "output tensor must have exactly one element" in str(cm.exception)
        )
