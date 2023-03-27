# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from copy import deepcopy

import pytest
import torch
from gpytorch.kernels import LinearKernel, PeriodicKernel, RBFKernel, ScaleKernel
from sts.abcd.expression import KernelExpression
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.abcd.utils import BASE_KERNELS, is_base_kernel
from sts.gp.kernels import ConstantKernel, WhiteNoiseKernel
from torch.testing import assert_allclose  # @manual

torch.manual_seed(0)
SAMPLE_DATA = torch.linspace(-3, 3, 100)


@pytest.mark.parametrize(
    "kernel",
    [
        deepcopy(BASE_KERNELS[0]),
        (deepcopy(BASE_KERNELS[0]) + deepcopy(BASE_KERNELS[1]))
        * deepcopy(BASE_KERNELS[2])
        * deepcopy(BASE_KERNELS[2]),
        (deepcopy(BASE_KERNELS[0]) * deepcopy(BASE_KERNELS[1]))
        + deepcopy(BASE_KERNELS[2])
        + deepcopy(BASE_KERNELS[2]),
        ChangeWindowABCDKernel(ConstantKernel(0.1), deepcopy(BASE_KERNELS[1])),
    ],
)
def test_kernel_expression(kernel):
    kernel_exp = KernelExpression(kernel)
    if is_base_kernel(kernel):
        assert kernel_exp.lhs is None
        assert kernel_exp.rhs is None

    else:
        if not is_base_kernel(kernel_exp.lhs.kernel):
            assert kernel_exp.lhs.lhs is not None and kernel_exp.lhs.rhs is not None
        if not is_base_kernel(kernel_exp.rhs.kernel):
            assert kernel_exp.rhs.lhs is not None and kernel_exp.rhs.rhs is not None


@pytest.mark.parametrize(
    "kernel, name",
    [
        (WhiteNoiseKernel(noise=1e-3), "WN"),
        (ConstantKernel(constant=1), "C"),
        (ScaleKernel(PeriodicKernel()), "PER"),
        (ScaleKernel(RBFKernel()), "RBF"),
        (LinearKernel(), "LIN"),
        (WhiteNoiseKernel(noise=1e-3) + ScaleKernel(PeriodicKernel()), "(WN + PER)"),
        (
            (WhiteNoiseKernel(noise=1e-3) + ScaleKernel(PeriodicKernel()))
            * LinearKernel(),
            "((WN + PER) x LIN)",
        ),
        (
            WhiteNoiseKernel(noise=1e-3) + ScaleKernel(RBFKernel()) * LinearKernel(),
            "(WN + (RBF x LIN))",
        ),
        (
            ChangePointABCDKernel(
                WhiteNoiseKernel(noise=1e-3), ScaleKernel(RBFKernel())
            )
            * LinearKernel(),
            "(CP(WN, RBF) x LIN)",
        ),
        (
            ChangeWindowABCDKernel(
                ScaleKernel(PeriodicKernel()), ScaleKernel(RBFKernel())
            )
            * LinearKernel(),
            "(CW(PER, RBF) x LIN)",
        ),
    ],
)
def test_kernel_repr(kernel, name):
    kernel_exp = KernelExpression(kernel)
    assert repr(kernel_exp) == name


@pytest.mark.parametrize(
    "kernel, x, kern_exp_new",
    [
        (
            (LinearKernel() + WhiteNoiseKernel(noise=1e-4))
            * ScaleKernel(PeriodicKernel())
            * (ScaleKernel(RBFKernel()) + ConstantKernel(constant=1.0)),
            SAMPLE_DATA,
            "(((((LIN x PER) x RBF) + ((LIN x PER) x C)) + ((WN x PER) x RBF)) + ((WN x PER) x C))",
        ),
        (
            ChangePointABCDKernel(
                (
                    ConstantKernel(constant=1.0)
                    + WhiteNoiseKernel(noise=1e-3)
                    + WhiteNoiseKernel(noise=2e-3)
                )
                * LinearKernel(),
                LinearKernel(),
            ),
            SAMPLE_DATA,
            "(((CP((C x LIN), C) + CP((WN x LIN), C)) + CP((WN x LIN), C)) + CP(C, LIN))",
        ),
    ],
)
def test_distributive_law(kernel, x, kern_exp_new):
    kernel_exp = KernelExpression(kernel)
    kernel_new = kernel_exp.additive_form_kernel()
    assert repr(KernelExpression(kernel_new)) == kern_exp_new
    assert_allclose(kernel(x).evaluate(), kernel_new(x).evaluate())
