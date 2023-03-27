# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from copy import deepcopy

import pytest
from gpytorch.kernels import LinearKernel, PeriodicKernel, RBFKernel, ScaleKernel
from sts.abcd.kernels import ChangeWindowABCDKernel
from sts.abcd.utils import is_base_kernel, is_kernel_type_eq, remove_redundancy
from sts.gp.kernels import ConstantKernel, WhiteNoiseKernel


@pytest.mark.parametrize(
    "k, is_base",
    [
        (WhiteNoiseKernel(noise=1e-4), True),
        (LinearKernel(), True),
        (ScaleKernel(RBFKernel()), True),
        (ScaleKernel(PeriodicKernel()), True),
        (ConstantKernel(constant=1.0), True),
        (ChangeWindowABCDKernel(ConstantKernel(constant=1.0), LinearKernel()), False),
        (RBFKernel(), False),
        (LinearKernel() + LinearKernel(), False),
    ],
)
def test_is_base_kernel_type_eq(k, is_base):
    assert is_base_kernel(deepcopy(k)) == is_base


@pytest.mark.parametrize(
    "k1, k2, is_equal",
    [
        (
            (WhiteNoiseKernel(noise=1e-3) + LinearKernel()) * ScaleKernel(RBFKernel()),
            ScaleKernel(RBFKernel()) * ScaleKernel(RBFKernel()),
            False,
        ),
        (ScaleKernel(RBFKernel()), RBFKernel(), False),
        (
            (WhiteNoiseKernel(noise=1e-3) + LinearKernel()) * ScaleKernel(RBFKernel()),
            (LinearKernel() + WhiteNoiseKernel(noise=1e-3)) * ScaleKernel(RBFKernel()),
            True,
        ),
        (
            ChangeWindowABCDKernel(ConstantKernel(0.1), LinearKernel()),
            ChangeWindowABCDKernel(ConstantKernel(1.0), LinearKernel()),
            True,
        ),
    ],
)
def test_is_kernel_type_eq(k1, k2, is_equal):
    assert is_kernel_type_eq(k1, k2) == is_equal


@pytest.mark.parametrize(
    "kernel_list",
    [
        [
            (WhiteNoiseKernel(noise=1e-3) + LinearKernel()) * ScaleKernel(RBFKernel()),
            ScaleKernel(RBFKernel()) * ScaleKernel(RBFKernel()),
            ScaleKernel(RBFKernel()),
            (LinearKernel() + WhiteNoiseKernel(noise=1e-3)) * ScaleKernel(RBFKernel()),
            ChangeWindowABCDKernel(ConstantKernel(0.1), LinearKernel()),
            ChangeWindowABCDKernel(ConstantKernel(1.0), LinearKernel()),
        ]
    ],
)
def test_remove_redundancy(kernel_list):
    no_duplicate_list = remove_redundancy(kernel_list)
    assert len(no_duplicate_list) == 4
