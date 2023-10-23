# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
from gpytorch.kernels import PeriodicKernel, ScaleKernel
from sts.abcd.expansion import initialize_kernel, simplify
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.abcd.utils import is_kernel_type_eq
from sts.gp.kernels import ConstantKernel, WhiteNoiseKernel


@pytest.mark.parametrize(
    "kernel, sim_kernel",
    [
        (
            ChangePointABCDKernel(
                WhiteNoiseKernel(noise=1e-2) + WhiteNoiseKernel(noise=1e-3),
                WhiteNoiseKernel(noise=1e-2) + WhiteNoiseKernel(noise=1e-3),
            ),
            ChangePointABCDKernel(
                WhiteNoiseKernel(noise=1e-2),
                WhiteNoiseKernel(noise=1e-2),
            ),
        ),
        (
            ChangeWindowABCDKernel(
                WhiteNoiseKernel(noise=1e-2) + WhiteNoiseKernel(noise=1e-3),
                WhiteNoiseKernel(noise=1e-2) + WhiteNoiseKernel(noise=1e-3),
            ),
            ChangeWindowABCDKernel(
                WhiteNoiseKernel(noise=1e-2),
                WhiteNoiseKernel(noise=1e-2),
            ),
        ),
        (
            ChangePointABCDKernel(
                ChangePointABCDKernel(
                    WhiteNoiseKernel(noise=1e-2) * WhiteNoiseKernel(noise=1e-3),
                    WhiteNoiseKernel(noise=1e-3),
                ),
                ConstantKernel(constant=1.0) * ConstantKernel(constant=1.0),
            ),
            ChangePointABCDKernel(
                ChangePointABCDKernel(
                    WhiteNoiseKernel(noise=1e-2), WhiteNoiseKernel(noise=1e-2)
                ),
                ConstantKernel(constant=1.0),
            ),
        ),
        (
            ChangeWindowABCDKernel(
                ConstantKernel(constant=1.0) + ConstantKernel(constant=1.0),
                ConstantKernel(constant=1.0) + ConstantKernel(constant=1.0),
            ),
            ChangeWindowABCDKernel(
                ConstantKernel(constant=1.0), ConstantKernel(constant=1.0)
            ),
        ),
        (
            ChangePointABCDKernel(
                ConstantKernel(constant=1.0) * ConstantKernel(constant=1.0),
                ConstantKernel(constant=1.0) + ConstantKernel(constant=1.0),
            ),
            ChangePointABCDKernel(
                ConstantKernel(constant=1.0),
                ConstantKernel(constant=1.0),
            ),
        ),
        (
            WhiteNoiseKernel(noise=1e-1)
            + ConstantKernel(constant=1.0)
            * (WhiteNoiseKernel(noise=1e-2) + WhiteNoiseKernel(noise=1e-3)),
            WhiteNoiseKernel(noise=1e-1),
        ),
        (
            WhiteNoiseKernel(noise=1e-1)
            + WhiteNoiseKernel(noise=1e-1)
            + WhiteNoiseKernel(noise=1e-1)
            + WhiteNoiseKernel(noise=1e-1),
            WhiteNoiseKernel(noise=1e-1),
        ),
        (
            ConstantKernel(constant=100)
            + ConstantKernel(constant=10)
            + ConstantKernel(constant=1),
            ConstantKernel(constant=1.0),
        ),
        (
            ConstantKernel(constant=1e-1)
            + ConstantKernel(constant=1.0)
            * (ConstantKernel(constant=1e-2) + ConstantKernel(constant=1e-3)),
            ConstantKernel(constant=1.0),
        ),
        (
            ConstantKernel(constant=1.0)
            * ConstantKernel(constant=1.0)
            * ConstantKernel(constant=1.0),
            ConstantKernel(constant=1.0),
        ),
    ],
)
def test_simplify(kernel, sim_kernel):
    simplified_kernel = simplify(kernel)
    assert is_kernel_type_eq(simplified_kernel, sim_kernel)


@pytest.mark.parametrize(
    "kernel",
    [
        ScaleKernel(PeriodicKernel()),
    ],
)
def test_random_start(kernel):
    kernels_init = initialize_kernel(kernel, (1.0, 10.0))
    assert len(kernels_init) == 2
    assert kernels_init[0].base_kernel.period_length == 1.0
    assert kernels_init[1].base_kernel.period_length == 10.0
