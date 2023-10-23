# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List

import pytest
from gpytorch.kernels import AdditiveKernel, ProductKernel
from sts.abcd.expansion import expand_kernel
from sts.abcd.kernels import ChangePointABCDKernel, ChangeWindowABCDKernel
from sts.abcd.utils import BASE_KERNELS, is_kernel_type_eq


@pytest.mark.parametrize(
    "kernel",
    BASE_KERNELS,
)
def test_expand_base(kernel):
    res = expand_kernel(kernel)
    assert isinstance(res, List)
    # the number of kernels is always 22 if the kernel to expand is any one of the base kernels {WN, PER, RBF, LIN}
    assert len(res) == 22
    cnt = 0
    for k in res:
        # test change point/window kernels
        if isinstance(k, ChangePointABCDKernel) or isinstance(
            k, ChangeWindowABCDKernel
        ):
            cnt += 1
        # test the basic rules
        elif isinstance(k, AdditiveKernel) or isinstance(k, ProductKernel):
            assert is_kernel_type_eq(k.kernels[0], kernel) or is_kernel_type_eq(
                k.kernels[1], kernel
            )
    # there are 4 change kernel operators
    assert cnt == 4


@pytest.mark.parametrize(
    "kernel, heuristic_list",
    [
        (
            (BASE_KERNELS[0] + BASE_KERNELS[1]) * BASE_KERNELS[2] * BASE_KERNELS[1],
            [
                (BASE_KERNELS[0] + BASE_KERNELS[1]) * BASE_KERNELS[2],
                (BASE_KERNELS[0] + BASE_KERNELS[1]) * BASE_KERNELS[1],
                BASE_KERNELS[2] * BASE_KERNELS[1],
                (BASE_KERNELS[0] + BASE_KERNELS[1]),
                BASE_KERNELS[2],
                BASE_KERNELS[1],
            ],
        )
    ],
)
def test_expand_composite(kernel, heuristic_list):
    res = expand_kernel(kernel)
    assert isinstance(res, List)
    cp_cnt = 0
    heursitic_cnt = 0

    for k in res:
        # test change point/window kernel rules
        if isinstance(k, ChangePointABCDKernel) or isinstance(
            k, ChangeWindowABCDKernel
        ):
            cp_cnt += 1
        # test heuristic rules
        else:
            for kh in heuristic_list:
                if is_kernel_type_eq(k, kh):
                    heursitic_cnt += 1
                    break
    assert cp_cnt == 4
    assert len(heuristic_list) == heursitic_cnt
