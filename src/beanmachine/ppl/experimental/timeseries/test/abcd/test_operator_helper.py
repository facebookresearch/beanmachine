# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from copy import deepcopy

import pytest
from gpytorch.kernels import Kernel
from sts.abcd.expression import KernelExpression
from sts.abcd.operator_helpers import (
    traverse_bottom_up,
    traverse_dfs_add,
    traverse_dfs_cp,
    traverse_dfs_cw,
    traverse_dfs_mul,
    traverse_dfs_mul_const,
    traverse_dfs_replace,
    traverse_dfs_simplify_add,
    traverse_dfs_simplify_base,
    traverse_dfs_simplify_mul,
)
from sts.abcd.utils import BASE_KERNELS, remove_redundancy

KERNEL = (BASE_KERNELS[0] + BASE_KERNELS[1]) * BASE_KERNELS[1] * BASE_KERNELS[2]


@pytest.mark.parametrize(
    "k1, k2, num_kernels, num_dedup_kernels",
    [
        (KERNEL, BASE_KERNELS[3], 7, 5),
    ],
)
def test_traverse_dfs_add(k1, k2, num_kernels, num_dedup_kernels):
    # S->S+B
    kernel_exp = KernelExpression(k1)
    traverse_dfs_add(kernel_exp, deepcopy(k2))
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert isinstance(res[0], Kernel)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize(
    "k1, k2, num_kernels, num_dedup_kernels",
    [
        (KERNEL, BASE_KERNELS[3], 7, 3),
    ],
)
def test_traverse_dfs_mul(k1, k2, num_kernels, num_dedup_kernels):
    # S->S*B
    kernel_exp = KernelExpression(k1)
    traverse_dfs_mul(kernel_exp, deepcopy(k2))
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize(
    "k1, k2, num_kernels, num_dedup_kernels",
    [
        (KERNEL, BASE_KERNELS[3], 4, 4),
    ],
)
def test_traverse_dfs_replace(k1, k2, num_kernels, num_dedup_kernels):
    # B->B1
    kernel_exp = KernelExpression(k1)
    traverse_dfs_replace(kernel_exp, deepcopy(k2))
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize(
    "kernel, num_kernels, num_dedup_kernels",
    [
        (KERNEL, 7, 7),
    ],
)
def test_traverse_dfs_cp(kernel, num_kernels, num_dedup_kernels):
    # S->CP(S,S)
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_cp(kernel_exp)
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize(
    "kernel, num_kernels, num_dedup_kernels, num_all_kernels",
    [
        (KERNEL, 7, 7, 14),
    ],
)
def test_traverse_dfs_cw(kernel, num_kernels, num_dedup_kernels, num_all_kernels):
    # S->CW(S,S)
    kernel_exp1 = KernelExpression(kernel)
    traverse_dfs_cw(kernel_exp1, pos_constant_kernel=1)
    res1 = traverse_bottom_up(kernel_exp1)
    assert len(res1) == num_kernels
    res = remove_redundancy(res1)
    assert len(res) == num_dedup_kernels

    kernel_exp2 = KernelExpression(kernel)
    traverse_dfs_cw(kernel_exp2, pos_constant_kernel=-1)
    res2 = traverse_bottom_up(kernel_exp2)
    assert len(res2) == num_kernels
    res = remove_redundancy(res2)
    assert len(res) == num_dedup_kernels

    res = res1 + res2
    res = remove_redundancy(res)
    assert len(res) == num_all_kernels

    kernel_exp = KernelExpression(kernel)
    traverse_dfs_cw(kernel_exp, pos_constant_kernel=0)
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize(
    "k1, k2, num_kernels, num_dedup_kernels",
    [
        (KERNEL, BASE_KERNELS[3], 7, 3),
    ],
)
def test_traverse_dfs_mul_const(k1, k2, num_kernels, num_dedup_kernels):
    # S->S*(B+C)
    kernel_exp = KernelExpression(k1)
    traverse_dfs_mul_const(kernel_exp, deepcopy(k2))
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize(
    "k1, k2, num_kernels, num_dedup_kernels", [(KERNEL, BASE_KERNELS[3], 3, 3)]
)
def test_traverse_dfs_simplify_base(k1, k2, num_kernels, num_dedup_kernels):
    # S->B
    kernel_exp = KernelExpression(k1)
    traverse_dfs_simplify_base(kernel_exp, deepcopy(k2))
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize("kernel, num_kernels, num_dedup_kernels", [(KERNEL, 2, 2)])
def test_traverse_dfs_simplify_add(kernel, num_kernels, num_dedup_kernels):
    # S+S1->S
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_simplify_add(kernel_exp)
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels


@pytest.mark.parametrize("kernel, num_kernels, num_dedup_kernels", [(KERNEL, 8, 6)])
def test_traverse_dfs_simplify_mul(kernel, num_kernels, num_dedup_kernels):
    # S*S1->S
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_simplify_mul(kernel_exp)
    res = traverse_bottom_up(kernel_exp)
    assert len(res) == num_kernels
    res = remove_redundancy(res)
    assert len(res) == num_dedup_kernels
