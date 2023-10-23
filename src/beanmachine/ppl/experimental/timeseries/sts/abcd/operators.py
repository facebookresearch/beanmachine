# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Wrapper funtions for operators.
"""
from typing import List

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
from sts.abcd.utils import BASE_CONST_KERNELS, BASE_KERNELS, remove_redundancy


def S_to_S_add_B(kernel: Kernel) -> List[Kernel]:
    """
    Operate S->S+B for any subexpression and each base kernel.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """
    res = []
    for base_kernel in BASE_CONST_KERNELS:
        kernel_exp = KernelExpression(kernel)
        traverse_dfs_add(kernel_exp, base_kernel)
        res.extend(traverse_bottom_up(kernel_exp))
    res = remove_redundancy(res)
    return res


def S_to_S_mul_B(kernel: Kernel) -> List[Kernel]:
    """
    Operate S->S*B for any subexpression and each base kernel.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """

    res = []
    for base_kernel in BASE_KERNELS:
        kernel_exp = KernelExpression(kernel)
        traverse_dfs_mul(kernel_exp, base_kernel)
        res.extend(traverse_bottom_up(kernel_exp))
    res = remove_redundancy(res)
    return res


def B_to_B1(kernel: Kernel) -> List[Kernel]:
    """
    Operate B->B1 for each base kernel.
    :param kernel: The kernel to expand, B is any base kernel of it.
    :return: list of kernels proposed with this operator.
    """

    res = []
    for base_kernel in BASE_CONST_KERNELS:
        kernel_exp = KernelExpression(kernel)
        traverse_dfs_replace(kernel_exp, base_kernel)
        res.extend(traverse_bottom_up(kernel_exp))
    res = remove_redundancy(res)
    return res


def S_to_CP_S_S(kernel: Kernel) -> List[Kernel]:
    """
    Operate changepoint S->CP(S,S) for any subexpression.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_cp(kernel_exp)
    res = traverse_bottom_up(kernel_exp)
    res = remove_redundancy(res)
    return res


def S_to_CW_S(kernel: Kernel, pos_constant_kernel: int) -> List[Kernel]:
    """
    Operate changewindow S->CW for any subexpression.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :param pos_constant_kernel: if -1, S->CW(S, C);
                                else if 0, S->CW(S, S);
                                else, S->CW(C, S).
    :return: list of kernels proposed with this operator.
    """
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_cw(kernel_exp, pos_constant_kernel)
    res = traverse_bottom_up(kernel_exp)
    res = remove_redundancy(res)
    return res


def S_to_S_mul_B_add_C(kernel: Kernel) -> List[Kernel]:
    """
    Operate S->S*(B+C) for any subexpression and each base kernel.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """

    res = []
    for base_kernel in BASE_KERNELS:
        kernel_exp = KernelExpression(kernel)
        traverse_dfs_mul_const(kernel_exp, base_kernel)
        res.extend(traverse_bottom_up(kernel_exp))
    res = remove_redundancy(res)
    return res


def S_to_B(kernel: Kernel) -> List[Kernel]:
    """
    Simplify S->B for any subexpression and each base kernel.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """

    res = []
    for base_kernel in BASE_CONST_KERNELS:
        kernel_exp = KernelExpression(kernel)
        traverse_dfs_simplify_base(kernel_exp, base_kernel)
        res.extend(traverse_bottom_up(kernel_exp))
    res = remove_redundancy(res)
    return res


def S_add_S1_to_S(kernel: Kernel) -> List[Kernel]:
    """
    Simplify S+S1->S for any subexpression.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_simplify_add(kernel_exp)
    res = traverse_bottom_up(kernel_exp)
    res = remove_redundancy(res)
    return res


def S_mul_S1_to_S(kernel: Kernel) -> List[Kernel]:
    """
    Simplify S*S1->S for any subexpression.

    :param kernel: The kernel to expand, S is any subexpressoin of it.
    :return: list of kernels proposed with this operator.
    """
    kernel_exp = KernelExpression(kernel)
    traverse_dfs_simplify_mul(kernel_exp)
    res = traverse_bottom_up(kernel_exp)
    res = remove_redundancy(res)
    return res
